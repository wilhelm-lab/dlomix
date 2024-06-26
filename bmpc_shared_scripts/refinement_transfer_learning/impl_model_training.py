import yaml
import os
import uuid

import wandb
from wandb.integration.keras import WandbCallback

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

import change_layers
import freezing


def load_config(config_file):
    with open(config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config


def model_training(config):
    def run():
        config['run_id'] = uuid.uuid4()

        # initialize weights and biases
        wandb.init(
            project=config["project"],
            config=config,
            tags=[config['dataset']['name']]
        )

        if 'cuda_device_nr' in wandb.config['processing']:
            os.environ["CUDA_VISIBLE_DEVICES"] = wandb.config['processing']['cuda_device_nr']

        if 'num_proc' in wandb.config['processing']:
            num_proc = wandb.config['processing']['num_proc']
            os.environ["OMP_NUM_THREADS"] = f"{num_proc}"
            os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_proc}"
            os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_proc}"
            tf.config.threading.set_inter_op_parallelism_threads(
                num_proc
            )
            tf.config.threading.set_intra_op_parallelism_threads(
                num_proc
            )

        os.environ['HF_HOME'] = wandb.config['dataset']['hf_home']
        os.environ['HF_DATASETS_CACHE'] = wandb.config['dataset']['hf_cache']

        from dlomix.constants import PTMS_ALPHABET, ALPHABET_NAIVE_MODS, ALPHABET_UNMOD
        from dlomix.data import load_processed_dataset
        from dlomix.models import PrositIntensityPredictor
        from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance

        # load dataset
        dataset = load_processed_dataset(wandb.config['dataset']['processed_path'])



        # load or create model
        if 'load_path' in wandb.config['model']:
            print(f"loading model from file {wandb.config['model']['load_path']}")
            model = tf.keras.models.load_model(wandb.config['model']['load_path'])
        else:
            # initialize model
            input_mapping = {
                    "SEQUENCE_KEY": "modified_sequence",
                    "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
                    "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
                    "FRAGMENTATION_TYPE_KEY": "method_nbr",
                }

            meta_data_keys=["collision_energy_aligned_normed", "precursor_charge_onehot", "method_nbr"]

            # select alphabet
            if isinstance(wandb.config['dataset']['alphabet'], dict):
                # this is a custom alphabet
                alphabet = wandb.config['dataset']['alphabet']
            elif wandb.config['dataset']['alphabet'] == 'PTMS_ALPHABET':
                alphabet = PTMS_ALPHABET
            elif wandb.config['dataset']['alphabet'] == 'ALPHABET_UNMOD':
                alphabet = ALPHABET_UNMOD
            elif wandb.config['dataset']['alphabet'] == 'ALPHABET_NAIVE_MODS':
                alphabet = ALPHABET_NAIVE_MODS
            else:
                raise ValueError('unknown alphabet selected')

            model = PrositIntensityPredictor(
                seq_length=wandb.config['dataset']['seq_length'],
                alphabet=alphabet,
                use_prosit_ptm_features=False,
                with_termini=False,
                input_keys=input_mapping,
                meta_data_keys=meta_data_keys
            )


        # initialize relevant stuff for training
        total_epochs = wandb.config['training']['num_epochs']
        recompile_callbacks = [{
            "epoch": total_epochs,
            "callback": lambda *args: None
        }]

        # refinement/transfer learning configuration
        rl_config = wandb.config['refinement_transfer_learning']

        # optionally: replacing of input/output layers
        if 'new_output_layer' in rl_config:
            change_layers.change_output_layer(model, rl_config['new_output_layer']['num_ions'])
        if 'new_input_layer' in rl_config:
            change_layers.change_input_layer(
                model,
                rl_config['new_input_layer']['new_alphabet'],
                rl_config['new_input_layer']['freeze_old_weights']
            )
        
        # optionally: freeze layers during training
        if 'freeze_layers' in rl_config:
            if 'activate' not in rl_config['freeze_layers'] or rl_config['freeze_layers']['activate']:
                freezing.freeze_model(
                    model, 
                    rl_config['freeze_layers']['is_first_layer_trainable'],
                    rl_config['freeze_layers']['is_last_layer_trainable']
                )
                wandb.log({'freeze_layers': 1})

                def release_callback():
                    freezing.release_model(model)
                    wandb.log({'freeze_layers': 0})

                recompile_callbacks.append({
                    'epoch': rl_config['freeze_layers']['release_after_epochs'],
                    'callback': release_callback
                })


        class LearningRateReporter(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, *args):
                wandb.log({'learning_rate': self.model.optimizer._learning_rate.numpy()})

        callbacks = [WandbCallback(save_model=False, log_batch_frequency=True, verbose=1), LearningRateReporter()]

        if 'early_stopping' in wandb.config['training']:
            print("using early stopping")
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=wandb.config['training']['early_stopping']['min_delta'],
                patience=wandb.config['training']['early_stopping']['patience'],
                restore_best_weights=True)

            callbacks.append(early_stopping)

        if 'lr_scheduler_plateau' in wandb.config['training']:
            print("using lr scheduler plateau")
            # Reduce LR on Plateau Callback
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=wandb.config['training']['lr_scheduler_plateau']['factor'],
                patience=wandb.config['training']['lr_scheduler_plateau']['patience'],
                min_delta=wandb.config['training']['lr_scheduler_plateau']['min_delta'],
                cooldown=wandb.config['training']['lr_scheduler_plateau']['cooldown']
            ) 

            callbacks.append(reduce_lr)

        if 'lr_warmup_linear' in wandb.config['training']:
            print("using lr warmup linear")
            num_epochs = wandb.config['training']['lr_warmup_linear']['num_epochs']
            start_lr = wandb.config['training']['lr_warmup_linear']['start_lr']
            end_lr = wandb.config['training']['lr_warmup_linear']['end_lr']
            def scheduler(epoch, lr):
                if epoch < num_epochs:
                    print("warmup step")
                    factor = epoch / num_epochs
                    return factor * end_lr + (1-factor) * start_lr
                else:
                    return lr
            
            lr_warmup_linear = LearningRateScheduler(scheduler)
            callbacks.append(lr_warmup_linear)


        # restructure recompile callbacks to form training parts
        assert not any([x['epoch'] > total_epochs for x in recompile_callbacks])
        rcb_dict = {}
        for rcb in recompile_callbacks:
            rcb_dict.setdefault(rcb['epoch'], [])
            rcb_dict[rcb['epoch']].append(rcb['callback'])
        rcb_keys = sorted([int(x) for x in rcb_dict])
        current_epoch = 0
        training_parts = []
        for epoch_key in rcb_keys:
            training_parts.append({
                "num_epochs": epoch_key - current_epoch,
                "callbacks": rcb_dict[epoch_key]
            })
            current_epoch = epoch_key


        # perform all training runs
        current_learning_rate = wandb.config['training']['learning_rate']
        for training_part in training_parts:
            optimizer = tf.keras.optimizers.Adam(learning_rate=current_learning_rate)
            model.compile(
                optimizer=optimizer,
                loss=masked_spectral_distance,
                metrics=[masked_pearson_correlation_distance]
            )

            if training_part['num_epochs'] > 0:
                # train model
                model.fit(
                    dataset.tensor_train_data,
                    validation_data=dataset.tensor_val_data,
                    epochs=training_part['num_epochs'],
                    callbacks=callbacks
                )

            # call callbacks
            for cb in training_part['callbacks']:
                cb()
            
            current_learning_rate = model.optimizer._learning_rate.numpy()


        out_path = None
        if 'save_dir' in wandb.config['model']:
            out_path = f"{wandb.config['model']['save_dir']}/{wandb.config['dataset']['name']}/{wandb.config['run_id']}.keras"
        if 'save_path' in wandb.config['model']:
            out_path = wandb.config['model']['save_path']
        if out_path is not None:
            model.save(out_path)


        # finish up training process
        wandb.finish()

    return run



# combines nested dicts
# source: https://stackoverflow.com/questions/70310388/how-to-merge-nested-dictionaries/70310511#70310511
def combine_into(d: dict, combined: dict) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            combine_into(v, combined.setdefault(k, {}))
        else:
            combined[k] = v