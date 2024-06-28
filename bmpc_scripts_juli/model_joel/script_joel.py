import os
os.environ['HF_HOME'] = "/cmnfs/proj/prosit_astral"
os.environ['HF_DATASETS_CACHE'] = "/cmnfs/proj/prosit_astral/datasets"

print("[UNIMOD:1]-K[UNIMOD:1]".count('[UNIMOD:' + '1' + ']'))

import numpy as np
from dlomix.data import FragmentIonIntensityDataset
import pandas as pd

from datasets import disable_caching
#disable_caching()

from dlomix.constants import PTMS_ALPHABET
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
import tensorflow as tf
import yaml

print('='*32)
print('Conda info')
print(f"Environment: {os.environ['CONDA_DEFAULT_ENV']}")
print('='*32)
print('Tensorflow info')
print(f"Version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Number of GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"List of GPUs available: {tf.config.list_physical_devices('GPU')}")
print('='*32)


with open("./config.yaml", 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

model_settings = config['model_settings']
train_settings = config['train_settings']

print("DataLoader Settings:")
print(f"Dataset: {config['dataloader']['dataset']}")
print(f"Batch Size: {config['dataloader']['batch_size']}")

print("\nModel config:")
for key, value in model_settings.items():
    print(f"{key}: {value}")

print("\nTraining Settings:")
for key, value in train_settings.items():
    print(f"{key}: {value}")
print('='*32)


################################################
#                  Dataset                     #
################################################
from dlomix.data import load_processed_dataset

match config['dataloader']['dataset']:
    case 'small':
        train_data_source = "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_train.parquet"
        val_data_source =   "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_val.parquet"
        test_data_source =  "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_test.parquet"
    case 'full':
        train_data_source = "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug_train.parquet"
        val_data_source =   "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug_val.parquet"
        test_data_source =  "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug_test.parquet"

# Faster loading if dataset is already saved
if os.path.exists(config['dataloader']['save_path'] + '/dataset_dict.json') and (config['dataloader']['dataset'] != 'small'):
    int_data = load_processed_dataset(config['dataloader']['save_path'])
else:
    int_data = FragmentIonIntensityDataset(
        data_source=train_data_source,
        val_data_source=val_data_source,
        test_data_source=test_data_source,
        data_format="parquet", 
        val_ratio=0.2, 
        max_seq_len=30, 
        encoding_scheme="naive-mods",
        alphabet=PTMS_ALPHABET,
        model_features=["precursor_charge_onehot", "collision_energy_aligned_normed","method_nbr"],
        batch_size=config['dataloader']['batch_size']
    )
    int_data.save_to_disk(config['dataloader']['save_path'])


#################################################
#         Choose and compile model              #
#################################################

optimizer = tf.keras.optimizers.Adam(learning_rate=train_settings['lr_base'])
if config['model_type'] == 'ours':
    from models.models import TransformerModel

    print("Loading Transformer Model")

    if model_settings['prec_type'] not in ['embed_input', 'pretoken', 'inject']:
        raise ValueError("Invalid model setting for 'prec_type'")

    model = TransformerModel(**model_settings)

elif config['model_type'] == 'prosit_t':
    from models.prosit_t.models.prosit_transformer import PrositTransformerMean174M2 as PrositTransformer

    model = PrositTransformer(
        vocab_dict=PTMS_ALPHABET,
        **config['prosit']
    )

print("Compiling Model")
model.compile(optimizer=optimizer, 
            loss=masked_spectral_distance,
            metrics=[masked_pearson_correlation_distance])
inp = [m for m in int_data.tensor_train_data.take(1)][0][0]
out = model(inp)
model.summary()

###################################################
#                   Wandb init                    #
###################################################

import wandb
WandbCallback  = wandb.keras.WandbCallback
from wandb.keras import WandbCallback

import random  
from string import ascii_lowercase, ascii_uppercase, digits
chars = ascii_lowercase + ascii_uppercase + digits

name =  config['dataloader']['dataset'][0] + '_' + \
        model_settings['prec_type'] + '_' + \
        (str(model_settings['inject_pre'])[0] +  
        str(model_settings['inject_post'])[0] +
        model_settings['inject_position'] + '_') \
            if model_settings['prec_type']=='inject' else "" + \
        'd' + str(model_settings['depth']) + '_' + \
        train_settings['lr_method'] + '_' + \
        ''.join([random.choice(chars) for _ in range(3)])
name = f"%s_%s%s_d%s_%s_%s_%s" % ( 
    config['dataloader']['dataset'][0],
    model_settings['prec_type'],
    (str(model_settings['inject_pre'])[0] +  str(model_settings['inject_post'])[0] + model_settings['inject_position'])
        if model_settings['prec_type']=='inject' else "",
    model_settings['depth'],
    train_settings['lr_method'],
    train_settings['lr_base'],
    ''.join([random.choice(chars) for _ in range(3)])
)

tags = [
    config['dataloader']['dataset'],
    'depth_' + str(model_settings['depth']),
    'prec_type_' + model_settings['prec_type'],
    'lr_method_' + train_settings['lr_method'],
    'lr_base_' + str(train_settings['lr_base']),
    'lr_max_' + str(train_settings['lr_max']),
]
tags + [model_settings['inject_pre'], 
        model_settings['inject_post'], 
        model_settings['inject_position']] if model_settings['prec_type'] == 'inject' else []

if train_settings['log_wandb']:
    #wandb.login(key='xxxxx')
    wandb.init(
        project="astral",
        #name=name,
        tags=tags,
        config=config,
        entity='joellapin'
    )


#######################################################
#                      Callbacks                      #
#######################################################

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=20,
    restore_best_weights=True)

# ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=saved_models/best_model_intensity_nan.keras
save_best = ModelCheckpoint(
    'saved_models/best_model_intensity_nan.keras',
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)

class WarmupCooldownLR(tf.keras.callbacks.Callback):
    def __init__(self, 
        start_lr, end_lr, steps, step_start=0
    ):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.steps = steps
        self.step_start = step_start
        self.step_end = step_start + steps

        self.schedule = np.linspace(start_lr, end_lr, steps)
        self.ticker = 0

    def on_train_batch_begin(self, batch, *args):
        global_step = model.optimizer.variables[0].numpy()
        if (global_step >= self.step_start) & (global_step < self.step_end):
            self.model.optimizer._learning_rate.assign(self.schedule[self.ticker])
            self.ticker += 1

class DecayLR(tf.keras.callbacks.Callback):
    # A decay learning rate that decreases the learning rate an order of magnitude
    # every mag_drop_every_n_steps steps.
    def __init__(self,
        mag_drop_every_n_steps=100000,
        start_step=20000,
        end_step=1e10
    ):
        self.alpha = np.exp(np.log(0.1) / mag_drop_every_n_steps)
        self.start_step = start_step
        self.end_step = end_step
        self.ticker = 1

    def on_train_batch_begin(self, batch, *args):
        global_step = model.optimizer.variables[0].numpy()
        if (global_step >= self.start_step) & (global_step < self.end_step):
            current_lr = model.optimizer._learning_rate.numpy()
            new_lr  = current_lr * self.alpha
            model.optimizer._learning_rate.assign(new_lr)
            self.ticker += 1


class LearningRateReporter(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, *args):
        wandb.log({'learning_rate': self.model.optimizer._learning_rate.numpy()})
    
callbacks = []
if train_settings['log_wandb']:
    callbacks.append(WandbCallback(save_model=False))
    callbacks.append(LearningRateReporter())
if train_settings['lr_method'] == 'cyclic':
    callbacks.append(cyclicLR)
elif train_settings['lr_method'] == 'warmup':
    #callbacks.append(WarmupCooldownLR(1e-7, 1e-3, 10000))
    callbacks.append(WarmupCooldownLR(5e-4, 1e-4, step_start=60000, steps=100000))
elif train_settings['lr_method'] == 'decay':
    callbacks.append(DecayLR(mag_drop_every_n_steps=1e5, start_step=2e4, end_step=1.2e5))

##############################################################
#                       Train model                          #
##############################################################

model.fit(
    int_data.tensor_train_data,
    validation_data=int_data.tensor_val_data,
    epochs=train_settings['epochs'],
    callbacks=callbacks
)

if train_settings['log_wandb']:
    wandb.finish()

#model.save('Prosit_cit/Intensity/')
