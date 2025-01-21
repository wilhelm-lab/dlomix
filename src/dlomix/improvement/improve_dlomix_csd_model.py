import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.chargestate import ChargeStateDistributionPredictor
from dlomix.data import ChargeStateDataset

from dlomix.constants import PTMS_ALPHABET
PTMS_ALPHABET['W[UNIMOD:425]'] = 57
PTMS_ALPHABET['K[UNIMOD:1342]'] = 58
PTMS_ALPHABET['[UNIMOD:27]-'] = 59

from custom_keras_utils import adjusted_mean_absolute_error, adjusted_mean_squared_error, masked_spectral_distance, masked_pearson_correlation_distance, euclidean_distance_loss, upscaled_mean_squared_error, euclidean_similarity
import keras
from keras.optimizers import Adam
from keras.losses import cosine_similarity
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import pickle


def get_CSD_data(seq_length: int) -> ChargeStateDataset:
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, '../../../data/seitz')

    train_fp = os.path.join(data_dir, 'train.parquet')
    val_fp = os.path.join(data_dir, 'val.parquet')
    test_fp = os.path.join(data_dir, 'test.parquet')

    data = ChargeStateDataset(
        data_source=train_fp,
        val_data_source=val_fp,
        test_data_source=test_fp,
        sequence_column="modified_sequence",
        label_column="charge_state_dist",
        max_seq_len=seq_length,
        pad=True,
        padding_value=0,
        alphabet=PTMS_ALPHABET,
        encoding_scheme="naive-mods"
    )

    return data


def get_callbacks(configs: dict) -> list:
    save = WandbModelCheckpoint(os.path.join(os.path.abspath(configs['model_dir_path']), 'checkpoints'))
    stop = keras.callbacks.EarlyStopping(patience=configs['earlystopping_patience'])
    reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                factor=configs['reduceLR_factor'],
                patience=configs['reduceLR_patience'],
                verbose=1,
                min_lr=0),
    wandb_log = WandbMetricsLogger(log_freq=10)

    return [save, stop, reduceLR, wandb_log]


if __name__ == '__main__':

    name = "dlomix_best_sweep" # TODO
    configs = dict(
        seq_length = 32,
        batch_size = 64,
        earlystopping_patience = 10,
        reduceLR_factor = 0.5,
        reduceLR_patience = 6,
        learning_rate = 1e-4,
        epochs = 100_000,
        loss = upscaled_mean_squared_error,
        metrics = [
            cosine_similarity,
            masked_pearson_correlation_distance,
            masked_spectral_distance,
            euclidean_similarity
        ],
        model_dir_path = os.path.join(
            os.path.dirname(__file__), 
            f"../../../data/dlomix/improvement/{name}"
        )
    )

    data = get_CSD_data(
        seq_length = configs['seq_length']
    )

    run = wandb.init(
        project = "charge_state_pred_eval",
        name = name,
        group = "dlomix",
        tags = ["seitz_gorshkov", "dlomix_improvement"],
        config = configs
    )

    csd_model = ChargeStateDistributionPredictor(
        embedding_output_dim=128,
        dropout_rate=0.55,
        latent_dropout_rate=0.01,
        recurrent_layers_sizes=(512, 512),
        regressor_layer_size=512,
        seq_length=configs['seq_length'],
        vocab_dict=PTMS_ALPHABET,
        output_activation_fn='softmax'
    )

    optimizer = Adam(
        learning_rate = configs['learning_rate']
    )

    csd_model.compile(
        optimizer = optimizer,
        loss = configs['loss'],
        metrics = configs['metrics']
    )

    callbacks = get_callbacks(configs)

    csd_model_history = csd_model.fit(
        data.tensor_train_data,
        validation_data = data.tensor_val_data,
        batch_size = configs['batch_size'],
        epochs = configs['epochs'],
        callbacks = callbacks
    )

    y_test_pred = csd_model.predict(data.tensor_test_data)

    test_pred_fp = os.path.join(configs['model_dir_path'], 'y_test_pred.pickle')
    if not os.path.exists(os.path.dirname(test_pred_fp)):
        os.mkdir(os.path.dirname(test_pred_fp))
    with open(test_pred_fp, 'wb') as f:
        pickle.dump(
            obj = y_test_pred, 
            file = f,
            protocol = pickle.HIGHEST_PROTOCOL
        )
    
    keras.saving.save_model(
        model = csd_model,
        filepath = os.path.join(
            configs['model_dir_path'],
            'trained_model.keras'
        )
    )

    run.finish()
