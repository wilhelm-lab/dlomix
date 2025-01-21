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

from custom_keras_utils import (
    masked_pearson_correlation, 
    masked_spectral_angle, 
    euclidean_similarity, 
    upscaled_mean_squared_error
)

import keras
from keras.optimizers import Adam
from keras.losses import cosine_similarity
import wandb
from wandb.integration.keras import WandbMetricsLogger


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
    stop = keras.callbacks.EarlyStopping(patience=configs['earlystopping_patience'])
    reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                factor=configs['reduceLR_factor'],
                patience=configs['reduceLR_patience'],
                verbose=1,
                min_lr=0),
    wandb_log = WandbMetricsLogger(log_freq=10)

    return [stop, reduceLR, wandb_log]


def main():
    static_configs = dict(
        seq_length = 32,
        epochs = 100_000,
        loss = upscaled_mean_squared_error,
        metrics = [
            euclidean_similarity,
            masked_pearson_correlation,
            masked_spectral_angle,
            cosine_similarity
        ]
    )

    data = get_CSD_data(
        seq_length = static_configs['seq_length']
    )

    run = wandb.init()

    configs = wandb.config

    csd_model = ChargeStateDistributionPredictor(
        embedding_output_dim = configs['embedding_output_dim'],
        seq_length = static_configs['seq_length'],
        vocab_dict = PTMS_ALPHABET,
        dropout_rate = configs['dropout_rate'],
        latent_dropout_rate = configs['latent_dropout_rate'],
        recurrent_layers_sizes = configs['recurrent_layers_sizes'],
        regressor_layer_size = configs['regressor_layer_size'],
        output_activation_fn = configs['output_activation_fn']
    )

    optimizer = Adam(
        learning_rate = configs['learning_rate']
    )

    csd_model.compile(
        optimizer = optimizer,
        loss = static_configs['loss'],
        metrics = static_configs['metrics']
    )

    callbacks = get_callbacks(configs)

    csd_model_history = csd_model.fit(
        data.tensor_train_data,
        validation_data = data.tensor_val_data,
        batch_size = configs['batch_size'],
        epochs = static_configs['epochs'],
        callbacks = callbacks
    )


if __name__ == '__main__':
    sweep_configs = dict(
        method = 'bayes',
        metric = {'goal': 'minimize', 'name': 'epoch/val_loss'},
        parameters = {
            'batch_size': {'values': [32, 64, 128]},
            'earlystopping_patience': {'values': [i for i in range(5, 8)]},
            'reduceLR_factor': {'values': [0.5]},
            'reduceLR_patience': {'values': [3, 4, 5]},
            'learning_rate': {'values': [0.001]},
            'embedding_output_dim': {'values': [16, 32, 64, 128, 256]},
            'dropout_rate': {'values': [0.5]},
            'latent_dropout_rate': {'values': [0.01]},
            'recurrent_layers_sizes': {'values': [(256, 512), (512, 512), (512, 1024), (1024, 1024)]},
            'regressor_layer_size': {'values': [256, 512, 1024]},
            'output_activation_fn': {'values': ['softmax']}
        },
        early_terminate = {'type': 'hyperband', 's': 2, 'eta': 3, 'max_iter': 27},
    )

    #sweep_id = wandb.sweep(sweep=sweep_configs, project='charge_state_pred_eval')
    sweep_id = 'nussbaumer-genetics/charge_state_pred_eval/dtcclz60'
    wandb.agent(sweep_id, function=main, count=2)
