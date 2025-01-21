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

from custom_keras_utils import adjusted_mean_absolute_error, adjusted_mean_squared_error, masked_spectral_distance, masked_pearson_correlation_distance, smoothed_csd_mean_absolute_error, smoothed_csd_mean_squared_error, smoothed_csd_categorical_crossentropy, smoothed_csd_masked_spectral_distance, euclidean_distance_loss, smoothed_csd_upscaled_mean_squared_error
import keras
from keras.optimizers import Adam
from keras.losses import cosine_similarity
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import pickle


def get_CSD_data(seq_length: int) -> ChargeStateDataset:
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, '../../../data/seitz')

    train_fp = os.path.join(data_dir, 'train_smoothed.parquet')
    val_fp = os.path.join(data_dir, 'val_smoothed.parquet')
    test_fp = os.path.join(data_dir, 'test_smoothed.parquet')

    data = ChargeStateDataset(
        data_source=train_fp,
        val_data_source=val_fp,
        test_data_source=test_fp,
        sequence_column="modified_sequence",
        label_column="smoothed_charge_state_dist",
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

    name = "dlomix_smoothed_csd_upscaled_mse" # TODO
    configs = dict(
        seq_length = 32,
        batch_size = 64,
        earlystopping_patience = 5,
        reduceLR_factor = 0.5,
        reduceLR_patience = 2,
        learning_rate = 1e-3,
        epochs = 100_000,
        loss = smoothed_csd_upscaled_mean_squared_error,
        metrics = [
            #adjusted_mean_absolute_error,
            #adjusted_mean_squared_error,
            #"mean_squared_error",
            #"mean_absolute_error",
            #cosine_similarity,
            #"categorical_crossentropy",
            #masked_spectral_distance,
            #masked_pearson_correlation_distance,
            euclidean_distance_loss
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
        tags = ["seitz_gorshkov", "dlomix_improvement", "smoothed_csd"],
        config = configs
    )

    csd_model = ChargeStateDistributionPredictor(
        seq_length=configs['seq_length'],
        vocab_dict=PTMS_ALPHABET
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
