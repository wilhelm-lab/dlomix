import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'preprocessing')))
from preprocessing.utils import read_parquet
import argparse
import json
from models.chargestate import ChargeStateDistributionPredictor
import keras
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from dlomix_preprocessing import to_dlomix
from dlomix.constants import PTMS_ALPHABET


def get_callbacks(config, model_dir_path):
    save = WandbModelCheckpoint(os.path.join(os.path.abspath(model_dir_path), 'checkpoints'))
    stop = keras.callbacks.EarlyStopping(patience=config.early_stopping_patience)
    reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                factor=0.5,
                patience=2,
                verbose=1,
                min_lr=0),
    wandb_log = WandbMetricsLogger(log_freq=10)

    return [save, stop, reduceLR, wandb_log]


def train(base_dir, X_train, X_test, y_train, y_test, config) -> keras.models.Model:
    PTMS_ALPHABET['W[UNIMOD:425]'] = 57
    PTMS_ALPHABET['K[UNIMOD:1342]'] = 58
    PTMS_ALPHABET['[UNIMOD:27]-'] = 59
    model = ChargeStateDistributionPredictor(
        num_classes=6,
        seq_length=32,
        vocab_dict=PTMS_ALPHABET
    )

    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    metrics = config.metrics

    if 'adjusted_mean_absolute_error' in metrics:
        sys.path.append(os.path.join(
            os.path.dirname(__file__),
            '..')
        )
        from custom_keras_utils import adjusted_mean_absolute_error

        i = metrics.index('adjusted_mean_absolute_error')
        metrics[i] = adjusted_mean_absolute_error

    model.compile(optimizer=optimizer, metrics=metrics, loss=config.loss)

    model.fit(X_train, y_train,
        epochs=config.epochs,
        batch_size = config.batch_size,
        validation_data =(X_test, y_test),
        verbose=1,
        callbacks=get_callbacks(config, base_dir))
    
    keras.saving.save_model(model, os.path.join(base_dir, 'trained_model.keras'))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)

    parser.add_argument('-j', '--json', help="Configuration file for wandb in json format", type=str, required=True)

    args = parser.parse_args()

    wandb_info_json_path = args.json

    with open(wandb_info_json_path, 'r') as f:
        wandb_info = json.load(f)

    config = wandb_info['config']

    script_dir = os.path.dirname(__file__)

    train_set_path = os.path.join(
        script_dir,
        '../..',
        config['train_set']
    )

    val_set_path = os.path.join(
        script_dir,
        '../..',
        config['val_set']
    )

    train_df = read_parquet(train_set_path)
    val_df = read_parquet(val_set_path)

    X_train, y_train = to_dlomix(train_df)
    X_test, y_test = to_dlomix(val_df)

    base_dir = os.path.join(
        script_dir,
        '../../data/dlomix',
        wandb_info['tags'][0]
    )

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    wandb.init(**wandb_info)

    model = train(
        base_dir,
        X_train,
        X_test,
        y_train,
        y_test,
        wandb.config
    )