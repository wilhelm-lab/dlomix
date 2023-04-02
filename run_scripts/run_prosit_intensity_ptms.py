import os
import sys

import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from dlomix.data import IntensityDataset
from dlomix.data.feature_extractors import (
    ModificationGainFeature,
    ModificationLocationFeature,
    ModificationLossFeature,
)
from dlomix.losses import masked_spectral_distance
from dlomix.models import PrositIntensityPredictor

# consider the use-case for starting from a saved model

model = PrositIntensityPredictor(seq_length=30, use_ptm_counts=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-7)

# TRAIN_DATAPATH = "../example_dataset/intensity/intensity_data.csv"
# TRAIN_DATAPATH = "../notebooks/data/third_pool_processed_sample.parquet"
TRAIN_DATAPATH = "../notebooks/data/third_pool_processed.parquet"
# TEST_DATAPATH = '../example_dataset/proteomTools_test.csv'

d = IntensityDataset(
    data_source=TRAIN_DATAPATH,
    seq_length=30,
    batch_size=128,
    val_ratio=0.3,
    precursor_charge_col="precursor_charge_onehot",
    sequence_col="modified_sequence",
    collision_energy_col="collision_energy_aligned_normed",
    intensities_col="intensities_raw",
    features_to_extract=[
        ModificationLocationFeature(),
        ModificationLossFeature(),
        ModificationGainFeature(),
    ],
    parser="proforma",
)

model.compile(optimizer=optimizer, loss=masked_spectral_distance, metrics=["mse"])

weights_file = "./prosit_intensity_test"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    weights_file, save_best_only=True, save_weights_only=True
)
decay = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=10, verbose=1, min_lr=0
)
early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
callbacks = [checkpoint, early_stop, decay]


history = model.fit(
    d.train_data, epochs=1, validation_data=d.val_data, callbacks=callbacks
)

predictions = model.predict(d.val_data)
# predictions = d.denormalize_targets(predictions)

print(predictions.shape)
print(predictions[0])