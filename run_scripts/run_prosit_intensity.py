import os
import sys

import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from dlomix.data import FragmentIonIntensityDataset
from dlomix.losses import masked_spectral_distance
from dlomix.models import PrositIntensityPredictor

# consider the use-case for starting from a saved model

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

TRAIN_DATAPATH = "../example_dataset/intensity/intensity_data.parquet"

d = FragmentIonIntensityDataset(
    data_format="parquet",
    data_source=TRAIN_DATAPATH,
    sequence_column="sequence",
    label_column="intensities",
    model_features=["precursor_charge_onehot", "collision_energy_aligned_normed"],
    max_seq_len=30,
    batch_size=128,
    val_ratio=0.2,
)

print(d)

model = PrositIntensityPredictor(
    seq_length=30,
    sequence_input_name="sequence",
    collision_energy_input_name="collision_energy_aligned_normed",
    precursor_charge_input_name="precursor_charge_onehot",
    # fragmentation_method_input_name="method_nbr",
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
    d.tensor_train_data,
    epochs=5,
    validation_data=d.tensor_val_data,
    callbacks=callbacks,
)


predictions = model.predict(d.tensor_val_data)

print(predictions.shape)
print(predictions[0])
