import tensorflow as tf

from dlomix.data import FragmentIonIntensityDataset
from dlomix.losses import masked_spectral_distance
from dlomix.models import PrositIntensityPredictor
from dlomix.reports.quarto import IntensityReportQuarto

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

TRAIN_DATAPATH = "https://github.com/wilhelm-lab/dlomix/raw/refs/heads/main/example_dataset/intensity/third_pool_processed_sample.parquet"
TEST_DATAPATH = "https://github.com/wilhelm-lab/dlomix/raw/refs/heads/main/example_dataset/intensity/third_pool_processed_sample.parquet"

d = FragmentIonIntensityDataset(
    data_format="parquet",
    data_source=TRAIN_DATAPATH,
    sequence_column="modified_sequence",
    label_column="intensities_raw",
    model_features=["precursor_charge_onehot", "collision_energy_aligned_normed"],
    max_seq_len=30,
    batch_size=128,
    val_ratio=0.2,
    with_termini=False,
)

print(d)

model = PrositIntensityPredictor(
    seq_length=30,
    input_keys={
        "SEQUENCE_KEY": "modified_sequence",
    },
    meta_data_keys={
        "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
        "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
    },
    with_termini=False,
)

model.compile(optimizer=optimizer, loss=masked_spectral_distance, metrics=["mse"])

weights_file = "./run_scripts/output/prosit_intensity_test"
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
    epochs=2,
    validation_data=d.tensor_val_data,
    callbacks=callbacks,
)


d_test = FragmentIonIntensityDataset(
    data_format="parquet",
    test_data_source=TEST_DATAPATH,
    sequence_column="modified_sequence",
    label_column="intensities_raw",
    model_features=["precursor_charge_onehot", "collision_energy_aligned_normed"],
    max_seq_len=30,
    batch_size=128,
    with_termini=False,
)

test_targets = d_test["test"]["intensities_raw"]
test_sequences = d_test["test"]["modified_sequence"]

predictions = model.predict(test_sequences)
predictions = predictions.ravel()

print(predictions.shape)

report = IntensityReportQuarto(
    history,
    test_data=d_test,
    train_data=d,
    model=model,
    title="Intensity report",
    fold_code=True,
    train_section=False,
    val_section=False,
    output_path="./run_scripts/output/intensity_quarto_report",
    precursor_charge_column_name="precursor_charge_onehot",
    collision_energy_column_name="collision_energy",
)

report.generate_report("int_quarto_report.qmd")
