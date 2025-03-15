import tensorflow as tf

from dlomix.constants import PTMS_ALPHABET
from dlomix.data import ChargeStateDataset
from dlomix.eval import adjusted_mean_absolute_error
from dlomix.models import ChargeStatePredictor

model = model = ChargeStatePredictor(
    num_classes=6, seq_length=30, alphabet=PTMS_ALPHABET, model_flavour="relative"
)
print(model)


optimizer = tf.keras.optimizers.Adam(lr=0.0001)


TESTING_DATA = "example_dataset/chargestate/chargestate_data.parquet"

d = ChargeStateDataset(
    data_format="parquet",  # "hub",
    data_source=TESTING_DATA,  # "Wilhelmlab/prospect-ptms-charge",
    sequence_column="modified_sequence",
    label_column="charge_state_dist",
    max_seq_len=30,
    batch_size=8,
)
print(d)
for x in d.tensor_train_data:
    print(x)
    break

test_d = ChargeStateDataset(
    data_format="parquet",  # "hub",
    test_data_source=TESTING_DATA,  # "Wilhelmlab/prospect-ptms-charge",
    sequence_column="modified_sequence",
    label_column="charge_state_dist",
    max_seq_len=30,
    batch_size=8,
)
test_targets = test_d["test"]["charge_state_dist"]
test_sequences = test_d["test"]["modified_sequence"]


# callbacks
weights_file = "./run_scripts/output/prosit_charge_dist_test"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    weights_file, save_best_only=True, save_weights_only=True
)
early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
callbacks = [checkpoint, early_stop]


model.compile(
    optimizer=optimizer,
    loss="mean_squared_error",
    metrics=[adjusted_mean_absolute_error],
)


history = model.fit(
    d.tensor_train_data,
    epochs=1,  # 2,
    validation_data=d.tensor_val_data,
    callbacks=callbacks,
)

predictions = model.predict(test_sequences)
predictions = predictions

print("first 5 test sequences:\n", test_sequences[:5])
print("first 5 test relative charge state vectors (label):\n", test_targets[:5])
print("first 5 relative charge state predictions for test:\n", predictions[:5])
print(
    "predictions.shape for test set:",
    predictions.shape,
    "number of test CS vectors (label):",
    len(test_targets),
)
