import tensorflow as tf

from dlomix.data import ChargeStateDataset
from dlomix.eval import adjusted_mean_absolute_error
from dlomix.models import ChargeStatePredictor

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


DATA = "example_dataset/chargestate/chargestate_data.parquet"

d = ChargeStateDataset(
    data_format="parquet",  # "hub",
    data_source=DATA,  # "Wilhelmlab/prospect-ptms-charge",
    sequence_column="modified_sequence",
    label_column="charge_state_dist",
    max_seq_len=30,
    batch_size=8,
    val_ratio=0.2,
    test_ratio=0.1,  # hold out a test split that is not used for training
)
print(d)
for x in d.tensor_train_data:
    print(x)
    break

test_targets = d["test"]["charge_state_dist"]
test_sequences = d["test"]["modified_sequence"]


# the model must use the vocabulary the dataset learned (its extended_alphabet), not the
# raw PTMS_ALPHABET constant, which lacks the padding and unknown tokens
model = ChargeStatePredictor(
    num_classes=6,
    seq_length=32,  # max_seq_len + the N-/C-terminal tokens kept by default
    alphabet=d.extended_alphabet,
    model_flavour="relative",
)
print(model)


# callbacks
weights_file = "./run_scripts/output/prosit_charge_dist_test.weights.h5"
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

predictions = model.predict(d.tensor_test_data)

print("first 5 test sequences:\n", test_sequences[:5])
print("first 5 test relative charge state vectors (label):\n", test_targets[:5])
print("first 5 relative charge state predictions for test:\n", predictions[:5])
print(
    "predictions.shape for test set:",
    predictions.shape,
    "number of test CS vectors (label):",
    len(test_targets),
)
