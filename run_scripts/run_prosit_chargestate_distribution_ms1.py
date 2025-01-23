import keras
import tensorflow as tf

from dlomix.constants import PTMS_ALPHABET
from dlomix.data import ChargeStateDataset
from dlomix.eval import adjusted_mean_absolute_error
from dlomix.models import ChargeStateDistributionPredictor

# Model

model = model = ChargeStateDistributionPredictor(
    num_classes=6, seq_length=32, alphabet=PTMS_ALPHABET
)

print(model)

optimizer = tf.keras.optimizers.Adam(lr=0.0001)

d = ChargeStateDataset(
    data_format="hub",
    data_source="Wilhelmlab/prospect-ptms-charge",
    sequence_column="modified_sequence",
    label_column="charge_state_dist",
    max_seq_len=30,
    batch_size=512,
)

print(d)

for x in d.tensor_train_data:
    print(x)
    break

test_targets = d["test"]["charge_state_dist"]
test_sequences = d["test"]["modified_sequence"]

# callbacks

weights_file = "./output/prosit_charge_dist_ms1_test"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    weights_file, save_best_only=True, save_weights_only=True
)

stop = keras.callbacks.EarlyStopping(patience=20)

callbacks = [checkpoint, stop]

metrics = [adjusted_mean_absolute_error]

model.compile(optimizer=optimizer, metrics=metrics, loss="mean_squared_error")

history = model.fit(
    d.tensor_train_data,
    epochs=2,
    validation_data=d.tensor_val_data,
    callbacks=callbacks,
)

predictions = model.predict(test_sequences)
predictions = predictions.ravel()

print(test_sequences[:5])
print(test_targets[:5])
print(predictions[:5])
