import pandas as pd
import tensorflow as tf

from dlomix.data import RetentionTimeDataset
from dlomix.eval import timedelta
from dlomix.models import PrositRetentionTimePredictor
from dlomix.reports import RetentionTimeReport

# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


TRAIN_DATAPATH = "example_dataset/proteomTools_train_val.csv"
TEST_DATAPATH = "example_dataset/proteomTools_test.csv"

d = RetentionTimeDataset(
    data_format="csv",
    data_source=TRAIN_DATAPATH,
    sequence_column="sequence",
    label_column="irt",
    max_seq_len=30,
    batch_size=512,
    val_ratio=0.2,
    with_termini=False,
    encoding_scheme="unmod",
)

model = PrositRetentionTimePredictor(seq_length=30, alphabet=d.extended_alphabet)

print(d)
print(model)

optimizer = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(
    optimizer=optimizer, loss="mse", metrics=["mean_absolute_error", timedelta]
)

weights_file = "./run_scripts/output/prosit_rt_test"
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

test_d = RetentionTimeDataset(
    data_format="csv",
    test_data_source=TEST_DATAPATH,
    sequence_column="sequence",
    label_column="irt",
    max_seq_len=30,
    batch_size=512,
    with_termini=False,
    alphabet=d.extended_alphabet,
    encoding_scheme="unmod",
)

test_targets = test_d["test"]["irt"]
test_sequences = test_d["test"]["sequence"]

predictions = model.predict(test_sequences)
predictions = predictions.ravel()

train_sequences = d["train"]["sequence"]
train_predictions = model.predict(train_sequences)
train_predictions = train_predictions.ravel()

print(train_sequences[:5])
print(train_predictions[:5])


print("-" * 50)
print("Test:")
print(test_sequences[:5])
print(test_targets[:5])
print(predictions[:5])


report = RetentionTimeReport(output_path="./run_scripts/output", history=history)

print("R2: ", report.calculate_r2(test_targets, predictions))

pd.DataFrame(
    {
        "sequence": test_d["test"]["_parsed_sequence"],
        "irt": test_targets,
        "predicted_irt": predictions,
    }
).to_csv("./run_scripts/output/predictions_prosit_rt_run.csv", index=False)

pd.DataFrame(
    {
        "sequence": d["train"]["_parsed_sequence"],
        "irt": d["train"]["irt"],
        "predicted_irt": train_predictions,
    }
).to_csv("./run_scripts/output/predictions_train_prosit_rt_run.csv", index=False)
