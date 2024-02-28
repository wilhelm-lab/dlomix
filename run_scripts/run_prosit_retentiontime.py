import os
import pickle
import sys

import pandas as pd
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from dlomix.data import RetentionTimeDataset
from dlomix.eval import TimeDeltaMetric
from dlomix.models import PrositRetentionTimePredictor
from dlomix.reports import RetentionTimeReport

# consider the use-case for starting from a saved model

model = PrositRetentionTimePredictor(seq_length=30)

optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-7)

TRAIN_DATAPATH = "../example_dataset/proteomTools_train_val.csv"
TEST_DATAPATH = "../example_dataset/proteomTools_test.csv"

d = RetentionTimeDataset(
    data_source=TRAIN_DATAPATH, seq_length=30, batch_size=512, val_ratio=0.2
)

model.compile(
    optimizer=optimizer, loss="mse", metrics=["mean_absolute_error", TimeDeltaMetric()]
)

weights_file = "./prosit_test"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    weights_file, save_best_only=True, save_weights_only=True
)
decay = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=10, verbose=1, min_lr=0
)
early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
callbacks = [checkpoint, early_stop, decay]


history = model.fit(
    d.train_data, epochs=150, validation_data=d.val_data, callbacks=callbacks
)

test_rtdata = RetentionTimeDataset(
    data_source=TEST_DATAPATH, seq_length=30, batch_size=512, test=True
)

predictions = model.predict(test_rtdata.test_data)
predictions = d.denormalize_targets(predictions)
predictions = predictions.ravel()
test_targets = test_rtdata.get_split_targets(split="test")

report = RetentionTimeReport(output_path="./output", history=history)

print("R2: ", report.calculate_r2(test_targets, predictions))

pd.DataFrame(
    {
        "sequence": test_rtdata.sequences,
        "irt": test_rtdata.targets,
        "predicted_irt": predictions,
    }
).to_csv("./predictions_prosit_fullrun.csv", index=False)


# TODO: function to store and load history object, or maybe consider saving the report object
with open("./history_prosit.pkl", "wb") as f:
    pickle.dump(history.history, f)
