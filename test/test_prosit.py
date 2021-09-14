import os
import sys
import pickle
import tensorflow as tf
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from mlomix.eval.rt_eval import TimeDeltaMetric
from mlomix.models.prosit import PrositRetentionTimePredictor
from mlomix.data.RetentionTimeDataset import RetentionTimeDataset
from mlomix.reports.RetentionTimeReport import RetentionTimeReport

# TODO: start from a saved model

model = PrositRetentionTimePredictor(seq_length=30)


optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-7)

TRAIN_DATAPATH = '/scratch/TMT/tmt_train_data.csv'
TEST_DATAPATH = '/scratch/TMT/tmt_test_data.csv'
#DATAPATH = '/scratch/RT_raw/iRT_ProteomeTools_ReferenceSet.csv'

d = RetentionTimeDataset(data_source=TRAIN_DATAPATH, seq_length=30, batch_size=512, val_ratio=0.2)

model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mean_absolute_error', TimeDeltaMetric(d.data_mean, d.data_std)])

weights_file = "./prosit_tmt_fullrun"
checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_file, save_best_only=True, save_weights_only=True)
decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=0)
early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
callbacks = [checkpoint, early_stop, decay]


history = model.fit(d.train_data, epochs=150, validation_data=d.val_data, callbacks=callbacks)

test_rtdata = RetentionTimeDataset(data_source=TEST_DATAPATH,
                                   seq_length=30, batch_size=512, test=True)

predictions = model.predict(test_rtdata.test_data)
predictions = d.denormalize_targets(predictions)
predictions = predictions.ravel()
test_targets = test_rtdata.get_split_targets(split="test")

report = RetentionTimeReport(output_path="./output", history=history)

print("R2: ", report.calculate_r2(test_targets, predictions))

pd.DataFrame(
    {"sequence": test_rtdata.sequences, "irt": test_rtdata.targets, "predicted_irt": predictions}
).to_csv("./predictions_prosit_fullrun.csv", index=False)


# TODO: function to store and load history object
with open("./history_prosit_fullrun.pkl", 'wb') as f:
    pickle.dump(history.history, f)

