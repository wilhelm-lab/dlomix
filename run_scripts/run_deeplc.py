import pickle

import tensorflow as tf

from dlomix.data.RetentionTimeDataset import RetentionTimeDataset
from dlomix.eval.rt_eval import TimeDeltaMetric
from dlomix.models.deepLC import DeepLCRetentionTimePredictor

model = DeepLCRetentionTimePredictor(seq_length=50)
opt = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)

model.compile(
    optimizer=opt, loss="mse", metrics=["mean_absolute_error", TimeDeltaMetric()]
)


DATAPATH = "../example_dataset/proteomTools_train_val.csv"

d = RetentionTimeDataset(
    data_source=DATAPATH,
    seq_length=50,
    batch_size=512,
    val_ratio=0.2,
    path_aminoacid_atomcounts="./lookups/aa_comp_rel.csv",
)

history = model.fit(
    d.tf_dataset["train"], epochs=3, validation_data=d.tf_dataset["val"]
)

model.save_weights("./output/deeplc_test")

with open("./history_deeplc.pkl", "wb") as f:
    pickle.dump(history.history, f)
