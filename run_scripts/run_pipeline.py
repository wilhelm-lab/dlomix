from dlomix.data import RetentionTimeDataset
from dlomix.eval import TimeDeltaMetric, TimeDeltaMetric2

# data path
from dlomix.models import RetentionTimePredictor

DATAPATH = "../example_dataset/proteomTools_train_val.csv"

"""
create dataset:
    - abstracts TF Dataset
    - splits sequences
    - padds sequences to the specified length
    - performs train/val split if needed
"""
d = RetentionTimeDataset(
    data_format="csv",
    data_source=DATAPATH,
    sequence_column="sequence",
    label_column="irt",
    max_seq_len=40,
    batch_size=128,
    val_ratio=0.2,
)


"""
create Model:
    - abstracts a RT Prediction Model
    - encoding to integers happens at beginning of the model
    - model has three main parts (embedding, encoder, regressor) with some parametrization
    """

model = RetentionTimePredictor(embedding_dim=50, seq_length=40, encoder="lstm")


"""'

Target is to always have the high-level model functions callabe to abstract out training, but still people
can write their own custom training loops since the model extends tf.keras.Model

 """

model.compile(
    optimizer="adam", loss="mse", metrics=["mean_absolute_error", TimeDeltaMetric()]
)

history = model.fit(d.tensor_train_data, epochs=5, validation_data=d.tensor_val_data)

"""

"""
