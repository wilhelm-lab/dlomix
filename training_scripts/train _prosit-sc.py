import numpy as np
import pandas as pd
import wandb
from wandb.keras import WandbCallback
from dlomix.models import PrositSumIntensityPredictor
import tensorflow as tf
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
#from dlomix.reports import IntensityReport
#from dlomix.reports.postprocessing import normalize_intensity_predictions
from dlomix.data import FragmentIonIntensityDataset
from tensorflow.keras.callbacks import *

tf.get_logger().setLevel('ERROR')

"""
In this version we keep the prosit model as it is and the way to read the data as well
but use sqrt intensity in the input files
also add sum_intensity and method_nbr as an addtional feature to the model.
"""

# enter project name
project_name = 'train_with_upd2_v3'
name = 'sc_intensity_model'


TRAIN_DATAPATH = '/cmnfs/proj/prosit/singleCell/train_upd2_sqrt2.parquet'
TEST_DATAPATH = '/cmnfs/proj/prosit/singleCell/test_upd2_sqrt2.parquet'
VAL_DATAPATH = '/cmnfs/proj/prosit/singleCell/val_upd2_sqrt2.parquet'
BATCH_SIZE = 2048

PTMS_ALPHABET = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "M[UNIMOD:35]": 21,
    "C[UNIMOD:4]": 22}

int_data = FragmentIonIntensityDataset(
    data_source=TRAIN_DATAPATH,
    val_data_source=VAL_DATAPATH,
    test_data_source=TEST_DATAPATH,
    data_format="parquet",
    val_ratio=0.2, max_seq_len=30, encoding_scheme="naive-mods",
    vocab=PTMS_ALPHABET,
    sequence_column="modified_sequence",
    label_column="intensities_raw",
    model_features=["aligned_collision_energy", "precursor_charge_onehot", "sum_intensities", "method_nbr"],
    batch_size = BATCH_SIZE,
)

save_best = ModelCheckpoint(
    '/cmnfs/proj/prosit/singleCell/best_sc_model.keras',
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=5, min_lr=0.00001,min_delta=0.01)

model = PrositSumIntensityPredictor(seq_length=30, vocab_dict=PTMS_ALPHABET)

# create the optimizer object
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001) # changed from 0.001

# compile the model  with the optimizer and the metrics we want to use, we can add our custom timedelta metric
model.compile(optimizer=optimizer,
              loss=masked_spectral_distance,
              metrics=[masked_pearson_correlation_distance])

wandb.init(project=project_name, name=name)
#model.build(input_shape={"sum_intensities":(2048), "modified_sequence":(2048,30),"aligned_collision_energy":(2048), "precursor_charge_onehot":(2048,6)})#[(2048,1),(2048,6),(2048,1)]
history = model.fit(int_data.tensor_train_data,
                    validation_data=int_data.tensor_val_data,
                    epochs=100,callbacks=[WandbCallback(save_model=False), save_best, reduce_lr])
#model.load_weights('/cmnfs/proj/prosit/singleCell/best_intensity_model_sqrt.keras')

model.save("/cmnfs/proj/prosit/singleCell/save_sc_model", save_format="tf")

# Mark the run as finished
wandb.finish()

"""TEST_DATAPATH = '/cmnfs/home/m.khanh/test_models/proteomeTools_sum_test.csv'
test_int_data = IntensityDataset(data_source=TEST_DATAPATH, collision_energy_col='collision_energy', feature_cols=['sum_intensities'],
                              seq_length=30, batch_size=32, test=True) 

# use model.predict from keras directly on the testdata
predictions = model.predict(test_int_data.test_data)

# create a report object by passing the history object and plot different metrics
report = IntensityReport(output_path="/cmnfs/home/m.khanh/test_models/output_feature", history=history)

# you can generate a complete report for intensity by calling generate_report
# the function takes the test dataset object and the predictions as arguments
report.generate_report(test_int_data, predictions)

# you can also manually see the results by calling other utility functions
predictions_df = report.generate_intensity_results_df(test_int_data, predictions)
print(predictions_df.head())

predictions_acc = normalize_intensity_predictions(predictions_df)
print(predictions_acc.head())
print(predictions_acc['spectral_angle'].describe())"""
