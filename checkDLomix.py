#!python -m pip install -q git+https:/hub.com/wilhelm-lab/dlomix
#script version of https://colab.research.google.com/github/wilhelm-lab/dlomix-resources/blob/main/notebooks/Example_RTModel_Walkthrough_colab.ipynb 
#USAGE: python dlomixRT.py <train.csv> <test.csv file>
import sys
import numpy as np
import pandas as pd
import dlomix
from dlomix import constants, data, eval, layers, models, pipelines, reports, utils
print([x for x in dir(dlomix) if not x.startswith("_")])
from dlomix.data import RetentionTimeDataset
TRAIN_DATAPATH = sys.argv[1]#'proteomTools_train_val.csv'
BATCH_SIZE = 64
rtdata = RetentionTimeDataset(data_source=TRAIN_DATAPATH,seq_length=30, batch_size=BATCH_SIZE, val_ratio=0.2, test=False)
print( "Training examples", BATCH_SIZE * len(rtdata.train_data))
print("Validation examples", BATCH_SIZE * len(rtdata.val_data))
from dlomix.models import RetentionTimePredictor
model = RetentionTimePredictor(seq_length=30)
from dlomix.eval import TimeDeltaMetric
model.compile(optimizer='adam',loss='mse',metrics=['mean_absolute_error', TimeDeltaMetric()])
history = model.fit(rtdata.train_data,validation_data=rtdata.val_data,epochs=20)
TEST_DATAPATH = sys.argv[2]#'proteomTools_test.csv'
test_rtdata = RetentionTimeDataset(data_source=TEST_DATAPATH,seq_length=30, batch_size=32, test=True)
predictions = model.predict(test_rtdata.test_data)
predictions = predictions.ravel()
test_targets = test_rtdata.get_split_targets(split="test")
print(test_targets, predictions)
save_path = "rtmodel"
model.save_weights(save_path)
trained_model = RetentionTimePredictor(seq_length=30)
trained_model.load_weights(save_path)
new_predictions = trained_model.predict(test_rtdata.test_data)
new_predictions = new_predictions.ravel()
print(np.allclose(predictions, new_predictions))
results_df = pd.DataFrame({"sequence": test_rtdata.sequences,"irt": test_rtdata.targets,"predicted_irt": predictions})
print(results_df)
results_df.to_csv("predictions_irt.csv", index=False)
print(pd.read_csv("predictions_irt.csv"))
