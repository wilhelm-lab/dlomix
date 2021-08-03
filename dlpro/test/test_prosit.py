
from dlpro.eval.rt_eval import delta99_metric, delta95_metric
from dlpro.models.prosit import PrositRetentionTimePredictor
from dlpro.data.data import RetentionTimeDataset
import pickle
import tensorflow as tf

import sys
sys.path.append('../../')

model = PrositRetentionTimePredictor()

reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=0)

optimizer = tf.keras.optimizers.Adam(lr=0.0001,decay=1e-7)

model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mean_absolute_error', delta95_metric, delta99_metric])


DATAPATH = '/scratch/RT_raw/iRT_ProteomeTools_ReferenceSet.csv'

d = RetentionTimeDataset(data_source=DATAPATH, pad_length=30, batch_size=512, val_ratio=0.2)

history = model.fit(d.tf_dataset['train'], epochs=2, validation_data=d.tf_dataset['val'], callbacks=[reduceLR])

model.save_weights('./API-Overview/prosit_test_adam')

with open("./history_prosit_test_adam.pkl", 'wb') as f:
   pickle.dump(history.history, f)


