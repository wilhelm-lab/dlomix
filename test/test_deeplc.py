from dlpro.eval.rt_eval import delta99_metric, delta95_metric
import dlpro.models
from dlpro.data.data import RetentionTimeDataset
import pickle
import tensorflow as tf

model = DeepLC(seq_length=50)
opt = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)

model.compile(optimizer=opt,
              loss='mse',
              metrics=['mean_absolute_error', delta95_metric, delta99_metric])


DATAPATH = '/scratch/RT_raw/iRT_ProteomeTools_ReferenceSet.csv'

d = RetentionTimeDataset(data_source=DATAPATH, pad_length=50, batch_size=512, val_ratio=0.2,
                         path_aminoacid_atomcounts="./lookups/aa_comp_rel.csv")

history = model.fit(d.tf_dataset['train'], epochs=3, validation_data=d.tf_dataset['val'])

model.save_weights('./API-Overview/deeplc_test_sgd')

with open("./history_deeplc_sgd.pkl", 'wb') as f:
   pickle.dump(history.history, f)


