from dlpro.data.data import RetentionTimeDataset
from dlpro.eval.rt_eval import delta95_metric

# data path
DATAPATH = '/scratch/RT_raw/iRT_ProteomeTools_ReferenceSet.csv'

'''
create dataset:
    - abstracts TF Dataset
    - splits sequences
    - padds sequences to the specified length
    - performs train/val split if needed
'''
d = RetentionTimeDataset(data_source=DATAPATH, sequence_col="sequence", target_col="irt",
                         pad_length=40, normalize_targets=True,
                         batch_size=128, val_ratio=0.2)


'''
create Model:
    - abstracts a RT Prediction Model
    - encoding to integers happens at beginning of the model
    - model has three main parts (embedding, encoder, regressor) with some parametrization
    '''

model = RetentionTimePredictor(embeddings_count=len(ALPHABET_UNMOD), embedding_dim=300, seq_length=40, encoder='lstm')


''''

Target is to always have the high-level model functions callabe to abstract out training, but still people
can write their own custom training loops since the model extends tf.keras.Model

 '''

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mean_absolute_error', delta95_metric])

history = model.fit(d.get_tf_dataset('train'), epochs=15, validation_data=d.get_tf_dataset('val'))

'''

'''