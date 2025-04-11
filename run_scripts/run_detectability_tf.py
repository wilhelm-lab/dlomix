import tensorflow as tf

from dlomix.constants import CLASSES_LABELS, aa_to_int_dict, alphabet
from dlomix.data import DetectabilityDataset

max_pep_length = 40
BATCH_SIZE = 128

# The Class handles all the inner details, we have to provide the column names and the alphabet for encoding
# If the data is already split with a specific logic (which is generally recommended) -> val_data_source and test_data_source are available as well

hf_data = "Wilhelmlab/detectability-proteometools"
detectability_data = DetectabilityDataset(
    data_source=hf_data,
    data_format="hub",
    max_seq_len=max_pep_length,
    label_column="Classes",
    sequence_column="Sequences",
    dataset_columns_to_keep=None,
    batch_size=BATCH_SIZE,
    with_termini=False,
    alphabet=aa_to_int_dict,
)


from dlomix.models import DetectabilityModel

total_num_classes = len(CLASSES_LABELS)
input_dimension = len(alphabet)
num_cells = 64

model = DetectabilityModel(num_units=num_cells, num_classes=total_num_classes)


model_save_path = "run_scripts/output/base_model_weights_detectability"

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_save_path,
    monitor="val_sparse_categorical_accuracy",
    mode="max",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)

model.compile(
    optimizer="adam",
    loss="SparseCategoricalCrossentropy",
    metrics="sparse_categorical_accuracy",
)

history = model.fit(
    detectability_data.tensor_train_data,
    validation_data=detectability_data.tensor_val_data,
    epochs=2,
    callbacks=[model_checkpoint],
)

predictions = model.predict(detectability_data.tensor_test_data)
print(predictions.shape)
