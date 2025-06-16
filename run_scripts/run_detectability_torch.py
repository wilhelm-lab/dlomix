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
    dataset_type="pt",
)


from dlomix.models import DetectabilityModel

total_num_classes = len(CLASSES_LABELS)
input_dimension = len(alphabet)
num_cells = 64

model = DetectabilityModel(
    padding_idx=aa_to_int_dict["0"],
    num_units=num_cells,
    num_classes=total_num_classes,
    alphabet_size=input_dimension,
)


for batch in detectability_data.tensor_train_data:
    input_sequences = batch["Sequences"]
    output = model(input_sequences)
    break

print(input_sequences.shape)
print(output.shape)
