import torch
from tqdm import tqdm

from dlomix.constants import ALPHABET_UNMOD
from dlomix.data import FragmentIonIntensityDataset
from dlomix.losses.intensity_torch import masked_spectral_distance
from dlomix.models import PrositIntensityPredictor

BATCH_SIZE = 128

model = PrositIntensityPredictor(
    seq_length=30,
    use_prosit_ptm_features=False,
    input_keys={
        "SEQUENCE_KEY": "modified_sequence",
    },
    # meta_data_keys={
    #     "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
    #     "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
    # },
    with_termini=False,
    alphabet=ALPHABET_UNMOD,
)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

TRAIN_DATAPATH = "example_dataset/intensity/third_pool_processed_sample.parquet"

d = FragmentIonIntensityDataset(
    data_source=TRAIN_DATAPATH,
    max_seq_len=30,
    batch_size=BATCH_SIZE,
    val_ratio=0.2,
    # model_features=["collision_energy_aligned_normed", "precursor_charge_onehot"],
    sequence_column="modified_sequence",
    label_column="intensities_raw",
    # features_to_extract=["mod_loss", "delta_mass"],
    dataset_type="pt",
    alphabet=ALPHABET_UNMOD,
    with_termini=False,
)


loss_criterion = masked_spectral_distance

for epoch in tqdm(range(0, 100)):
    epoch_loss = 0
    model.train()
    data_size = len(d.tensor_train_data)
    print(data_size)
    for batch in d.tensor_train_data:
        optimizer.zero_grad()

        # output = model(batch["modified_sequence"])
        output = model(batch)
        # print("output: ", output)
        # print("label: ", batch["intensities_raw"])
        loss = loss_criterion(batch["intensities_raw"], output)
        # print(loss.item())
        epoch_loss += loss.item()

        loss.backward()

        # Add before optimizer.step()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1, norm_type=2, error_if_nonfinite=False
        )
        optimizer.step()
    print(epoch_loss / data_size)

    # Validation phase.
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        val_data_size = len(d.tensor_val_data)
        print(val_data_size)
        for batch in d.tensor_val_data:
            val_seq = batch["modified_sequence"]
            val_label = batch["intensities_raw"]

            val_pred_cs = model(val_seq)
            val_loss = loss_criterion(val_label, val_pred_cs)
            val_loss_total += val_loss.item()

        avg_val_loss = val_loss_total / val_data_size
    print(f"Epoch {epoch} Summary:  Validation Loss: {avg_val_loss:.4f}")


# # to add test data, a pool for example
# td = IntensityDataset(
#     data_source=TRAIN_DATAPATH,
#     seq_length=30,
#     batch_size=128,
#     val_ratio=0,
#     precursor_charge_col="precursor_charge_onehot",
#     sequence_col="modified_sequence",
#     collision_energy_col="collision_energy_aligned_normed",
#     intensities_col="intensities_raw",
#     features_to_extract=[
#         ModificationLocationFeature(),
#         ModificationLossFeature(),
#         ModificationGainFeature(),
#     ],
#     parser="proforma",
#     test=True,
# )
# predictions = model.predict(td.test_data)

# print(predictions.shape)
# print(predictions[0])

# from dlomix.reports import IntensityReport

# # create a report object by passing the history object and plot different metrics
# report = IntensityReport(output_path="./output", history=history)
# report.generate_report(td, predictions)
# # you can also manually see the results by calling other utility functions

# from dlomix.reports.postprocessing import normalize_intensity_predictions

# predictions_df = report.generate_intensity_results_df(td, predictions)
# predictions_df.to_csv("./predictions_prosit_intensity_ptm_fullrun.csv", index=False)
