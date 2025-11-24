"""
    To run this script, use the following command:
    DLOMIX_BACKEND=torch python run_scripts/run_prosit_intensity_ptms_torch.py
"""

import logging

import torch
from tqdm import tqdm

from dlomix.data import FragmentIonIntensityDataset
from dlomix.losses.intensity_torch import masked_spectral_distance
from dlomix.models import PrositIntensityPredictor

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

BATCH_SIZE = 8
N_EPOCHS = 20

model = PrositIntensityPredictor(
    seq_length=32,
    use_prosit_ptm_features=True,
    input_keys={
        "SEQUENCE_KEY": "modified_sequence",
    },
    meta_data_keys={
        "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
        "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
    },
    with_termini=True,
)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

TRAIN_DATAPATH = "example_dataset/intensity/third_pool_processed_sample.parquet"

d = FragmentIonIntensityDataset(
    data_source=TRAIN_DATAPATH,
    max_seq_len=32,
    batch_size=BATCH_SIZE,
    val_ratio=0.2,
    model_features=["collision_energy_aligned_normed", "precursor_charge_onehot"],
    sequence_column="modified_sequence",
    label_column="intensities_raw",
    features_to_extract=["mod_loss", "delta_mass"],
    dataset_type="pt",
    with_termini=True,
)

print(d)

loss_criterion = masked_spectral_distance

for epoch in tqdm(range(0, N_EPOCHS)):
    epoch_loss = 0
    model.train()
    data_size = len(d.tensor_train_data)
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
    print(f"Epoch {epoch} Summary: Training Loss: {epoch_loss / data_size:.4f}")

    # Validation phase.
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        val_data_size = len(d.tensor_val_data)
        for batch in d.tensor_val_data:
            val_pred_cs = model(batch)
            val_loss = loss_criterion(batch["intensities_raw"], val_pred_cs)
            val_loss_total += val_loss.item()

        avg_val_loss = val_loss_total / val_data_size
    print(f"Epoch {epoch} Summary:  Validation Loss: {avg_val_loss:.4f}")

print(val_pred_cs.shape)
print(val_pred_cs[0])
