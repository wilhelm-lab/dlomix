"""Collaborative Filtering model for imputation.

with Ligthening: `pip install lightning`
"""

"""Pending:
ensure data is available, at least for examples -> URL is available
https://raw.githubusercontent.com/RasmussenLab/pimms/refs/heads/main/project/data/dev_datasets/HeLa_6070/protein_groups_wide_N50_M454.csv

- Data is loaded via native Hugging Face datasets OR extend the datasets class to handle tabular data:
    - adapt Datasets base class (if needed) -> to handle task formulation that does not strictly have the sequence as an input to the model.
- data prep script to be done outside, then the exported CSV is to be used for the dataset class
- Replace the load_datasets call with a class in DLOmix
"""

# %%
import pathlib

import lightning as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from lightning.pytorch.tuner import Tuner
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.mps.is_available() else "cpu")
device

# %% [markdown]
# Load data from URL once

# %%
DATASOURCE = pathlib.Path("protein_groups_wide_N50_M454.csv")
if not DATASOURCE.exists():
    URLSOURCE = (
        "https://raw.githubusercontent.com/RasmussenLab/pimms/refs/heads/main/"
        "project/data/dev_datasets/HeLa_6070/protein_groups_wide_N50_M454.csv"
    )
    data = (
        pd.read_csv(URLSOURCE, index_col=0)
        .rename_axis("protein_group", axis=1)
        .stack()
        .squeeze()
        .to_frame("intensity")
        .map(np.log2)
    )
    data.to_csv("protein_groups_wide_N50_M454.csv")  # local dump

# %% [markdown]
# map default columns to dataset columns

# %%
col_map = {"sample": "Sample ID", "feature": "protein_group", "intensity": "intensity"}

# %% [markdown]
# - load tabular dataset (long format of LFQ protein groups)

# %%
ds = load_dataset(
    "csv",
    data_files=str(DATASOURCE),
    split="train",
    cache_dir=None,
).with_format("torch")
ds

# %%
set_samples = set(ds[col_map["sample"]])
n_samples = len(set_samples)
set_features = set(ds[col_map["feature"]])
n_features = len(set_features)

lookup = dict()
lookup["sample"] = {c: i for i, c in enumerate(set_samples)}
lookup["feature"] = {c: i for i, c in enumerate(set_features)}

# %%
ds_dict = ds.train_test_split(test_size=0.2)
ds_dict

# %% [markdown]
# inspect a single dataset entry and a batch

# %%
for item in ds_dict["train"]:
    break
item

# %%
dl_train = DataLoader(ds_dict["train"], batch_size=4096)
dl_test = DataLoader(ds_dict["test"], batch_size=1024)
for batch in dl_train:
    break
batch

# %%
sample_ids, feature_ids, intensities = (
    batch[col_map["sample"]],
    batch[col_map["feature"]],
    batch[col_map["intensity"]],
)

# %% [markdown]
# Model
# - coupling to external knowledge about the data (categories and lookup)
# - embedding indices have to be look-up on each forward pass
# - same dimension for sample and feature embedding, simple dot-product loss with MSE


# %%
COL_MAP = {"sample": "sample", "feature": "feature", "intensity": "intensity"}


class CollaborativeFilteringModel(pl.LightningModule):
    def __init__(
        self,
        num_samples: int,
        num_features: int,
        lookup: dict[str, dict[str, int]],
        col_map: dict[str, str] = COL_MAP,
        learning_rate: float = 0.001,
        embedding_dim: int = 32,
    ):
        super(CollaborativeFilteringModel, self).__init__()
        self.sample_embedding = nn.Embedding(num_samples, embedding_dim, device=device)
        self.feature_embedding = nn.Embedding(
            num_features, embedding_dim, device=device
        )
        self.col_map = col_map
        self.lookup = lookup
        self.fc = nn.Linear(embedding_dim, 1)
        self.loss = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, sample_ids, feature_ids):
        sample_ids = torch.tensor(
            [self.lookup["sample"][sample] for sample in sample_ids],
            device=self.device,
        )
        feature_ids = torch.tensor(
            [self.lookup["feature"][feat] for feat in feature_ids],
            device=self.device,
        )
        sample_embeds = self.sample_embedding(sample_ids)
        feature_embeds = self.feature_embedding(feature_ids)
        sample_embeds
        dot_product = (sample_embeds * feature_embeds).sum(1)
        return dot_product

    def training_step(self, batch):
        sample_ids, feature_ids, intensities = (
            batch[self.col_map["sample"]],
            batch[self.col_map["feature"]],
            batch[self.col_map["intensity"]],
        )

        predictions = self(sample_ids, feature_ids)
        loss = self.loss(predictions, intensities.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        sample_ids, feature_ids, intensities = (
            batch[self.col_map["sample"]],
            batch[self.col_map["feature"]],
            batch[self.col_map["intensity"]],
        )
        predictions = self(sample_ids, feature_ids)
        loss = self.loss(predictions, intensities.float())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# %%
model = CollaborativeFilteringModel(
    num_samples=n_samples,
    num_features=n_features,
    lookup=lookup,
    col_map=col_map,
    embedding_dim=10,
)
model

# %%
trainer = pl.Trainer(accelerator=str(device), max_epochs=50)
tuner = Tuner(trainer)

# %%
# Create a Tuner
tuner.lr_find(model, train_dataloaders=dl_train, attr_name="learning_rate")

# %%
trainer.fit(model, dl_train)
train_loss = trainer.callback_metrics.get("train_loss")
# val_loss = trainer.callback_metrics.get("val_loss")

print(f"Training Loss: {train_loss}")
# print(f"Validation Loss: {val_loss}")


# %%
# Evaluate the model
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in dl_train:
        sample_ids, feature_ids, intensities = (
            batch[col_map["sample"]],
            batch[col_map["feature"]],
            batch[col_map["intensity"]],
        )
        preds = model(sample_ids, feature_ids)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(intensities.cpu().numpy())

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
print(f"RMSE (train): {rmse}")
mae = mean_absolute_error(all_targets, all_preds)
print(f"MAE (train): {mae}")

# %%
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in dl_test:
        sample_ids, feature_ids, intensities = (
            batch[col_map["sample"]],
            batch[col_map["feature"]],
            batch[col_map["intensity"]],
        )
        preds = model(sample_ids, feature_ids)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(intensities.cpu().numpy())

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
print(f"RMSE (test): {rmse}")
mae = mean_absolute_error(all_targets, all_preds)
print(f"MAE (test): {mae}")

# %%
