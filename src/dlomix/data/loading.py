"""
Resolve a dataset's data source(s) into a HuggingFace dataset and decide how splitting
should be handled.

``DataSourceLoader`` handles the three source modes (local files, HF Hub, in-memory
HF objects), determines the :class:`_DatasetSplitMode`, and emits the user-facing
warnings / conflict errors. ``PeptideDataset`` consumes the returned :class:`LoadResult`.
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from datasets import Dataset, DatasetDict, load_dataset

DEFAULT_SPLIT_NAMES = ["train", "val", "test"]


class _DatasetSplitMode(Enum):
    AUTO = "auto"  # Single train source; run the configured splitter
    PREDEFINED = "predefined"  # Splits defined externally; pass through as-is
    TEST_ONLY = "test_only"  # Only a test set provided; no splitting needed


@dataclass
class LoadResult:
    hf_dataset: Optional[Union[Dataset, DatasetDict]]
    available_splits: dict
    empty: bool
    split_mode: Optional[_DatasetSplitMode]


class DataSourceLoader:
    """Load data from files / hub / in-memory and resolve the split mode."""

    def __init__(self, config, hub_kwargs: Optional[dict] = None):
        self.config = config
        self.hub_kwargs = hub_kwargs or {}

    def load(self) -> LoadResult:
        if self.config.data_format == "hub":
            hf_dataset, available, empty = self._load_from_hub()
        elif self.config.data_format == "hf":
            hf_dataset, available, empty = self._load_from_inmemory()
        else:
            hf_dataset, available, empty = self._load_from_files()

        split_mode = self._determine_split_mode(available)
        self._validate_and_warn(split_mode, available)
        return LoadResult(hf_dataset, available, empty, split_mode)

    # ------------------------------------------------------------------- sources

    def _load_from_files(self):
        sources = [
            self.config.data_source,
            self.config.val_data_source,
            self.config.test_data_source,
        ]
        available = {
            split: source
            for split, source in zip(DEFAULT_SPLIT_NAMES, sources)
            if source is not None
        }

        if not available:
            warnings.warn(
                "No data files provided, please provide at least one data source if you "
                "plan to use this dataset directly. Otherwise, you can later load data "
                "into this empty dataset"
            )
            return None, available, True

        hf_dataset = load_dataset(self.config.data_format, data_files=available)
        return hf_dataset, available, False

    def _load_from_hub(self):
        hf_dataset = load_dataset(self.config.data_source, **self.hub_kwargs)
        warnings.warn(
            "The provided data is assumed to be hosted on the Hugging Face Hub since "
            'data_format is set to "hub". Validation and test data sources will be ignored.'
        )

        if isinstance(hf_dataset, DatasetDict):
            for split in hf_dataset.keys():
                if split not in DEFAULT_SPLIT_NAMES:
                    raise ValueError(
                        f"The split name {split} is not a valid split name. Please use "
                        f"one of the default split names: {DEFAULT_SPLIT_NAMES}."
                    )
            available = {
                split: f"HF hub dataset - {self.config.data_source} - {split}"
                for split in hf_dataset
            }
        else:
            available = {
                DEFAULT_SPLIT_NAMES[0]: f"HF hub dataset - {self.config.data_source}"
            }
        return hf_dataset, available, False

    def _load_from_inmemory(self):
        source = self.config.data_source

        if isinstance(source, DatasetDict):
            warnings.warn(
                'data_format="hf" with a DatasetDict: using the provided splits as-is. '
                f"Split names must follow {DEFAULT_SPLIT_NAMES}. "
                "val_data_source and test_data_source are ignored."
            )
            available = {
                split: f"in-memory Dataset object - {split}" for split in source
            }
            return source, available, False

        if isinstance(source, Dataset):
            warnings.warn(
                'data_format="hf" with a Dataset: the dataset will be automatically split '
                "into train/val (and optionally test) according to the split configuration. "
                "val_data_source and test_data_source are ignored."
            )
            hf_dataset = DatasetDict({DEFAULT_SPLIT_NAMES[0]: source})
            available = {DEFAULT_SPLIT_NAMES[0]: "in-memory Dataset object"}
            return hf_dataset, available, False

        raise ValueError(
            "The provided data source is not a valid Hugging Face Dataset/DatasetDict "
            "object. The data_format value should be set to 'hf' if you plan to use an "
            "in-memory Hugging Face Dataset/DatasetDict object."
        )

    # ------------------------------------------------------------- split decision

    def _determine_split_mode(self, available: dict) -> _DatasetSplitMode:
        # Hub datasets are always predefined: trust the hub's split structure
        if self.config.data_format == "hub":
            return _DatasetSplitMode.PREDEFINED

        # In-memory DatasetDict is predefined by definition
        if self.config.data_format == "hf" and isinstance(
            self.config.data_source, DatasetDict
        ):
            return _DatasetSplitMode.PREDEFINED

        # Multiple sources provided → all splits are already defined externally
        if len(available) >= 2:
            return _DatasetSplitMode.PREDEFINED

        if DEFAULT_SPLIT_NAMES[2] in available:  # "test" only
            return _DatasetSplitMode.TEST_ONLY

        if DEFAULT_SPLIT_NAMES[1] in available:  # "val" only, no train
            return _DatasetSplitMode.PREDEFINED

        # Single "train" source (file or in-memory Dataset) → auto-split
        return _DatasetSplitMode.AUTO

    def _has_explicit_split_params(self) -> bool:
        return (
            self.config.val_ratio is not None
            or self.config.test_ratio is not None
            or (
                self.config.split_strategy
                and self.config.split_strategy.lower() != "random"
            )
            or self.config.stratify_by_column is not None
        )

    def _validate_and_warn(
        self, split_mode: _DatasetSplitMode, available: dict
    ) -> None:
        if split_mode == _DatasetSplitMode.PREDEFINED:
            warnings.warn(
                f"Using provided splits as-is: {list(available.keys())}. "
                "No automatic splitting will occur."
            )
        elif split_mode == _DatasetSplitMode.TEST_ONLY:
            warnings.warn(
                "Only a test split was provided. No automatic splitting will occur."
            )

        if split_mode != _DatasetSplitMode.AUTO and self.config.split_seed is not None:
            warnings.warn(
                f"split_seed={self.config.split_seed} is set but no automatic splitting "
                "will occur (splits are predefined or only a test set was provided). "
                "The seed is ignored."
            )

        if (
            split_mode == _DatasetSplitMode.PREDEFINED
            and self._has_explicit_split_params()
        ):
            raise ValueError(
                "Cannot use split configuration parameters (val_ratio, test_ratio, "
                "split_strategy, stratify_by_column) when providing predefined data "
                f"sources. Found predefined splits: {list(available.keys())}. Either "
                "provide only data_source for automatic splitting or provide predefined "
                "splits without split configuration parameters."
            )

        if (
            split_mode == _DatasetSplitMode.TEST_ONLY
            and self._has_explicit_split_params()
        ):
            raise ValueError(
                "Cannot use split configuration parameters (val_ratio, test_ratio, "
                "split_strategy, stratify_by_column) when only test data is provided — "
                "there is no training data to split. "
                "Either omit the split parameters or also provide data_source."
            )
