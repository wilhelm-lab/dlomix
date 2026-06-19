"""
Dataset splitting strategies for peptide datasets.

Provides random, stratified, and sequence-unique splitting via a common interface.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict


class SplitStrategy(str, Enum):
    """Enumeration of available dataset splitting strategies."""

    RANDOM = "random"
    SEQUENCE_UNIQUE = "sequence_unique"
    STRATIFIED = "stratified"

    @classmethod
    def _missing_(cls, value: object) -> "SplitStrategy":
        # Accept hyphens as well as underscores (e.g. "sequence-unique")
        if isinstance(value, str):
            normalized = value.lower().replace("-", "_")
            for member in cls:
                if member.value == normalized:
                    return member
        valid = [m.value for m in cls]
        raise ValueError(
            f"{value!r} is not a valid split strategy. " f"Valid options are: {valid}"
        )


@dataclass(frozen=True)
class SplitConfig:
    """
    Configuration for dataset splitting.

    Parameters
    ----------
    val_ratio : Optional[float]
        Fraction of data for the validation split. None or 0 means no val split. Default None.
    test_ratio : Optional[float]
        Fraction of data for the test split. None or 0 means no test split. Default None.
    strategy : str or SplitStrategy
        One of 'random', 'sequence_unique', 'stratified'. Default 'random'.
    seed : Optional[int]
        Random seed for reproducibility. Default None.
    stratify_column : Optional[str]
        Column to stratify on. Required for 'stratified' strategy. Default None.
    sequence_column : str
        Sequence column name. Used by 'sequence_unique' strategy. Default 'sequence'.

    At least one of val_ratio or test_ratio must be provided and > 0.
    """

    val_ratio: Optional[float] = None
    test_ratio: Optional[float] = None
    strategy: str = "random"
    seed: Optional[int] = None
    stratify_column: Optional[str] = None
    sequence_column: str = "sequence"

    def __post_init__(self):
        if isinstance(self.strategy, str):
            object.__setattr__(self, "strategy", SplitStrategy(self.strategy.lower()))

        # Treat 0 the same as None: no split for that portion
        if self.val_ratio == 0:
            object.__setattr__(self, "val_ratio", None)
        if self.test_ratio == 0:
            object.__setattr__(self, "test_ratio", None)

        if self.val_ratio is None and self.test_ratio is None:
            raise ValueError(
                "At least one of val_ratio or test_ratio must be set and > 0."
            )

        if self.val_ratio is not None and not 0 < self.val_ratio < 1:
            raise ValueError(f"val_ratio must be between 0 and 1, got {self.val_ratio}")

        if self.test_ratio is not None and not 0 < self.test_ratio < 1:
            raise ValueError(
                f"test_ratio must be between 0 and 1, got {self.test_ratio}"
            )

        if self.val_ratio is not None and self.test_ratio is not None:
            if self.val_ratio + self.test_ratio >= 1:
                raise ValueError(
                    f"val_ratio + test_ratio must be < 1, got {self.val_ratio + self.test_ratio}"
                )

        if self.strategy == SplitStrategy.STRATIFIED and self.stratify_column is None:
            raise ValueError(
                "stratify_column must be provided for stratified splitting strategy"
            )

        if (
            self.strategy == SplitStrategy.SEQUENCE_UNIQUE
            and self.sequence_column is None
        ):
            raise ValueError(
                "sequence_column must be provided for sequence_unique splitting strategy"
            )


class DatasetSplitter(ABC):
    """Abstract base class for dataset splitting strategies."""

    def __init__(self, config: SplitConfig):
        self.config = config

    def _check_min_split_size(
        self, n_samples: int, ratio: float, split_names: list
    ) -> None:
        """Raise ValueError if any split would be empty."""
        n_second = max(1, round(n_samples * ratio))
        n_first = n_samples - n_second
        if n_first < 1 or n_second < 1:
            raise ValueError(
                f"Dataset too small for the requested split ratios: "
                f"{n_samples} samples with ratio {ratio:.2f} would produce "
                f"splits {split_names} with sizes {n_first} / {n_second}. "
                f"Provide more data or reduce the split ratio."
            )

    @abstractmethod
    def split(self, dataset: Dataset) -> DatasetDict:
        """
        Split a dataset according to the configured strategy.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace Dataset to split.

        Returns
        -------
        DatasetDict
            Dictionary with 'train' and any combination of 'val' / 'test' splits.
        """

    def _perform_two_way_split(
        self,
        dataset: Dataset,
        ratio: float,
        second_name: str = "val",
        seed: Optional[int] = None,
        **kwargs,
    ) -> DatasetDict:
        self._check_min_split_size(len(dataset), ratio, ["train", second_name])
        split = dataset.train_test_split(test_size=ratio, seed=seed, **kwargs)
        return DatasetDict({"train": split["train"], second_name: split["test"]})

    def _perform_three_way_split(
        self,
        dataset: Dataset,
        val_size: float,
        test_size: float,
        seed: Optional[int] = None,
        **kwargs,
    ) -> DatasetDict:
        self._check_min_split_size(len(dataset), test_size, ["train", "val", "test"])
        first = dataset.train_test_split(test_size=test_size, seed=seed, **kwargs)
        # val_size is expressed as a fraction of the full dataset; adjust for the remaining portion
        adjusted_val = val_size / (1 - test_size)
        second = first["train"].train_test_split(
            test_size=adjusted_val, seed=seed, **kwargs
        )
        return DatasetDict(
            {"train": second["train"], "val": second["test"], "test": first["test"]}
        )


class RandomSplitter(DatasetSplitter):
    """Splits dataset randomly into train/val, train/test, or train/val/test sets."""

    def split(self, dataset: Dataset) -> DatasetDict:
        val, test = self.config.val_ratio, self.config.test_ratio
        if val and test:
            return self._perform_three_way_split(
                dataset, val_size=val, test_size=test, seed=self.config.seed
            )
        if val:
            return self._perform_two_way_split(
                dataset, ratio=val, second_name="val", seed=self.config.seed
            )
        return self._perform_two_way_split(
            dataset, ratio=test, second_name="test", seed=self.config.seed  # type: ignore[arg-type]
        )


class StratifiedSplitter(DatasetSplitter):
    """Splits dataset while preserving the class distribution of a given column."""

    def split(self, dataset: Dataset) -> DatasetDict:
        if self.config.stratify_column not in dataset.column_names:
            raise ValueError(
                f"Stratification column '{self.config.stratify_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        # HF train_test_split requires ClassLabel feature type for stratification.
        # Auto-cast if the column is a plain Value type so users don't need to do this manually.
        if not isinstance(dataset.features[self.config.stratify_column], ClassLabel):
            warnings.warn(
                f"Stratification column '{self.config.stratify_column}' is not ClassLabel type "
                f"(got {dataset.features[self.config.stratify_column]}). "
                f"Auto-casting to ClassLabel for stratified splitting."
            )
            dataset = dataset.class_encode_column(self.config.stratify_column)

        val, test = self.config.val_ratio, self.config.test_ratio
        strat = self.config.stratify_column
        if val and test:
            return self._perform_three_way_split(
                dataset,
                val_size=val,
                test_size=test,
                seed=self.config.seed,
                stratify_by_column=strat,
            )
        if val:
            return self._perform_two_way_split(
                dataset,
                ratio=val,
                second_name="val",
                seed=self.config.seed,
                stratify_by_column=strat,
            )
        return self._perform_two_way_split(
            dataset,
            ratio=test,
            second_name="test",
            seed=self.config.seed,  # type: ignore[arg-type]
            stratify_by_column=strat,
        )


class SequenceUniqueSplitter(DatasetSplitter):
    """
    Splits dataset so that each base sequence appears in exactly one split.

    PTM notation (e.g. [UNIMOD:1], [+57]) is stripped before grouping, so
    PEPTIDE and PEP[UNIMOD:1]TIDE are treated as the same sequence.
    """

    def split(self, dataset: Dataset) -> DatasetDict:
        if self.config.sequence_column not in dataset.column_names:
            raise ValueError(
                f"Sequence column '{self.config.sequence_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)

        # Strip PTM notation before computing unique sequences so that
        # PEPTIDE and PEP[UNIMOD:1]TIDE map to the same base sequence.
        base_sequences = df[self.config.sequence_column].str.replace(
            r"\[.*?\]", "", regex=True
        )
        unique_sequences = base_sequences.unique()
        n_unique = len(unique_sequences)

        rng = np.random.default_rng(self.config.seed)
        shuffled = unique_sequences.copy()
        rng.shuffle(shuffled)

        val, test = self.config.val_ratio, self.config.test_ratio

        if val and test:
            test_size = max(1, int(n_unique * test))
            val_size = max(1, int(n_unique * val))
            if n_unique - test_size - val_size < 1:
                raise ValueError(
                    f"Dataset too small: {n_unique} unique base sequences with "
                    f"val_ratio={val}, test_ratio={test} would leave no train sequences."
                )
            test_seqs = shuffled[:test_size]
            val_seqs = shuffled[test_size : test_size + val_size]
            train_seqs = shuffled[test_size + val_size :]
        elif val:
            val_size = max(1, int(n_unique * val))
            if n_unique - val_size < 1:
                raise ValueError(
                    f"Dataset too small: {n_unique} unique base sequences with "
                    f"val_ratio={val} would leave no train sequences."
                )
            val_seqs = shuffled[:val_size]
            train_seqs = shuffled[val_size:]
            test_seqs = None
        else:
            assert test is not None
            test_size = max(1, int(n_unique * test))
            if n_unique - test_size < 1:
                raise ValueError(
                    f"Dataset too small: {n_unique} unique base sequences with "
                    f"test_ratio={test} would leave no train sequences."
                )
            test_seqs = shuffled[:test_size]
            train_seqs = shuffled[test_size:]
            val_seqs = None

        train_mask = base_sequences.isin(train_seqs)
        result = DatasetDict(
            {"train": Dataset.from_pandas(df[train_mask], preserve_index=False)}
        )
        split_bases = {"train": set(base_sequences[train_mask])}

        if val_seqs is not None:
            val_mask = base_sequences.isin(val_seqs)
            result["val"] = Dataset.from_pandas(df[val_mask], preserve_index=False)
            split_bases["val"] = set(base_sequences[val_mask])

        if test_seqs is not None:
            test_mask = base_sequences.isin(test_seqs)
            result["test"] = Dataset.from_pandas(df[test_mask], preserve_index=False)
            split_bases["test"] = set(base_sequences[test_mask])

        # Verify no base-sequence leakage across splits
        names = list(split_bases)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                overlap = split_bases[a] & split_bases[b]
                if overlap:
                    warnings.warn(
                        f"Base-sequence overlap between {a} and {b}: {len(overlap)} sequences"
                    )

        return result


def create_splitter(config: SplitConfig) -> DatasetSplitter:
    """
    Create the appropriate splitter for the given configuration.

    Examples
    --------
    >>> config = SplitConfig(val_ratio=0.2, strategy='random', seed=42)
    >>> splitter = create_splitter(config)
    >>> split_data = splitter.split(dataset)

    >>> config = SplitConfig(val_ratio=0.2, strategy='stratified', stratify_column='label', seed=42)
    >>> splitter = create_splitter(config)
    >>> split_data = splitter.split(dataset)

    >>> config = SplitConfig(val_ratio=0.15, test_ratio=0.15, strategy='sequence_unique', seed=42)
    >>> splitter = create_splitter(config)
    >>> split_data = splitter.split(dataset)
    """
    strategy_map = {
        SplitStrategy.RANDOM: RandomSplitter,
        SplitStrategy.STRATIFIED: StratifiedSplitter,
        SplitStrategy.SEQUENCE_UNIQUE: SequenceUniqueSplitter,
    }
    return strategy_map[config.strategy](config)  # type: ignore[index]
