"""
Dataset splitting strategies for peptide datasets.

This module provides a flexible and extensible framework for splitting datasets
using various strategies including random, sequence-based uniqueness, and stratified
splitting.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


class SplitStrategy(str, Enum):
    """Enumeration of available dataset splitting strategies."""

    RANDOM = "random"
    SEQUENCE_UNIQUE = "sequence_unique"
    STRATIFIED = "stratified"


@dataclass(frozen=True)
class SplitConfig:
    """
    Configuration for dataset splitting.

    Parameters
    ----------
    val_ratio : float
        Ratio of validation data (0 < val_ratio < 1). Default is 0.2.
    test_ratio : Optional[float]
        Ratio of test data for three-way splits (0 < test_ratio < 1).
        If None, only train/val split is performed. Default is None.
    strategy : str or SplitStrategy
        Splitting strategy to use. Options: 'random', 'sequence_unique', 'stratified'.
        Default is 'random'.
    seed : Optional[int]
        Random seed for reproducibility. Default is None.
    stratify_column : Optional[str]
        Column name to use for stratified splitting. Can be a label column or
        any other feature column. Only used with 'stratified' strategy. Default is None.
    sequence_column : str
        Column name containing sequences. Used for 'sequence_unique' strategy.
        Default is 'sequence'.
    Raises
    ------
    ValueError
        If val_ratio or test_ratio are not in valid range (0, 1).
        If val_ratio + test_ratio >= 1.
        If stratify_column is not provided for stratified strategy.
        If sequence_column is not provided for sequence_unique strategy.
    """

    val_ratio: float = 0.2
    test_ratio: Optional[float] = None
    strategy: str = "random"
    seed: Optional[int] = None
    stratify_column: Optional[str] = None
    sequence_column: str = "sequence"

    def __post_init__(self):
        """Validate configuration parameters."""
        # Normalize strategy to enum
        if isinstance(self.strategy, str):
            object.__setattr__(self, "strategy", SplitStrategy(self.strategy.lower()))

        # Validate ratios
        if not 0 < self.val_ratio < 1:
            raise ValueError(f"val_ratio must be between 0 and 1, got {self.val_ratio}")

        if self.test_ratio is not None:
            if not 0 < self.test_ratio < 1:
                raise ValueError(
                    f"test_ratio must be between 0 and 1, got {self.test_ratio}"
                )
            if self.val_ratio + self.test_ratio >= 1:
                raise ValueError(
                    f"val_ratio + test_ratio must be < 1, got {self.val_ratio + self.test_ratio}"
                )

        # Validate strategy-specific requirements
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
    """
    Abstract base class for dataset splitting strategies.

    This class follows the processor pattern used in the dlomix codebase,
    providing a consistent interface for different splitting strategies.
    """

    def __init__(self, config: SplitConfig):
        """
        Initialize the splitter with a configuration.

        Parameters
        ----------
        config : SplitConfig
            Configuration object containing splitting parameters.
        """
        self.config = config

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
            Dictionary containing 'train', 'val', and optionally 'test' splits.
        """

    def _perform_two_way_split(
        self, dataset: Dataset, test_size: float, seed: Optional[int] = None, **kwargs
    ) -> DatasetDict:
        """
        Helper method to perform a two-way split.

        Parameters
        ----------
        dataset : Dataset
            Dataset to split.
        test_size : float
            Ratio of the second split.
        seed : Optional[int]
            Random seed for reproducibility.
        **kwargs
            Additional arguments passed to train_test_split.

        Returns
        -------
        DatasetDict
            Dictionary with 'train' and 'val' keys.
        """
        split_dataset = dataset.train_test_split(
            test_size=test_size, seed=seed, **kwargs
        )
        return DatasetDict(
            {"train": split_dataset["train"], "val": split_dataset["test"]}
        )

    def _perform_three_way_split(
        self,
        dataset: Dataset,
        val_size: float,
        test_size: float,
        seed: Optional[int] = None,
        **kwargs,
    ) -> DatasetDict:
        """
        Helper method to perform a three-way split.

        Parameters
        ----------
        dataset : Dataset
            Dataset to split.
        val_size : float
            Ratio of validation data.
        test_size : float
            Ratio of test data.
        seed : Optional[int]
            Random seed for reproducibility.
        **kwargs
            Additional arguments passed to train_test_split.

        Returns
        -------
        DatasetDict
            Dictionary with 'train', 'val', and 'test' keys.
        """
        # First split: separate test set
        first_split = dataset.train_test_split(test_size=test_size, seed=seed, **kwargs)
        test_dataset = first_split["test"]
        train_val_dataset = first_split["train"]

        # Calculate adjusted val ratio for remaining data
        adjusted_val_ratio = val_size / (1 - test_size)

        # Second split: separate train and val from remaining data
        second_split = train_val_dataset.train_test_split(
            test_size=adjusted_val_ratio, seed=seed, **kwargs
        )

        return DatasetDict(
            {
                "train": second_split["train"],
                "val": second_split["test"],
                "test": test_dataset,
            }
        )


class RandomSplitter(DatasetSplitter):
    """
    Random splitting strategy.

    Splits dataset randomly into train/val or train/val/test sets.
    Uses HuggingFace's train_test_split with optional seed for reproducibility.
    """

    def split(self, dataset: Dataset) -> DatasetDict:
        """
        Perform random splitting.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace Dataset to split.

        Returns
        -------
        DatasetDict
            Dictionary containing split datasets.
        """
        logger.info(
            "Performing random split with val_ratio=%.2f, test_ratio=%s, seed=%s",
            self.config.val_ratio,
            self.config.test_ratio,
            self.config.seed,
        )

        if self.config.test_ratio is None:
            # Two-way split: train/val
            return self._perform_two_way_split(
                dataset, test_size=self.config.val_ratio, seed=self.config.seed
            )
        else:
            # Three-way split: train/val/test
            return self._perform_three_way_split(
                dataset,
                val_size=self.config.val_ratio,
                test_size=self.config.test_ratio,
                seed=self.config.seed,
            )


class StratifiedSplitter(DatasetSplitter):
    """
    Stratified splitting strategy.

    Splits dataset while maintaining the distribution of a specified column
    (e.g., label column or any feature column) across splits.
    """

    def split(self, dataset: Dataset) -> DatasetDict:
        """
        Perform stratified splitting.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace Dataset to split.

        Returns
        -------
        DatasetDict
            Dictionary containing split datasets.

        Raises
        ------
        ValueError
            If stratify_column is not found in the dataset.
        """
        # Validate stratify column exists
        if self.config.stratify_column not in dataset.column_names:
            raise ValueError(
                f"Stratification column '{self.config.stratify_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        logger.info(
            "Performing stratified split on column '%s' with val_ratio=%.2f, test_ratio=%s, seed=%s",
            self.config.stratify_column,
            self.config.val_ratio,
            self.config.test_ratio,
            self.config.seed,
        )

        if self.config.test_ratio is None:
            # Two-way split: train/val
            return self._perform_two_way_split(
                dataset,
                test_size=self.config.val_ratio,
                seed=self.config.seed,
                stratify_by_column=self.config.stratify_column,
            )
        else:
            # Three-way split: train/val/test
            return self._perform_three_way_split(
                dataset,
                val_size=self.config.val_ratio,
                test_size=self.config.test_ratio,
                seed=self.config.seed,
                stratify_by_column=self.config.stratify_column,
            )


class SequenceUniqueSplitter(DatasetSplitter):
    """
    Sequence-based unique splitting strategy.

    Ensures that sequences are unique across train/val splits and optionally
    test split. This is important for preventing data leakage when the same
    sequence appears in both training and validation sets.
    """

    def split(self, dataset: Dataset) -> DatasetDict:
        """
        Perform sequence-unique splitting.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace Dataset to split.

        Returns
        -------
        DatasetDict
            Dictionary containing split datasets with unique sequences.

        Raises
        ------
        ValueError
            If sequence_column is not found in the dataset.
        """
        # Validate sequence column exists
        if self.config.sequence_column not in dataset.column_names:
            raise ValueError(
                f"Sequence column '{self.config.sequence_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        logger.info(
            "Performing sequence-unique split with val_ratio=%.2f, test_ratio=%s, seed=%s",
            self.config.val_ratio,
            self.config.test_ratio,
            self.config.seed,
        )

        # Convert to pandas for easier groupby operations
        import pandas as pd

        df: pd.DataFrame = dataset.to_pandas()

        # Group by sequence to get unique sequences
        unique_sequences = df[self.config.sequence_column].unique()
        n_unique = len(unique_sequences)

        logger.info(
            "Dataset contains %d total samples with %d unique sequences",
            len(df),
            n_unique,
        )

        # Create a dataset of unique sequences for splitting
        import numpy as np

        rng = np.random.default_rng(self.config.seed)

        # Shuffle unique sequences
        shuffled_sequences = unique_sequences.copy()
        rng.shuffle(shuffled_sequences)

        # Calculate split indices
        if self.config.test_ratio is None:
            # Two-way split
            val_size = int(n_unique * self.config.val_ratio)
            train_sequences = shuffled_sequences[val_size:]
            val_sequences = shuffled_sequences[:val_size]
            test_sequences = None

            logger.info(
                "Split into %d train sequences, %d val sequences",
                len(train_sequences),
                len(val_sequences),
            )
        else:
            # Three-way split
            test_size = int(n_unique * self.config.test_ratio)
            val_size = int(n_unique * self.config.val_ratio)

            test_sequences = shuffled_sequences[:test_size]
            val_sequences = shuffled_sequences[test_size : test_size + val_size]
            train_sequences = shuffled_sequences[test_size + val_size :]

            logger.info(
                "Split into %d train sequences, %d val sequences, %d test sequences",
                len(train_sequences),
                len(val_sequences),
                len(test_sequences),
            )

        # Filter dataframe by sequence membership
        train_mask = df[self.config.sequence_column].isin(train_sequences)
        val_mask = df[self.config.sequence_column].isin(val_sequences)

        train_df = df[train_mask]
        val_df = df[val_mask]

        logger.info(
            "Train split: %d samples, Val split: %d samples",
            len(train_df),
            len(val_df),
        )

        # Convert back to HuggingFace datasets
        result = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, preserve_index=False),
                "val": Dataset.from_pandas(val_df, preserve_index=False),
            }
        )

        # Handle test split
        if test_sequences is not None:
            test_df = df[df[self.config.sequence_column].isin(test_sequences)]
            logger.info("Test split: %d samples", len(test_df))
            result["test"] = Dataset.from_pandas(test_df, preserve_index=False)

        # Verify uniqueness
        train_seqs = set(result["train"][self.config.sequence_column])
        val_seqs = set(result["val"][self.config.sequence_column])

        if "test" in result:
            test_seqs = set(result["test"][self.config.sequence_column])
            train_val_overlap = train_seqs & val_seqs
            train_test_overlap = train_seqs & test_seqs
            val_test_overlap = val_seqs & test_seqs

            if train_val_overlap or train_test_overlap or val_test_overlap:
                warnings.warn(
                    f"Sequence overlap detected after splitting: "
                    f"train-val: {len(train_val_overlap)}, "
                    f"train-test: {len(train_test_overlap)}, "
                    f"val-test: {len(val_test_overlap)}"
                )
        else:
            overlap = train_seqs & val_seqs
            if overlap:
                warnings.warn(
                    f"Sequence overlap detected between train and val: {len(overlap)} sequences"
                )

        return result


def create_splitter(config: SplitConfig) -> DatasetSplitter:
    """
    Factory method to create the appropriate splitter based on configuration.

    Parameters
    ----------
    config : SplitConfig
        Configuration object specifying the splitting strategy and parameters.

    Returns
    -------
    DatasetSplitter
        Concrete splitter instance based on the specified strategy.

    Raises
    ------
    ValueError
        If an unknown strategy is specified.

    Examples
    --------
    >>> # Random splitting
    >>> config = SplitConfig(val_ratio=0.2, strategy='random', seed=42)
    >>> splitter = create_splitter(config)
    >>> split_data = splitter.split(dataset)

    >>> # Stratified splitting by label
    >>> config = SplitConfig(val_ratio=0.2, strategy='stratified', stratify_column='label', seed=42)
    >>> splitter = create_splitter(config)
    >>> split_data = splitter.split(dataset)

    >>> # Sequence-unique splitting with three-way split
    >>> config = SplitConfig(
    ...     val_ratio=0.15,
    ...     test_ratio=0.15,
    ...     strategy='sequence_unique',
    ...     sequence_column='sequence',
    ...     seed=42
    ... )
    >>> splitter = create_splitter(config)
    >>> split_data = splitter.split(dataset)
    """
    strategy_map = {
        SplitStrategy.RANDOM: RandomSplitter,
        SplitStrategy.STRATIFIED: StratifiedSplitter,
        SplitStrategy.SEQUENCE_UNIQUE: SequenceUniqueSplitter,
    }

    splitter_class = strategy_map.get(config.strategy)

    if splitter_class is None:
        raise ValueError(
            f"Unknown splitting strategy: {config.strategy}. "
            f"Available strategies: {list(strategy_map.keys())}"
        )

    return splitter_class(config)
