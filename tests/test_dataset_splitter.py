"""
Tests for dataset splitting strategies.

This module contains comprehensive tests for the dataset splitter functionality,
including all splitting strategies, backward compatibility, edge cases, and
reproducibility checks.
"""

import logging

import pytest
from datasets import Dataset

from dlomix.data import SplitConfig, SplitStrategy, create_splitter
from dlomix.data.dataset_splitter import (
    RandomSplitter,
    SequenceUniqueSplitter,
    StratifiedSplitter,
)

logger = logging.getLogger(__name__)


# Fixtures
@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    data = {
        "sequence": [f"PEPTIDE{i}" for i in range(100)],
        "label": [i % 3 for i in range(100)],  # 3 classes for stratification
        "feature": [float(i) for i in range(100)],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def duplicate_sequence_dataset():
    """Create a dataset with duplicate sequences."""
    sequences = [
        f"PEPTIDE{i % 20}" for i in range(100)
    ]  # 20 unique sequences, 100 samples
    data = {
        "sequence": sequences,
        "label": [i % 3 for i in range(100)],
        "feature": [float(i) for i in range(100)],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def imbalanced_dataset():
    """Create an imbalanced dataset for stratification testing."""
    from datasets import ClassLabel, Features, Value

    data = {
        "sequence": [f"PEPTIDE{i}" for i in range(100)],
        "label": [0] * 70 + [1] * 20 + [2] * 10,  # Imbalanced classes
        "feature": [float(i) for i in range(100)],
    }

    # Define features with ClassLabel for stratification
    features = Features(
        {
            "sequence": Value("string"),
            "label": ClassLabel(names=["class0", "class1", "class2"]),
            "feature": Value("float32"),
        }
    )

    return Dataset.from_dict(data, features=features)


# Tests for SplitConfig
class TestSplitConfig:
    """Test SplitConfig dataclass validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = SplitConfig()
        assert config.val_ratio == 0.2
        assert config.test_ratio is None
        assert config.strategy == SplitStrategy.RANDOM
        assert config.seed is None

    def test_invalid_val_ratio(self):
        """Test validation of val_ratio."""
        with pytest.raises(ValueError, match="val_ratio must be between 0 and 1"):
            SplitConfig(val_ratio=1.5)

        with pytest.raises(ValueError, match="val_ratio must be between 0 and 1"):
            SplitConfig(val_ratio=-0.1)

        with pytest.raises(ValueError, match="val_ratio must be between 0 and 1"):
            SplitConfig(val_ratio=0)

    def test_invalid_test_ratio(self):
        """Test validation of test_ratio."""
        with pytest.raises(ValueError, match="test_ratio must be between 0 and 1"):
            SplitConfig(test_ratio=1.5)

        with pytest.raises(ValueError, match="test_ratio must be between 0 and 1"):
            SplitConfig(test_ratio=-0.1)

    def test_invalid_combined_ratios(self):
        """Test validation when val_ratio + test_ratio >= 1."""
        with pytest.raises(ValueError, match="val_ratio \\+ test_ratio must be < 1"):
            SplitConfig(val_ratio=0.6, test_ratio=0.5)

    def test_stratified_without_column(self):
        """Test that stratified strategy requires stratify_column."""
        with pytest.raises(ValueError, match="stratify_column must be provided"):
            SplitConfig(strategy="stratified")

    def test_sequence_unique_without_column(self):
        """Test that sequence_unique strategy requires sequence_column."""
        with pytest.raises(ValueError, match="sequence_column must be provided"):
            SplitConfig(strategy="sequence_unique", sequence_column=None)

    def test_strategy_normalization(self):
        """Test that strategy string is normalized to enum."""
        config = SplitConfig(strategy="RANDOM")
        assert config.strategy == SplitStrategy.RANDOM

        config = SplitConfig(strategy="sequence_unique")
        assert config.strategy == SplitStrategy.SEQUENCE_UNIQUE


# Tests for RandomSplitter
class TestRandomSplitter:
    """Test random splitting strategy."""

    def test_two_way_split(self, simple_dataset):
        """Test basic two-way split (train/val)."""
        config = SplitConfig(val_ratio=0.2, strategy="random", seed=42)
        splitter = create_splitter(config)

        result = splitter.split(simple_dataset)

        assert "train" in result
        assert "val" in result
        assert "test" not in result

        assert len(result["train"]) == 80
        assert len(result["val"]) == 20
        assert len(result["train"]) + len(result["val"]) == len(simple_dataset)

    def test_three_way_split(self, simple_dataset):
        """Test three-way split (train/val/test)."""
        config = SplitConfig(val_ratio=0.2, test_ratio=0.1, strategy="random", seed=42)
        splitter = create_splitter(config)

        result = splitter.split(simple_dataset)

        assert "train" in result
        assert "val" in result
        assert "test" in result

        # Test has 10% of data
        assert len(result["test"]) == 10
        # Remaining 90 samples split 80/20 for train/val -> ~72/18
        assert len(result["train"]) + len(result["val"]) == 90
        assert len(result["train"]) + len(result["val"]) + len(result["test"]) == len(
            simple_dataset
        )

    def test_reproducibility(self, simple_dataset):
        """Test that same seed produces same splits."""
        config1 = SplitConfig(val_ratio=0.2, strategy="random", seed=42)
        splitter1 = create_splitter(config1)
        result1 = splitter1.split(simple_dataset)

        config2 = SplitConfig(val_ratio=0.2, strategy="random", seed=42)
        splitter2 = create_splitter(config2)
        result2 = splitter2.split(simple_dataset)

        # Same sequences should be in train/val
        assert result1["train"]["sequence"] == result2["train"]["sequence"]
        assert result1["val"]["sequence"] == result2["val"]["sequence"]

    def test_different_seeds(self, simple_dataset):
        """Test that different seeds produce different splits."""
        config1 = SplitConfig(val_ratio=0.2, strategy="random", seed=42)
        splitter1 = create_splitter(config1)
        result1 = splitter1.split(simple_dataset)

        config2 = SplitConfig(val_ratio=0.2, strategy="random", seed=123)
        splitter2 = create_splitter(config2)
        result2 = splitter2.split(simple_dataset)

        # Different sequences should be in train (with high probability)
        assert result1["train"]["sequence"] != result2["train"]["sequence"]


# Tests for StratifiedSplitter
class TestStratifiedSplitter:
    """Test stratified splitting strategy."""

    def test_stratified_two_way_split(self, imbalanced_dataset):
        """Test stratified splitting maintains class distribution."""
        config = SplitConfig(
            val_ratio=0.2, strategy="stratified", stratify_column="label", seed=42
        )
        splitter = create_splitter(config)

        result = splitter.split(imbalanced_dataset)

        # Check that splits exist
        assert "train" in result
        assert "val" in result

        # Calculate class distributions
        train_labels = result["train"]["label"]
        val_labels = result["val"]["label"]

        # Original distribution: 70% class 0, 20% class 1, 10% class 2
        # Check approximate distribution is maintained
        train_class_0 = sum(1 for l in train_labels if l == 0)
        train_class_1 = sum(1 for l in train_labels if l == 1)
        train_class_2 = sum(1 for l in train_labels if l == 2)

        # Should be roughly 70%, 20%, 10%
        assert 0.65 < train_class_0 / len(train_labels) < 0.75
        assert 0.15 < train_class_1 / len(train_labels) < 0.25
        assert 0.05 < train_class_2 / len(train_labels) < 0.15

    def test_stratified_three_way_split(self, imbalanced_dataset):
        """Test stratified three-way split."""
        config = SplitConfig(
            val_ratio=0.15,
            test_ratio=0.15,
            strategy="stratified",
            stratify_column="label",
            seed=42,
        )
        splitter = create_splitter(config)

        result = splitter.split(imbalanced_dataset)

        assert "train" in result
        assert "val" in result
        assert "test" in result

        # All samples accounted for
        total = len(result["train"]) + len(result["val"]) + len(result["test"])
        assert total == len(imbalanced_dataset)

    def test_stratified_with_invalid_column(self, simple_dataset):
        """Test error when stratify column doesn't exist."""
        config = SplitConfig(
            val_ratio=0.2, strategy="stratified", stratify_column="nonexistent"
        )
        splitter = create_splitter(config)

        with pytest.raises(ValueError, match="Stratification column.*not found"):
            splitter.split(simple_dataset)

    def test_stratified_reproducibility(self, imbalanced_dataset):
        """Test reproducibility of stratified splits."""
        config1 = SplitConfig(
            val_ratio=0.2, strategy="stratified", stratify_column="label", seed=42
        )
        splitter1 = create_splitter(config1)
        result1 = splitter1.split(imbalanced_dataset)

        config2 = SplitConfig(
            val_ratio=0.2, strategy="stratified", stratify_column="label", seed=42
        )
        splitter2 = create_splitter(config2)
        result2 = splitter2.split(imbalanced_dataset)

        assert result1["train"]["sequence"] == result2["train"]["sequence"]
        assert result1["val"]["sequence"] == result2["val"]["sequence"]


# Tests for SequenceUniqueSplitter
class TestSequenceUniqueSplitter:
    """Test sequence-unique splitting strategy."""

    def test_sequence_unique_two_way_split(self, duplicate_sequence_dataset):
        """Test that sequences are unique across train/val."""
        config = SplitConfig(
            val_ratio=0.2,
            strategy="sequence_unique",
            sequence_column="sequence",
            seed=42,
        )
        splitter = create_splitter(config)

        result = splitter.split(duplicate_sequence_dataset)

        # Get unique sequences from each split
        train_sequences = set(result["train"]["sequence"])
        val_sequences = set(result["val"]["sequence"])

        # Check no overlap
        overlap = train_sequences & val_sequences
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping sequences"

        # Check that splits exist
        assert len(result["train"]) > 0
        assert len(result["val"]) > 0

    def test_sequence_unique_three_way_split_with_uniqueness(
        self, duplicate_sequence_dataset
    ):
        """Test three-way split with test uniqueness enabled."""
        config = SplitConfig(
            val_ratio=0.2,
            test_ratio=0.2,
            strategy="sequence_unique",
            sequence_column="sequence",
            seed=42,
        )
        splitter = create_splitter(config)

        result = splitter.split(duplicate_sequence_dataset)

        train_sequences = set(result["train"]["sequence"])
        val_sequences = set(result["val"]["sequence"])
        test_sequences = set(result["test"]["sequence"])

        # Check no overlaps
        assert len(train_sequences & val_sequences) == 0
        assert len(train_sequences & test_sequences) == 0
        assert len(val_sequences & test_sequences) == 0

    def test_sequence_unique_with_invalid_column(self, simple_dataset):
        """Test error when sequence column doesn't exist."""
        config = SplitConfig(
            val_ratio=0.2, strategy="sequence_unique", sequence_column="nonexistent"
        )
        splitter = create_splitter(config)

        with pytest.raises(ValueError, match="Sequence column.*not found"):
            splitter.split(simple_dataset)

    def test_sequence_unique_reproducibility(self, duplicate_sequence_dataset):
        """Test reproducibility of sequence-unique splits."""
        config1 = SplitConfig(
            val_ratio=0.2,
            strategy="sequence_unique",
            sequence_column="sequence",
            seed=42,
        )
        splitter1 = create_splitter(config1)
        result1 = splitter1.split(duplicate_sequence_dataset)

        config2 = SplitConfig(
            val_ratio=0.2,
            strategy="sequence_unique",
            sequence_column="sequence",
            seed=42,
        )
        splitter2 = create_splitter(config2)
        result2 = splitter2.split(duplicate_sequence_dataset)

        # Should produce identical splits
        train_seq1 = set(result1["train"]["sequence"])
        train_seq2 = set(result2["train"]["sequence"])
        assert train_seq1 == train_seq2

    def test_sequence_unique_sample_counts(self, duplicate_sequence_dataset):
        """Test that samples are correctly distributed across splits."""
        # Dataset has 20 unique sequences, 100 total samples
        config = SplitConfig(
            val_ratio=0.2,
            strategy="sequence_unique",
            sequence_column="sequence",
            seed=42,
        )
        splitter = create_splitter(config)

        result = splitter.split(duplicate_sequence_dataset)

        # All samples should be in either train or val
        total_samples = len(result["train"]) + len(result["val"])
        assert total_samples == len(duplicate_sequence_dataset)

        # Should have roughly 4 unique sequences in val (20% of 20)
        val_unique = len(set(result["val"]["sequence"]))
        train_unique = len(set(result["train"]["sequence"]))
        assert 3 <= val_unique <= 5  # Allow some variance
        assert train_unique + val_unique == 20  # All unique sequences accounted for


# Tests for create_splitter factory
class TestCreateSplitter:
    """Test the create_splitter factory function."""

    def test_create_random_splitter(self):
        """Test creating a random splitter."""
        config = SplitConfig(strategy="random")
        splitter = create_splitter(config)
        assert isinstance(splitter, RandomSplitter)

    def test_create_stratified_splitter(self):
        """Test creating a stratified splitter."""
        config = SplitConfig(strategy="stratified", stratify_column="label")
        splitter = create_splitter(config)
        assert isinstance(splitter, StratifiedSplitter)

    def test_create_sequence_unique_splitter(self):
        """Test creating a sequence-unique splitter."""
        config = SplitConfig(strategy="sequence_unique")
        splitter = create_splitter(config)
        assert isinstance(splitter, SequenceUniqueSplitter)

    def test_invalid_strategy(self):
        """Test that SplitConfig rejects unknown strategy strings."""
        with pytest.raises(ValueError):
            SplitConfig(strategy="invalid_strategy")


# Edge cases
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_dataset(self):
        """Test splitting a very small dataset."""
        data = {
            "sequence": ["PEP1", "PEP2", "PEP3", "PEP4", "PEP5"],
            "label": [0, 1, 0, 1, 0],
        }
        dataset = Dataset.from_dict(data)

        config = SplitConfig(val_ratio=0.2, strategy="random", seed=42)
        splitter = create_splitter(config)
        result = splitter.split(dataset)

        assert len(result["train"]) + len(result["val"]) == len(dataset)

    def test_large_val_ratio(self):
        """Test with large validation ratio."""
        data = {
            "sequence": [f"PEP{i}" for i in range(100)],
            "label": [i % 2 for i in range(100)],
        }
        dataset = Dataset.from_dict(data)

        config = SplitConfig(val_ratio=0.8, strategy="random", seed=42)
        splitter = create_splitter(config)
        result = splitter.split(dataset)

        assert len(result["val"]) == 80
        assert len(result["train"]) == 20

    def test_single_unique_sequence(self, simple_dataset):
        """Test sequence-unique split when all sequences are unique."""
        # simple_dataset has unique sequences
        config = SplitConfig(
            val_ratio=0.2,
            strategy="sequence_unique",
            sequence_column="sequence",
            seed=42,
        )
        splitter = create_splitter(config)
        result = splitter.split(simple_dataset)

        # Should still work correctly
        assert len(result["train"]) + len(result["val"]) == len(simple_dataset)

        # No overlapping sequences
        train_seqs = set(result["train"]["sequence"])
        val_seqs = set(result["val"]["sequence"])
        assert len(train_seqs & val_seqs) == 0
