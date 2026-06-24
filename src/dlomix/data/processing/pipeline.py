"""
Build and apply the ordered sequence-processing pipeline for a peptide dataset.

``ProcessingPipeline`` turns the dataset configuration into an ordered list of
processors (parse → optional PTM removal → encode → optional pad → feature extractors)
and applies them across the dataset's splits. Per-split behavior (learn-then-apply
encoding, dropping truncated rows) lives on the processors themselves via
``apply_to_split``; the pipeline just iterates, keeping it open for extension.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from ..dataset_utils import EncodingScheme
from .chain import build_processing_chain
from .processors import (
    PeptideDatasetBaseProcessor,
    SequencePaddingProcessor,
    SequenceParsingProcessor,
)

DEFAULT_SPLIT_NAMES = ["train", "val", "test"]


@dataclass
class PipelineContext:
    """Mutable state shared with processors while applying the pipeline.

    ``alphabet`` is updated in place by the encoding processor as it learns the
    vocabulary on the fit splits; the dataset reads it back afterwards.
    """

    alphabet: dict
    num_proc: Optional[int]
    batch_size: int
    fit_splits: tuple


class ProcessingPipeline:
    """An ordered list of processors plus the columns they introduce."""

    def __init__(
        self,
        processors: List[PeptideDatasetBaseProcessor],
        extracted_feature_names: List[str],
        split_names: Optional[List[str]] = None,
    ):
        self.processors = processors
        self.extracted_feature_names = extracted_feature_names
        self.split_names = split_names or DEFAULT_SPLIT_NAMES

    @property
    def parsed_columns(self) -> List[str]:
        """Columns added by sequence parsing (kept in the HF dataset, not tensorized)."""
        return list(SequenceParsingProcessor.PARSED_COL_NAMES.values())

    @classmethod
    def from_config(
        cls,
        sequence_column: str,
        encoding_scheme: EncodingScheme,
        with_termini: bool,
        max_seq_len: int,
        padding_value: str,
        alphabet: dict,
        learning_alphabet_mode: bool,
        pad: bool,
        features_to_extract: Optional[List[Union[str, Callable]]] = None,
        split_names: Optional[List[str]] = None,
    ) -> "ProcessingPipeline":
        # training-time encoding: learn/extend the vocabulary on the fit splits
        # (per-split learn vs fallback is handled by the encoder's apply_to_split)
        processors, extracted_feature_names = build_processing_chain(
            sequence_column=sequence_column,
            encoding_scheme=encoding_scheme,
            with_termini=with_termini,
            max_seq_len=max_seq_len,
            padding_value=padding_value,
            alphabet=alphabet,
            pad=pad,
            features_to_extract=features_to_extract,
            encoding_extend_alphabet=learning_alphabet_mode,
        )
        return cls(processors, extracted_feature_names, split_names)

    def apply(self, hf_dataset, ctx: PipelineContext):
        """Apply every processor across all splits and drop the padding bookkeeping column."""
        for processor in self.processors:
            for split in self._split_order(hf_dataset, processor):
                hf_dataset[split] = processor.apply_to_split(
                    hf_dataset[split], split, ctx
                )

        first_split = next(iter(hf_dataset))
        if (
            SequencePaddingProcessor.KEEP_COLUMN_NAME
            in hf_dataset[first_split].column_names
        ):
            hf_dataset = hf_dataset.remove_columns(
                SequencePaddingProcessor.KEEP_COLUMN_NAME
            )

        return hf_dataset

    def _split_order(self, hf_dataset, processor) -> List[str]:
        splits = list(hf_dataset.keys())
        if not processor.requires_ordered_splits:
            return splits
        # fit splits (train/val) before eval (test) so the vocab is learned first
        ordered = [s for s in self.split_names if s in hf_dataset]
        return ordered + [s for s in splits if s not in ordered]
