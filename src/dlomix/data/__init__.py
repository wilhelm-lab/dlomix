from .charge_state import ChargeStateDataset
from .dataset import PeptideDataset, load_processed_dataset
from .detectability import DetectabilityDataset
from .fragment_ion_intensity import FragmentIonIntensityDataset
from .ion_mobility import IonMobilityDataset
from .retention_time import RetentionTimeDataset

__all__ = [
    "RetentionTimeDataset",
    "FragmentIonIntensityDataset",
    "ChargeStateDataset",
    "PeptideDataset",
    "load_processed_dataset",
    "DetectabilityDataset",
    "IonMobilityDataset",
]

# Flag to track if info has been shown
_info_shown = False


def _show_features_info_once():
    """Show feature extractors info only once."""
    global _info_shown
    if not _info_shown:
        from .processing import show_available_features

        show_available_features()
        _info_shown = True


# Store original classes
_DATASET_CLASSES = {
    "RetentionTimeDataset": RetentionTimeDataset,
    "FragmentIonIntensityDataset": FragmentIonIntensityDataset,
    "ChargeStateDataset": ChargeStateDataset,
    "PeptideDataset": PeptideDataset,
    "DetectabilityDataset": DetectabilityDataset,
    "IonMobilityDataset": IonMobilityDataset,
}


def __getattr__(name):
    """Intercept attribute access to show info when dataset classes are imported."""
    if name in _DATASET_CLASSES:
        _show_features_info_once()
        return _DATASET_CLASSES[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
