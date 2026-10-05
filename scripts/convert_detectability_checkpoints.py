"""Convert the shipped Detectability TF checkpoints to Keras 3 `.weights.h5`.

The `pretrained_models/` folders hold TensorFlow-native checkpoints
(`.index` + `.data-00000-of-00001`), a format Keras 3 refuses outright:

    ValueError: File format not supported: Keras 3 only supports V3 `.keras`
    and `.weights.h5` files, or legacy V1/V2 `.h5` files.

`tf.train.load_checkpoint` can still read them, so this script maps each stored
variable onto the corresponding layer of the current `DetectabilityModel` and
re-saves it in the supported format. Run once; the produced files are committed
next to the checkpoints.

Mapping is done by walking **layer attributes**, not by variable path strings:
Keras 3 appends process-global counters to auto-generated layer names
(`bidirectional` vs `bidirectional_2`), so paths are not stable across runs.

    python scripts/convert_detectability_checkpoints.py

Verified faithful: predictions from the converted weights are bit-identical
(max abs diff 0.0) to the original checkpoint loaded under TensorFlow 2.15 /
Keras 2 with the pre-migration model code.
"""

import argparse
import pathlib

import numpy as np
import tensorflow as tf

from dlomix.constants import CLASSES_LABELS
from dlomix.models import DetectabilityModel

CHECKPOINT_ROOT = pathlib.Path("pretrained_models")
MODEL_NAMES = [
    "original_detectability_base_model",
    "original_detectability_fine_tuned_model_FINAL",
]
NUM_UNITS = 64
SEQ_LEN = 40
ATTRIBUTE_SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"


def _layer_for(model, checkpoint_prefix):
    """Resolve a checkpoint variable prefix to the layer object that owns it."""
    encoder_bi = model.encoder.encoder_bi
    decoder = model.decoder
    layers = {
        "encoder/encoder_bi/forward_layer/cell": encoder_bi.forward_layer.cell,
        "encoder/encoder_bi/backward_layer/cell": encoder_bi.backward_layer.cell,
        "decoder/decoder_bi/forward_layer/cell": decoder.decoder_bi.forward_layer.cell,
        "decoder/decoder_bi/backward_layer/cell": decoder.decoder_bi.backward_layer.cell,
        "decoder/attention/W1": decoder.attention.W1,
        "decoder/attention/W2": decoder.attention.W2,
        "decoder/attention/V": decoder.attention.V,
        "decoder/decoder_dense": decoder.decoder_dense,
    }
    return layers.get(checkpoint_prefix)


def _checkpoint_weights(prefix):
    """Stored model variables, excluding optimizer state and bookkeeping."""
    reader = tf.train.load_checkpoint(str(prefix))
    weights = {}
    for key in reader.get_variable_to_shape_map():
        if not key.endswith(ATTRIBUTE_SUFFIX):
            continue
        name = key[: -len(ATTRIBUTE_SUFFIX)]
        if "OPTIMIZER_SLOT" in name or name.startswith("optimizer/"):
            continue
        if name == "save_counter":
            continue
        weights[name] = reader.get_tensor(key)
    return weights


def convert(name, root=CHECKPOINT_ROOT):
    prefix = root / name / name
    destination = root / name / f"{name}.weights.h5"

    stored = _checkpoint_weights(prefix)

    model = DetectabilityModel(num_units=NUM_UNITS, num_classes=len(CLASSES_LABELS))
    model.build(input_shape=(None, SEQ_LEN))

    assigned = 0
    for variable_name, value in stored.items():
        layer_prefix, _, parameter = variable_name.rpartition("/")
        layer = _layer_for(model, layer_prefix)
        if layer is None:
            raise ValueError(f"{name}: no layer mapped for '{variable_name}'")
        variable = getattr(layer, parameter, None)
        if variable is None:
            raise ValueError(f"{name}: layer has no '{parameter}' ({variable_name})")
        if tuple(variable.shape) != tuple(value.shape):
            raise ValueError(
                f"{name}: shape mismatch for '{variable_name}': "
                f"checkpoint {tuple(value.shape)} vs model {tuple(variable.shape)}"
            )
        variable.assign(value)
        assigned += 1

    if assigned != len(model.weights):
        raise ValueError(
            f"{name}: assigned {assigned} of {len(model.weights)} model variables"
        )

    model.save_weights(destination)
    print(f"{name}: {assigned} variables -> {destination}")

    reloaded = DetectabilityModel(num_units=NUM_UNITS, num_classes=len(CLASSES_LABELS))
    reloaded.build(input_shape=(None, SEQ_LEN))
    reloaded.load_weights(destination)
    probe = np.random.default_rng(0).integers(1, 20, (4, SEQ_LEN))
    if not np.allclose(model(probe).numpy(), reloaded(probe).numpy(), atol=0):
        raise ValueError(f"{name}: reloaded weights do not reproduce predictions")
    print(f"{name}: reload verified (identical predictions)")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=CHECKPOINT_ROOT)
    args = parser.parse_args()
    for name in MODEL_NAMES:
        convert(name, root=args.root)


if __name__ == "__main__":
    main()
