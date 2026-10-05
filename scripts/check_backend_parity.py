"""
Check that the TensorFlow and PyTorch Prosit intensity models agree.

DLOmix exposes the same model on both backends, so a model trained on either should
reach the same performance on the same data. This script checks that before longer
experiments are run, for ``PrositIntensityPredictor`` with precursor charge and
collision energy as metadata inputs. It runs three checks:

1. data      Both backends load the same fixed train/val/test files and must encode
             them identically (alphabet, encoded sequences, features, labels).
2. forward   The Keras weights are copied into the PyTorch model and both models run
             on the same batch. Identical implementations give the same output up to
             float precision; a difference is reported at the first stage where the two
             diverge (embedding, encoder, attention, metadata, decoder, output).
             No training and no seeds are involved. When training runs too, the
             comparison is repeated with the trained Keras weights, where the
             activations have realistic magnitudes (an untrained model's are tiny).
3. training  Each backend trains several runs from its own random initialization, with
             the same hyperparameters and early stopping on the validation loss. The
             runs are scored on the same test split with one NumPy implementation of
             the spectral angle. The gap between the backends is judged against the
             run-to-run spread within each backend, so no seeds need to be fixed.

Each backend runs in its own subprocess, since DLOmix fixes the backend at import time.

On macOS, TensorFlow runs on the CPU: on an Apple GPU, Keras GRUs either use
tensorflow-metal's fused kernel, which computes a different function (see
``dlomix.layers.gru_kernel``), or the standard kernel, which is much slower there.

Usage, from the repository root:

    python scripts/check_backend_parity.py                      # all checks
    python scripts/check_backend_parity.py --checks data forward  # quick, no training
    python scripts/check_backend_parity.py --repeats 3 --max-epochs 100
    python scripts/check_backend_parity.py --reuse-training     # redo checks, keep runs

Results (report, JSON, plots, predictions) are written to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

KERAS_EPSILON = 1e-7  # keras.config.epsilon(), used by the loss masking
L2_NORMALIZE_EPSILON = 1e-12  # floor used by the loss' l2 normalization

# ---------------------------------------------------------------------------
# Shared helpers (no backend imports)
# ---------------------------------------------------------------------------


def spectral_angle(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-spectrum spectral angle, mirroring ``dlomix.losses.masked_spectral_distance``.

    Returns ``1 - masked_spectral_distance``: 1 is a perfect match, 0 orthogonal.
    Computed in NumPy so both backends are scored by exactly the same code.
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + KERAS_EPSILON)
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + KERAS_EPSILON)

    def l2_normalize(x):
        square_sum = np.sum(np.square(x), axis=-1, keepdims=True)
        return x / np.sqrt(np.maximum(square_sum, L2_NORMALIZE_EPSILON))

    product = np.sum(l2_normalize(pred_masked) * l2_normalize(true_masked), axis=-1)
    distance = 2 * np.arccos(np.clip(product, -1.0, 1.0)) / np.pi
    return 1.0 - distance


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def prepare_splits(args) -> dict:
    """Write fixed train/val/test parquet files, split by unique peptide sequence."""
    import pandas as pd

    splits_dir = Path(args.output_dir) / "splits"
    paths = {s: splits_dir / f"{s}.parquet" for s in ("train", "val", "test")}
    if all(p.exists() for p in paths.values()) and not args.resplit:
        print(f"Reusing fixed splits in {splits_dir}")
        return {s: str(p) for s, p in paths.items()}

    df = pd.read_parquet(args.data)
    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.split_seed)

    # split by unique sequence so no peptide appears in two splits
    sequences = df[args.sequence_column].unique()
    rng = np.random.default_rng(args.split_seed)
    rng.shuffle(sequences)
    n_val = int(len(sequences) * args.val_ratio)
    n_test = int(len(sequences) * args.test_ratio)
    split_of = {}
    for i, seq in enumerate(sequences):
        split_of[seq] = (
            "test" if i < n_test else "val" if i < n_test + n_val else "train"
        )
    assignment = df[args.sequence_column].map(split_of)

    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, path in paths.items():
        df[assignment == split].reset_index(drop=True).to_parquet(path)
        print(f"  {split}: {(assignment == split).sum()} spectra -> {path}")
    return {s: str(p) for s, p in paths.items()}


# ---------------------------------------------------------------------------
# Worker side: runs inside a subprocess with DLOMIX_BACKEND set
# ---------------------------------------------------------------------------


def build_dataset(cfg, dataset_type):
    from dlomix.data import FragmentIonIntensityDataset

    return FragmentIonIntensityDataset(
        data_format="parquet",
        data_source=cfg["splits"]["train"],
        val_data_source=cfg["splits"]["val"],
        test_data_source=cfg["splits"]["test"],
        sequence_column=cfg["sequence_column"],
        label_column=cfg["label_column"],
        model_features=[cfg["charge_column"], cfg["ce_column"]],
        max_seq_len=cfg["max_seq_len"],
        batch_size=cfg["batch_size"],
        with_termini=cfg["with_termini"],
        shuffle=True,  # applies to the training split only
        dataset_type=dataset_type,
    )


def model_kwargs(cfg, alphabet):
    return dict(
        seq_length=cfg["max_seq_len"],
        with_termini=cfg["with_termini"],
        alphabet=alphabet,
        use_meta_data=True,
        input_keys={"SEQUENCE_KEY": cfg["sequence_column"]},
        meta_data_keys={
            "COLLISION_ENERGY_KEY": cfg["ce_column"],
            "PRECURSOR_CHARGE_KEY": cfg["charge_column"],
        },
    )


def split_arrays(dataset, cfg, split):
    """The encoded inputs and labels of one split, in dataset order."""
    hf = dataset.hf_dataset[split]
    return {
        "sequence": np.asarray(hf[cfg["sequence_column"]], dtype=np.int64),
        "charge": np.asarray(hf[cfg["charge_column"]], dtype=np.float32),
        "ce": np.asarray(hf[cfg["ce_column"]], dtype=np.float32),
        "labels": np.asarray(hf[cfg["label_column"]], dtype=np.float32),
    }


def save_data_fingerprint(dataset, cfg, backend, out_dir):
    fingerprint = {"alphabet": dataset.extended_alphabet}
    for split in ("train", "val", "test"):
        arrays = split_arrays(dataset, cfg, split)
        fingerprint[split] = {k: array_digest(v) for k, v in arrays.items()}
        fingerprint[split]["rows"] = int(len(arrays["labels"]))
    (out_dir / f"data_{backend}.json").write_text(json.dumps(fingerprint, indent=1))


# --- weight exchange: Keras layout <-> canonical numpy dict <-> PyTorch layout ---
#
# Canonical names follow the model structure. GRU weights are stored in the Keras
# layout: kernel (in, 3H), recurrent_kernel (H, 3H), bias (2, 3H) with input and
# recurrent biases (reset_after=True), gates ordered [z, r, h]. PyTorch orders the
# gates [r, z, n] and stores transposed matrices.


def keras_weights(model) -> dict:
    w = {}

    def gru(prefix, layer):
        cell = layer.cell
        w[f"{prefix}.kernel"] = cell.kernel.numpy()
        w[f"{prefix}.recurrent_kernel"] = cell.recurrent_kernel.numpy()
        w[f"{prefix}.bias"] = cell.bias.numpy()

    w["embedding"] = model.embedding.embeddings.numpy()
    bi_gru = model.sequence_encoder.layers[0]
    gru("encoder.gru1.forward", bi_gru.forward_layer)
    gru("encoder.gru1.backward", bi_gru.backward_layer)
    gru("encoder.gru2", model.sequence_encoder.layers[2])
    w["attention.W"] = model.attention.W.numpy()
    w["attention.b"] = model.attention.b.numpy()
    meta_dense = model.meta_encoder.layers[1]
    w["meta_dense.kernel"] = meta_dense.kernel.numpy()
    w["meta_dense.bias"] = meta_dense.bias.numpy()
    gru("decoder.gru", model.decoder.layers[0])
    w["decoder.attention.kernel"] = model.decoder.layers[2].dense.kernel.numpy()
    w["decoder.attention.bias"] = model.decoder.layers[2].dense.bias.numpy()
    time_dense = model.regressor.layers[0].layer
    w["time_dense.kernel"] = time_dense.kernel.numpy()
    w["time_dense.bias"] = time_dense.bias.numpy()
    return w


def load_weights_into_keras(model, w: dict) -> None:
    """Assign canonical weights to a built Keras model (the inverse of keras_weights)."""

    def gru(prefix, layer):
        layer.cell.kernel.assign(w[f"{prefix}.kernel"])
        layer.cell.recurrent_kernel.assign(w[f"{prefix}.recurrent_kernel"])
        layer.cell.bias.assign(w[f"{prefix}.bias"])

    model.embedding.embeddings.assign(w["embedding"])
    bi_gru = model.sequence_encoder.layers[0]
    gru("encoder.gru1.forward", bi_gru.forward_layer)
    gru("encoder.gru1.backward", bi_gru.backward_layer)
    gru("encoder.gru2", model.sequence_encoder.layers[2])
    model.attention.W.assign(w["attention.W"])
    model.attention.b.assign(w["attention.b"])
    model.meta_encoder.layers[1].kernel.assign(w["meta_dense.kernel"])
    model.meta_encoder.layers[1].bias.assign(w["meta_dense.bias"])
    gru("decoder.gru", model.decoder.layers[0])
    model.decoder.layers[2].dense.kernel.assign(w["decoder.attention.kernel"])
    model.decoder.layers[2].dense.bias.assign(w["decoder.attention.bias"])
    model.regressor.layers[0].layer.kernel.assign(w["time_dense.kernel"])
    model.regressor.layers[0].layer.bias.assign(w["time_dense.bias"])


def load_keras_weights_into_torch(model, w: dict) -> None:
    """Copy canonical (Keras-layout) weights into the PyTorch model, in place."""
    import torch

    def keras_gates_to_torch(matrix, axis=-1):
        z, r, h = np.split(matrix, 3, axis=axis)
        return np.concatenate([r, z, h], axis=axis)

    state = {}

    def gru(prefix, torch_prefix, suffix=""):
        state[f"{torch_prefix}.weight_ih_l0{suffix}"] = keras_gates_to_torch(
            w[f"{prefix}.kernel"]
        ).T
        state[f"{torch_prefix}.weight_hh_l0{suffix}"] = keras_gates_to_torch(
            w[f"{prefix}.recurrent_kernel"]
        ).T
        state[f"{torch_prefix}.bias_ih_l0{suffix}"] = keras_gates_to_torch(
            w[f"{prefix}.bias"][0]
        )
        state[f"{torch_prefix}.bias_hh_l0{suffix}"] = keras_gates_to_torch(
            w[f"{prefix}.bias"][1]
        )

    state["embedding.weight"] = w["embedding"]
    gru("encoder.gru1.forward", "sequence_encoder.bidirectional_GRU")
    gru("encoder.gru1.backward", "sequence_encoder.bidirectional_GRU", "_reverse")
    gru("encoder.gru2", "sequence_encoder.unidirectional_GRU")
    state["attention.W"] = w["attention.W"]
    state["attention.b"] = w["attention.b"]
    state["meta_encoder.meta_dense.weight"] = w["meta_dense.kernel"].T
    state["meta_encoder.meta_dense.bias"] = w["meta_dense.bias"]
    gru("decoder.gru", "decoder.unidirectional_GRU")
    state["decoder.attention.linear.weight"] = w["decoder.attention.kernel"].T
    state["decoder.attention.linear.bias"] = w["decoder.attention.bias"]
    state["regressor.time_dense.weight"] = w["time_dense.kernel"].T
    state["regressor.time_dense.bias"] = w["time_dense.bias"]

    own = model.state_dict()
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    if missing or unexpected:
        raise RuntimeError(
            f"Weight mapping does not cover the PyTorch model. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )
    for name, value in state.items():
        if tuple(own[name].shape) != value.shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: PyTorch {tuple(own[name].shape)}, "
                f"Keras {value.shape}. The two architectures differ here."
            )
    model.load_state_dict(
        {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in state.items()}
    )


def tf_forward_stages(model, batch: dict, cfg) -> dict:
    """Run the Keras model stage by stage in inference mode."""
    import tensorflow as tf

    x = {
        cfg["sequence_column"]: tf.constant(batch["sequence"]),
        cfg["charge_column"]: tf.constant(batch["charge"]),
        cfg["ce_column"]: tf.constant(batch["ce"]),
    }
    stages = {}
    stages["embedding"] = model.embedding(x[cfg["sequence_column"]])
    stages["encoder"] = model.sequence_encoder(stages["embedding"], training=False)
    stages["attention"] = model.attention(stages["encoder"])
    meta = model._collect_values_from_inputs_if_exists(x, model.meta_data_keys)
    stages["meta_encoder"] = model.meta_encoder(meta, training=False)
    stages["meta_fusion"] = model.meta_data_fusion_layer(
        [stages["attention"], stages["meta_encoder"]]
    )
    stages["decoder"] = model.decoder(stages["meta_fusion"], training=False)
    stages["output_dense"] = model.regressor.layers[0](stages["decoder"])
    stages["output"] = model(x, training=False)
    return {k: v.numpy() for k, v in stages.items()}


def torch_forward_stages(model, batch: dict, cfg) -> dict:
    """Run the PyTorch model stage by stage in inference mode."""
    import torch

    x = {
        cfg["sequence_column"]: torch.from_numpy(batch["sequence"]).long(),
        cfg["charge_column"]: torch.from_numpy(batch["charge"]),
        cfg["ce_column"]: torch.from_numpy(batch["ce"]),
    }
    stages = {}
    model.eval()
    with torch.no_grad():
        stages["embedding"] = model.embedding(x[cfg["sequence_column"]])
        stages["encoder"] = model.sequence_encoder(stages["embedding"])
        stages["attention"] = model.attention(stages["encoder"])
        meta = model._collect_values_from_inputs_if_exists(x, model.meta_data_keys)
        stages["meta_encoder"] = model.meta_encoder(torch.cat(meta, dim=-1))
        stages["meta_fusion"] = model.meta_data_fusion_layer(
            [stages["attention"], stages["meta_encoder"]]
        )
        stages["decoder"] = model.decoder(stages["meta_fusion"])
        stages["output_dense"] = model.regressor.time_dense(stages["decoder"])
        stages["output"] = model(x)
    return {k: v.numpy() for k, v in stages.items()}


def materialize_torch_model(model, batch, cfg):
    """Run one forward pass so the lazy layers create their weights."""
    torch_forward_stages(model, {k: v[:2] for k, v in batch.items()}, cfg)


def torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reuse_runs(cfg, out_dir: Path, backend: str) -> bool:
    """With --reuse-training, skip training when every run of this backend exists."""
    files = [out_dir / f"run_{backend}_run{r}.json" for r in range(cfg["repeats"])]
    if backend == "tensorflow":
        files.append(out_dir / "trained_weights_tensorflow_run0.npz")
    if cfg["reuse_training"] and all(f.exists() for f in files):
        print(f"training: reusing the existing {backend} runs in {out_dir}")
        return True
    return False


def worker_tensorflow(cfg, out_dir: Path) -> None:
    import tensorflow as tf

    if sys.platform == "darwin":
        # On an Apple GPU, Keras GRUs either use tensorflow-metal's fused kernel, which
        # computes a different function, or the standard kernel, which trains ~8x
        # slower there than on the CPU (Prosit intensity: 622 vs 78 s/epoch on an M1 Max)
        tf.config.set_visible_devices([], "GPU")  # before TensorFlow uses a GPU

    dataset = build_dataset(cfg, "tf")
    save_data_fingerprint(dataset, cfg, "tensorflow", out_dir)
    test = split_arrays(dataset, cfg, "test")
    np.save(out_dir / "test_labels.npy", test["labels"])
    np.savez(
        out_dir / "forward_batch.npz",
        **{k: v[: cfg["forward_batch"]] for k, v in test.items()},
    )

    if "training" in cfg["checks"] and not reuse_runs(cfg, out_dir, "tensorflow"):
        train_tensorflow(cfg, out_dir, dataset)
    if "forward" in cfg["checks"]:
        save_tensorflow_forward(cfg, out_dir, dataset)


def train_tensorflow(cfg, out_dir: Path, dataset) -> None:
    import keras

    from dlomix.losses import masked_spectral_distance
    from dlomix.models import PrositIntensityPredictor

    class RestoreBestWeights(keras.callbacks.Callback):
        """Keep the weights of the epoch with the lowest val_loss (as the PyTorch loop does)."""

        def __init__(self):
            super().__init__()
            self.best, self.best_weights, self.best_epoch = np.inf, None, 0

        def on_epoch_end(self, epoch, logs=None):
            if logs["val_loss"] < self.best:
                self.best, self.best_epoch = logs["val_loss"], epoch + 1
                self.best_weights = self.model.get_weights()

        def on_train_end(self, logs=None):
            self.model.set_weights(self.best_weights)

    for run in range(cfg["repeats"]):
        model = PrositIntensityPredictor(**model_kwargs(cfg, dataset.extended_alphabet))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
            loss=masked_spectral_distance,
        )
        if cfg["torch_init"] == "keras":
            # build, then hand the initial weights to the PyTorch worker
            model.predict(dataset.tensor_val_data.take(1), verbose=0)
            np.savez(out_dir / f"init_weights_run{run}.npz", **keras_weights(model))

        best = RestoreBestWeights()
        stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["patience"]
        )
        start = time.time()
        history = model.fit(
            dataset.tensor_train_data,
            validation_data=dataset.tensor_val_data,
            epochs=cfg["max_epochs"],
            callbacks=[best, stop],
            verbose=2,
        )
        predictions = model.predict(dataset.tensor_test_data, verbose=0)
        save_run(
            out_dir,
            "tensorflow",
            run,
            predictions,
            history.history,
            best.best_epoch,
            best.best,
            start,
        )
        if run == 0:
            np.savez(
                out_dir / "trained_weights_tensorflow_run0.npz", **keras_weights(model)
            )


def save_tensorflow_forward(cfg, out_dir: Path, dataset) -> None:
    """The Keras forward stages, with initial and (if trained) run 0's weights."""
    from dlomix.models import PrositIntensityPredictor

    batch = dict(np.load(out_dir / "forward_batch.npz"))

    model = PrositIntensityPredictor(**model_kwargs(cfg, dataset.extended_alphabet))
    stages = tf_forward_stages(model, batch, cfg)  # also builds every layer
    np.savez(out_dir / "forward_tensorflow.npz", **stages)
    np.savez(out_dir / "forward_weights.npz", **keras_weights(model))
    print(f"forward: saved {len(stages)} stages for {len(batch['labels'])} spectra")

    trained = out_dir / "trained_weights_tensorflow_run0.npz"
    if "training" in cfg["checks"] and trained.exists():
        load_weights_into_keras(model, dict(np.load(trained)))
        stages = tf_forward_stages(model, batch, cfg)
        np.savez(out_dir / "forward_trained_tensorflow.npz", **stages)
        print("forward: saved the stages with the trained weights of run 0")


def worker_pytorch(cfg, out_dir: Path) -> None:
    import torch

    from dlomix.losses import masked_spectral_distance
    from dlomix.models import PrositIntensityPredictor

    dataset = build_dataset(cfg, "pt")
    save_data_fingerprint(dataset, cfg, "pytorch", out_dir)
    test = split_arrays(dataset, cfg, "test")
    seq_col, charge_col, ce_col = (
        cfg["sequence_column"],
        cfg["charge_column"],
        cfg["ce_column"],
    )

    if "forward" in cfg["checks"]:
        batch = dict(np.load(out_dir / "forward_batch.npz"))
        model = PrositIntensityPredictor(**model_kwargs(cfg, dataset.extended_alphabet))
        materialize_torch_model(model, batch, cfg)
        load_keras_weights_into_torch(
            model, dict(np.load(out_dir / "forward_weights.npz"))
        )
        stages = torch_forward_stages(model, batch, cfg)
        np.savez(out_dir / "forward_pytorch.npz", **stages)
        print(f"forward: saved {len(stages)} stages for {len(batch['labels'])} spectra")

        trained = out_dir / "trained_weights_tensorflow_run0.npz"
        if "training" in cfg["checks"] and trained.exists():
            load_keras_weights_into_torch(model, dict(np.load(trained)))
            stages = torch_forward_stages(model, batch, cfg)
            np.savez(out_dir / "forward_trained_pytorch.npz", **stages)
            print("forward: saved the stages with the trained Keras weights")

    if "training" not in cfg["checks"] or reuse_runs(cfg, out_dir, "pytorch"):
        return

    device = torch_device()
    print(f"training on {device}")

    def to_device(batch):
        inputs = {
            seq_col: batch[seq_col].to(device, dtype=torch.long),
            charge_col: batch[charge_col].to(device, dtype=torch.float32),
            ce_col: batch[ce_col].to(device, dtype=torch.float32),
        }
        return inputs, batch[cfg["label_column"]].to(device, dtype=torch.float32)

    def mean_loss(model, loader):
        # sample-weighted mean over the split, as Keras reports val_loss
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = to_device(batch)
                losses = masked_spectral_distance(labels, model(inputs))
                total += float(losses.sum())
                count += len(labels)
        return total / count

    sample_batch = {k: v[:2] for k, v in test.items()}
    for run in range(cfg["repeats"]):
        model = PrositIntensityPredictor(**model_kwargs(cfg, dataset.extended_alphabet))
        materialize_torch_model(model, sample_batch, cfg)
        if cfg["torch_init"] == "keras":
            load_keras_weights_into_torch(
                model, dict(np.load(out_dir / f"init_weights_run{run}.npz"))
            )
        model.to(device)
        # Keras' Adam epsilon, so the two optimizers are configured identically
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["learning_rate"], eps=KERAS_EPSILON
        )

        history = {"loss": [], "val_loss": []}
        best_val_loss, best_epoch, best_state = np.inf, 0, None
        epochs_without_improvement = 0
        start = time.time()
        for epoch in range(cfg["max_epochs"]):
            model.train()
            total, count = 0.0, 0
            for batch in dataset.tensor_train_data:
                inputs, labels = to_device(batch)
                optimizer.zero_grad()
                losses = masked_spectral_distance(labels, model(inputs))
                losses.mean().backward()
                optimizer.step()
                total += float(losses.detach().sum())
                count += len(labels)
            history["loss"].append(total / count)
            history["val_loss"].append(mean_loss(model, dataset.tensor_val_data))
            print(
                f"epoch {epoch + 1}: loss {history['loss'][-1]:.4f} "
                f"val_loss {history['val_loss'][-1]:.4f}",
                flush=True,
            )

            if history["val_loss"][-1] < best_val_loss:
                best_val_loss, best_epoch = history["val_loss"][-1], epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg["patience"]:
                    break

        model.load_state_dict(best_state)
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataset.tensor_test_data:
                inputs, _ = to_device(batch)
                predictions.append(model(inputs).cpu().numpy())
        save_run(
            out_dir,
            "pytorch",
            run,
            np.concatenate(predictions),
            history,
            best_epoch,
            best_val_loss,
            start,
        )


def save_run(
    out_dir, backend, run, predictions, history, best_epoch, best_val_loss, start
):
    np.save(out_dir / f"predictions_{backend}_run{run}.npy", predictions)
    record = {
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "epochs_trained": len(history["val_loss"]),
        "seconds": round(time.time() - start, 1),
    }
    (out_dir / f"run_{backend}_run{run}.json").write_text(json.dumps(record, indent=1))
    print(
        f"{backend} run {run}: best epoch {record['best_epoch']} of "
        f"{record['epochs_trained']}, val_loss {record['best_val_loss']:.4f}, "
        f"{record['seconds']}s"
    )


# ---------------------------------------------------------------------------
# Orchestration and report (parent process)
# ---------------------------------------------------------------------------


def run_worker(worker: str, cfg: dict, out_dir: Path) -> None:
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))
    backend = "pytorch" if worker == "pytorch" else "tensorflow"
    # unbuffered, so progress reaches the log while the worker runs
    env = {**os.environ, "DLOMIX_BACKEND": backend, "PYTHONUNBUFFERED": "1"}
    print(f"\n=== {worker} worker ===", flush=True)
    log_path = out_dir / f"worker_{worker}.log"
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            [sys.executable, __file__, "--worker", worker, "--config", str(cfg_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            # surface progress lines (Keras prints the epoch metrics on a line
            # starting with the step count), keep the full output in the log
            if (
                line.startswith(("forward:", "training", "epoch"))
                or "val_loss" in line
                or (" run " in line and "best epoch" in line)
            ):
                print(f"  [{worker}] {line.rstrip()}", flush=True)
        if process.wait() != 0:
            sys.exit(f"The {worker} worker failed; see {log_path}")


def check_data(out_dir: Path) -> dict:
    tf_fp = json.loads((out_dir / "data_tensorflow.json").read_text())
    pt_fp = json.loads((out_dir / "data_pytorch.json").read_text())
    differences = [
        f"{split}.{key}"
        for split in ("train", "val", "test")
        for key in tf_fp[split]
        if tf_fp[split][key] != pt_fp[split][key]
    ]
    if tf_fp["alphabet"] != pt_fp["alphabet"]:
        differences.insert(0, "alphabet")
    return {"passed": not differences, "differences": differences}


def check_forward(out_dir: Path, rtol: float, atol: float, tag: str = "") -> dict:
    """Compare the stage outputs; a stage differs if max|a-b| > atol + rtol*max|a|."""
    tf_stages = np.load(out_dir / f"forward{tag}_tensorflow.npz")
    pt_stages = np.load(out_dir / f"forward{tag}_pytorch.npz")
    stages, first_divergence = {}, None
    for name in tf_stages.files:
        a, b = tf_stages[name], pt_stages[name]
        if a.shape != b.shape:
            stages[name] = {"shape_tensorflow": a.shape, "shape_pytorch": b.shape}
            first_divergence = first_divergence or name
            continue
        max_abs = float(np.max(np.abs(a - b)))
        scale = float(np.max(np.abs(a))) or 1.0
        stages[name] = {"max_abs_diff": max_abs, "max_rel_diff": max_abs / scale}
        if max_abs > atol + rtol * scale and first_divergence is None:
            first_divergence = name
    return {
        "passed": first_divergence is None,
        "first_divergence": first_divergence,
        "stages": stages,
    }


def check_training(out_dir: Path, cfg: dict) -> dict:
    labels = np.load(out_dir / "test_labels.npy")
    runs = {}
    for backend in ("tensorflow", "pytorch"):
        runs[backend] = []
        for run in range(cfg["repeats"]):
            predictions = np.load(out_dir / f"predictions_{backend}_run{run}.npy")
            record = json.loads((out_dir / f"run_{backend}_run{run}.json").read_text())
            angles = spectral_angle(labels, predictions)
            np.save(out_dir / f"spectral_angle_{backend}_run{run}.npy", angles)
            runs[backend].append(
                {
                    **{
                        k: record[k]
                        for k in ("best_epoch", "epochs_trained", "seconds")
                    },
                    "best_val_loss": record["best_val_loss"],
                    "median_sa": float(np.median(angles)),
                    "mean_sa": float(np.mean(angles)),
                }
            )

    summary = {}
    for backend, backend_runs in runs.items():
        medians = np.array([r["median_sa"] for r in backend_runs])
        summary[backend] = {
            "median_sa_mean": float(medians.mean()),
            "median_sa_sd": float(medians.std(ddof=1)) if len(medians) > 1 else 0.0,
        }
    gap = abs(
        summary["tensorflow"]["median_sa_mean"] - summary["pytorch"]["median_sa_mean"]
    )
    noise = max(summary[b]["median_sa_sd"] for b in summary)
    allowed = max(cfg["sa_tolerance"], 2 * noise)

    # Do the backends get the same spectra right and wrong? Correlate single runs
    # pairwise, so the cross-backend value is comparable to the within-backend one.
    angles = {
        b: [
            np.load(out_dir / f"spectral_angle_{b}_run{r}.npy")
            for r in range(cfg["repeats"])
        ]
        for b in runs
    }

    def mean_correlation(pairs):
        return float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in pairs]))

    per_spectrum_r = mean_correlation(
        [(a, b) for a in angles["tensorflow"] for b in angles["pytorch"]]
    )
    within_r = {
        b: mean_correlation(
            [
                (runs_b[i], runs_b[j])
                for i in range(len(runs_b))
                for j in range(i + 1, len(runs_b))
            ]
        )
        for b, runs_b in angles.items()
        if cfg["repeats"] > 1
    }
    return {
        "passed": gap <= allowed,
        "gap_median_sa": gap,
        "allowed_gap": allowed,
        "run_to_run_sd": noise,
        "per_spectrum_sa_correlation": per_spectrum_r,
        "within_backend_sa_correlation": within_r,
        "summary": summary,
        "runs": runs,
    }


def plot_training(out_dir: Path, cfg: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"tensorflow": "tab:orange", "pytorch": "tab:blue"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for backend, color in colors.items():
        for run in range(cfg["repeats"]):
            angles = np.load(out_dir / f"spectral_angle_{backend}_run{run}.npy")
            ax1.hist(
                angles,
                bins=50,
                range=(0, 1),
                histtype="step",
                color=color,
                label=f"{backend} run {run}",
            )
            record = json.loads((out_dir / f"run_{backend}_run{run}.json").read_text())
            ax2.plot(record["history"]["val_loss"], color=color, alpha=0.8)
    ax1.set(xlabel="test spectral angle", ylabel="spectra", title="Test spectral angle")
    ax1.legend(fontsize=8, loc="upper left")
    ax2.set(
        xlabel="epoch", ylabel="val loss (spectral distance)", title="Validation loss"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "training_parity.png", dpi=150)


def print_report(results: dict, cfg: dict) -> None:
    print(
        "\n"
        + "=" * 72
        + "\nBackend parity report: PrositIntensityPredictor\n"
        + "=" * 72
    )
    if "data" in results:
        r = results["data"]
        print(f"\n[data]     {'PASS' if r['passed'] else 'FAIL'}", end="")
        print("" if r["passed"] else f"  differs: {', '.join(r['differences'])}")
    for key, label in (
        ("forward", "initial weights"),
        ("forward_trained", "trained weights"),
    ):
        if key not in results:
            continue
        r = results[key]
        verdict = (
            "PASS"
            if r["passed"]
            else f"FAIL (first divergence: {r['first_divergence']})"
        )
        print(
            f"\n[{key}] {verdict}   Keras weights copied into PyTorch, {label}; "
            f"tolerance |diff| <= {cfg['forward_atol']} + {cfg['forward_rtol']} x max|value|"
        )
        for name, stage in r["stages"].items():
            if "max_rel_diff" in stage:
                print(
                    f"  {name:<14} max abs diff {stage['max_abs_diff']:.2e}   "
                    f"rel {stage['max_rel_diff']:.2e}"
                )
            else:
                print(f"  {name:<14} shapes differ: {stage}")
    if "training" in results:
        r = results["training"]
        print(f"\n[training] {'PASS' if r['passed'] else 'FAIL'}")
        print(
            f"  {'backend':<11} {'run':>3} {'best ep':>7} {'epochs':>6} {'val loss':>8} "
            f"{'median SA':>9} {'mean SA':>8} {'time':>7}"
        )
        for backend, runs in r["runs"].items():
            for i, run in enumerate(runs):
                print(
                    f"  {backend:<11} {i:>3} {run['best_epoch']:>7} {run['epochs_trained']:>6} "
                    f"{run['best_val_loss']:>8.4f} {run['median_sa']:>9.4f} "
                    f"{run['mean_sa']:>8.4f} {run['seconds']:>6.0f}s"
                )
        for backend, s in r["summary"].items():
            print(
                f"  {backend:<11} median SA {s['median_sa_mean']:.4f} +/- {s['median_sa_sd']:.4f} (sd over runs)"
            )
        print(
            f"  gap {r['gap_median_sa']:.4f}, allowed {r['allowed_gap']:.4f} "
            f"(max of --sa-tolerance and 2 x run-to-run sd)"
        )
        within = ", ".join(
            f"{b} {v:.3f}" for b, v in r["within_backend_sa_correlation"].items()
        )
        print(
            f"  per-spectrum SA correlation between backends: "
            f"{r['per_spectrum_sa_correlation']:.3f} (between runs of one backend: {within})"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        default=["data", "forward", "training"],
        choices=["data", "forward", "training"],
    )
    parser.add_argument(
        "--data",
        default="example_dataset/intensity/intensity_data.parquet",
        help="Parquet file with sequences, intensities, charge and CE.",
    )
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--label-column", default="intensities")
    parser.add_argument("--charge-column", default="precursor_charge_onehot")
    parser.add_argument("--ce-column", default="collision_energy_aligned_normed")
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument(
        "--with-termini",
        action="store_true",
        help="Keep the N-/C-terminal tokens (default: off, as in Prosit).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Subsample the data before splitting, for a quicker check.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed of the data split only; training is never seeded.",
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        help="Rewrite the fixed splits even if they exist.",
    )
    parser.add_argument(
        "--repeats", type=int, default=2, help="Training runs per backend."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=60,
        help="Upper bound; early stopping ended the Prosit intensity runs between "
        "epochs 36 and 58.",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--torch-init",
        choices=["torch", "keras"],
        default="torch",
        help="'keras' starts each PyTorch run from the Keras run's initial "
        "weights, isolating initialization from training differences.",
    )
    parser.add_argument("--forward-batch", type=int, default=64)
    parser.add_argument("--forward-rtol", type=float, default=1e-4)
    parser.add_argument("--forward-atol", type=float, default=1e-6)
    parser.add_argument(
        "--sa-tolerance",
        type=float,
        default=0.01,
        help="Smallest median-SA gap that is reported as a difference.",
    )
    parser.add_argument(
        "--reuse-training",
        action="store_true",
        help="Keep finished training runs in --output-dir and only redo the checks.",
    )
    parser.add_argument("--output-dir", default="run_scripts/output/backend_parity")
    parser.add_argument(
        "--worker",
        choices=["tensorflow", "pytorch"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--config", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        cfg = json.loads(Path(args.config).read_text())
        out_dir = Path(cfg["output_dir"])
        workers = {
            "tensorflow": worker_tensorflow,
            "pytorch": worker_pytorch,
        }
        workers[args.worker](cfg, out_dir)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in vars(args).items() if k not in ("worker", "config")}
    cfg["splits"] = prepare_splits(args)

    # TensorFlow trains first and exports the forward batch, its forward stages, the
    # trained weights and, with --torch-init keras, the initial weights of each run;
    # PyTorch then compares against them and trains last.
    run_worker("tensorflow", cfg, out_dir)
    run_worker("pytorch", cfg, out_dir)

    results = {"data": check_data(out_dir)}
    if "forward" in args.checks:
        results["forward"] = check_forward(
            out_dir, args.forward_rtol, args.forward_atol
        )
        if (
            out_dir / "forward_trained_pytorch.npz"
        ).exists() and "training" in args.checks:
            results["forward_trained"] = check_forward(
                out_dir, args.forward_rtol, args.forward_atol, tag="_trained"
            )
    if "training" in args.checks:
        results["training"] = check_training(out_dir, cfg)
        plot_training(out_dir, cfg)

    print_report(results, cfg)
    (out_dir / "parity_report.json").write_text(
        json.dumps(results, indent=1, default=str)
    )
    print(f"\nFull results in {out_dir}")
    verdicts = [r["passed"] for r in results.values() if "passed" in r]
    sys.exit(0 if all(verdicts) else 1)


if __name__ == "__main__":
    main()
