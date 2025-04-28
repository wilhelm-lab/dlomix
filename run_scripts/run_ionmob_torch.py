#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim

from dlomix.data import IonMobilityDataset
from dlomix.losses import MaskedIonmobLoss
from dlomix.models import Ionmob


def main():
    parser = argparse.ArgumentParser(description="Train Ionmob model with PyTorch")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--gru1", type=int, default=64, help="First GRU hidden size")
    parser.add_argument("--gru2", type=int, default=32, help="Second GRU hidden size")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=50,
        help="Maximum sequence length for padded sequences",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience (in epochs)"
    )
    parser.add_argument(
        "--lr_factor", type=float, default=0.1, help="Learning rate reduction factor"
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=2,
        help="Learning rate scheduler patience (in epochs)",
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-7, help="Minimum learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'mps', or 'cpu'. Auto-detect if not specified.",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=25,
        help="Frequency (in steps) to print training loss",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./run_scripts/output/ionmob_best_model.pth",
        help="Path to save the best trained model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode (print additional information)",
    )
    # New arguments for logging and plotting:
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="./run_scripts/output/ionmob_metrics_log.csv",
        help="Path to save the CSV metrics log",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default="./run_scripts/output/ionmob_training_plots.png",
        help="Path to save the high resolution training plots image",
    )
    args = parser.parse_args()

    # Verbose: print out configuration settings
    if args.verbose:
        print("Configuration:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

    # Set device: use provided value or auto-detect
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create the dataset
    ds_huggingface = IonMobilityDataset(
        data_source="theGreatHerrLebert/ionmob",
        data_format="hub",
        dataset_type="pt",
        batch_size=args.batch_size,
    )

    # Create the model and move it to the selected device
    model = Ionmob(
        emb_dim=args.emb_dim,
        gru_1=args.gru1,
        gru_2=args.gru2,
        num_tokens=len(ds_huggingface.extended_alphabet),
    )
    model.to(device)

    # Loss function and optimizer.
    criterion = MaskedIonmobLoss(use_mse=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
    )

    # Early stopping tracking.
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    # Prepare a list to store metrics per epoch for CSV logging.
    metrics_log = (
        []
    )  # Each element will be a dict with keys: epoch, train_loss, val_loss, test_loss, learning_rate

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        local_step = 0

        for batch in ds_huggingface.tensor_train_data:
            seq = batch["sequence_modified"]
            mz = batch["mz"]
            charge = batch["charge"]
            target_ccs = batch["ccs"]
            target_ccs_std = batch["ccs_std"]

            # Ensure tensors are on the correct device and type
            seq = seq.to(device, dtype=torch.long)
            mz = mz.to(device, dtype=torch.float32)
            charge = charge.to(device, dtype=torch.long)
            target_ccs = torch.unsqueeze(target_ccs.to(device, dtype=torch.float32), -1)
            target_ccs_std = torch.unsqueeze(
                target_ccs_std.to(device, dtype=torch.float32), -1
            )

            optimizer.zero_grad()
            ccs_predicted, _, ccs_std_predicted = model(seq, mz, charge)
            loss = criterion(
                (ccs_predicted, ccs_std_predicted), (target_ccs, target_ccs_std)
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if local_step % args.print_freq == 0:
                avg_loss = running_loss / (local_step + 1)
                print(
                    f"Epoch {epoch}, Step {local_step}, Training Loss: {avg_loss:.4f}"
                )
            local_step += 1

        avg_train_loss = running_loss / len(ds_huggingface.tensor_train_data)

        # Validation phase.
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in ds_huggingface.tensor_val_data:
                seq = batch["sequence_modified"]
                mz = batch["mz"]
                charge = batch["charge"]
                target_ccs = batch["ccs"]
                target_ccs_std = batch["ccs_std"]

                # Ensure tensors are on the correct device and type
                seq = seq.to(device, dtype=torch.long)
                mz = mz.to(device, dtype=torch.float32)
                charge = charge.to(device, dtype=torch.long)
                target_ccs = torch.unsqueeze(
                    target_ccs.to(device, dtype=torch.float32), -1
                )
                target_ccs_std = torch.unsqueeze(
                    target_ccs_std.to(device, dtype=torch.float32), -1
                )

                ccs_predicted, _, ccs_std_predicted = model(seq, mz, charge)
                loss = criterion(
                    (ccs_predicted, ccs_std_predicted), (target_ccs, target_ccs_std)
                )
                val_loss_total += loss.item()
        avg_val_loss = val_loss_total / len(ds_huggingface.tensor_val_data)
        print(
            f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

        # Learning rate scheduler using the validation loss.
        scheduler.step(avg_val_loss)
        current_lr = scheduler.optimizer.param_groups[0]["lr"]
        if args.verbose:
            print(f"Epoch {epoch}: Current Learning Rate: {current_lr}")

        # Early stopping check.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                # Log current epoch metrics before breaking.
                metrics_log.append(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "test_loss": None,
                        "learning_rate": current_lr,
                    }
                )
                break

        # Log metrics for the current epoch.
        metrics_log.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "test_loss": None,  # will be filled later
                "learning_rate": current_lr,
            }
        )

    # Restore the best model.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state.")

    # Save the best model.
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Best model saved to {args.save_path}")

    # Test phase.
    model.eval()
    test_loss_total = 0.0
    with torch.no_grad():
        for batch in ds_huggingface.tensor_test_data:
            seq = batch["sequence_modified"]
            mz = batch["mz"]
            charge = batch["charge"]
            target_ccs = batch["ccs"]
            target_ccs_std = batch["ccs_std"]

            # Ensure tensors are on the correct device and type
            seq = seq.to(device, dtype=torch.long)
            mz = mz.to(device, dtype=torch.float32)
            charge = charge.to(device, dtype=torch.long)
            target_ccs = torch.unsqueeze(target_ccs.to(device, dtype=torch.float32), -1)
            target_ccs_std = torch.unsqueeze(
                target_ccs_std.to(device, dtype=torch.float32), -1
            )

            ccs_predicted, _, ccs_std_predicted = model(seq, mz, charge)
            loss = criterion(
                (ccs_predicted, ccs_std_predicted), (target_ccs, target_ccs_std)
            )
            test_loss_total += loss.item()
    avg_test_loss = test_loss_total / len(ds_huggingface.tensor_test_data)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Append final test metrics as an extra row in our log.
    metrics_log.append(
        {
            "epoch": "test",
            "train_loss": None,
            "val_loss": None,
            "test_loss": avg_test_loss,
            "learning_rate": None,
        }
    )

    # Save the metrics log to CSV.
    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(args.metrics_csv, index=False)
    print(f"Metrics logged to CSV file: {args.metrics_csv}")

    # Set a nice plotting style.
    sns.set_style("darkgrid")

    # Filter out the epochs (numerical entries only).
    epochs = [
        entry["epoch"] for entry in metrics_log if isinstance(entry["epoch"], int)
    ]
    train_losses = [
        entry["train_loss"] for entry in metrics_log if isinstance(entry["epoch"], int)
    ]
    val_losses = [
        entry["val_loss"] for entry in metrics_log if isinstance(entry["epoch"], int)
    ]
    lrs = [
        entry["learning_rate"]
        for entry in metrics_log
        if isinstance(entry["epoch"], int)
    ]

    # Create a figure with two subplots: one for loss and one for learning rate.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Plot training and validation loss.
    ax1.plot(epochs, train_losses, label="Train Loss", marker="o")
    ax1.plot(epochs, val_losses, label="Validation Loss", marker="s")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Add a text box with key model parameters and best validation loss.
    info_text = (
        f"Model: Ionmob\n"
        f"Emb dim: {args.emb_dim}\n"
        f"GRU1: {args.gru1}\n"
        f"GRU2: {args.gru2}\n"
        f"Batch size: {args.batch_size}\n"
        f"Initial LR: {args.lr}\n"
        f"Best Val Loss: {best_val_loss:.4f}\n"
        f"Test Loss: {avg_test_loss:.4f}"
    )
    ax1.text(
        0.95,
        0.95,
        info_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    # Plot learning rate over epochs.
    ax2.plot(epochs, lrs, label="Learning Rate", marker="^", color="orange")
    ax2.set_title("Learning Rate Schedule")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.plot_path)
    print(f"Training plots saved to {args.plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
