import torch
import torch.nn as nn
import torch.optim as optim

from dlomix.constants import PTMS_ALPHABET
from dlomix.data import ChargeStateDataset
from dlomix.models import ChargeStatePredictorTorch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


TESTING_DATA = "example_dataset/chargestate/chargestate_data.parquet"

d = ChargeStateDataset(
    data_format="parquet",  # "hub",
    data_source=TESTING_DATA,  # "Wilhelmlab/prospect-ptms-charge",
    sequence_column="modified_sequence",
    label_column="charge_state_dist",
    max_seq_len=30,
    batch_size=8,
    dataset_type="pt",
)
print(d)
for x in d.tensor_train_data:
    print(x)
    break

test_d = ChargeStateDataset(
    data_format="parquet",  # "hub",
    test_data_source=TESTING_DATA,  # "Wilhelmlab/prospect-ptms-charge",
    sequence_column="modified_sequence",
    label_column="charge_state_dist",
    max_seq_len=30,
    batch_size=8,
    dataset_type="pt",
)


model = ChargeStatePredictorTorch(
    num_classes=6, seq_length=30, alphabet=PTMS_ALPHABET, model_flavour="relative"
)
print(model)
model.to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.1,  # lr factor
    patience=2,  # lr patience
    min_lr=1e-7,
)

# Early stopping tracking.
best_val_loss = float("inf")
epochs_without_improvement = 0
best_model_state = None

# Prepare a list to store metrics per epoch for CSV logging.
metrics_log = (
    []
)  # Each element will be a dict with keys: epoch, train_loss, val_loss, test_loss, learning_rate

for epoch in range(1, 2):
    model.train()
    running_loss = 0.0
    local_step = 0

    for batch in d.tensor_train_data:
        train_seq = batch["modified_sequence"]
        train_label = batch["charge_state_dist"]

        # Ensure tensors are on the correct device and type
        train_seq = train_seq.to(device, dtype=torch.int32)
        train_label = train_label.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        pred_cs = model(train_seq)
        loss = criterion(pred_cs, train_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if local_step % 25 == 0:
            avg_loss = running_loss / (local_step + 1)
            print(f"Epoch {epoch}, Step {local_step}, Training Loss: {avg_loss:.4f}")
        local_step += 1

    avg_train_loss = running_loss / len(d.tensor_train_data)

    # Validation phase.
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for batch in d.tensor_val_data:
            val_seq = batch["modified_sequence"]
            val_label = batch["charge_state_dist"]

            # Ensure tensors are on the correct device and type
            val_seq = val_seq.to(device, dtype=torch.int32)
            val_label = val_label.to(device, dtype=torch.float32)

            val_pred_cs = model(val_seq)
            val_loss = criterion(val_pred_cs, val_label)
            val_loss_total += val_loss.item()

            # TODO add adjusted_mean_absolute_error metric

    avg_val_loss = val_loss_total / len(d.tensor_val_data)
    print(
        f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
    )

    # Learning rate scheduler using the validation loss.
    scheduler.step(avg_val_loss)
    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch}: Current Learning Rate: {current_lr}")

    # Early stopping check.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        best_model_state = model.state_dict()
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")
        if epochs_without_improvement >= 5:  # patience
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
# if args.save_path:
#     torch.save(model.state_dict(), args.save_path)
#     print(f"Best model saved to {args.save_path}")


# Test phase.
model.eval()
test_loss_total = 0.0
with torch.no_grad():
    for batch in test_d.tensor_test_data:
        test_seq = batch["modified_sequence"]
        test_label = batch["charge_state_dist"]

        # Ensure tensors are on the correct device and type
        test_seq = test_seq.to(device, dtype=torch.int32)
        test_label = test_label.to(device, dtype=torch.float32)

        test_pred_cs = model(test_seq)
        test_loss = criterion(test_pred_cs, test_label)

        test_loss_total += test_loss.item()
avg_test_loss = test_loss_total / len(test_d.tensor_test_data)
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

# # Save the metrics log to CSV.
# df_metrics = pd.DataFrame(metrics_log)
# df_metrics.to_csv(args.metrics_csv, index=False)
# print(f"Metrics logged to CSV file: {args.metrics_csv}")


print("first 5 test sequences:\n", test_seq[:5])
print("first 5 test relative charge state vectors (label):\n", test_label[:5])
print("first 5 relative charge state predictions for test:\n", test_pred_cs[:5])
print(
    "predictions.shape for test set:",
    test_pred_cs.shape,
    "number of test CS vectors (label):",
    len(test_label),
)
