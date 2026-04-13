import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import random

SEED = 2
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "GridPosition", "GridPosition_norm", "quali_delta_to_pole",
    "quali_delta_to_median", "rolling_avg_finish_5", "rolling_avg_grid_5",
    "rolling_podium_rate_5", "rolling_dnf_rate_5", "rolling_avg_points_5",
    "rolling_avg_finish_3", "rolling_podium_rate_3", "rolling_avg_finish_10",
    "rolling_podium_rate_10", "driver_encoded", "team_encoded",
    "is_street_circuit", "circuit_encoded", "air_temp_norm",
    "track_temp_norm", "humidity_norm", "wind_speed_norm", "Rainfall",
    "grid_x_rainfall", "driver_team_form", "form_vs_grid_gap",
]

TARGET_COL = "IsPodium"

HYPERPARAMS = {
    "learning_rate": 0.001,
    "epochs": 200,
    "batch_size": 32,
    "dropout_rate": 0.35,
    "hidden_layers": [64, 32, 16],
    "patience": 20,
    "pos_weight": 3.5,
}


def get_device():
    #Pick the best available device: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")
    return device

def load_and_prepare_data(device):
    #Load CSVs, extract features/targets, standardise, convert to tensors.
    train_df = pd.read_csv(os.path.join(DATA_DIR, "f1_features_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "f1_features_val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "f1_features_test.csv"))

    # Extract numpy arrays
    X_train = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    X_val = val_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_val = val_df[TARGET_COL].values.astype(np.float32)
    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    # Standardise using training stats only
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1.0

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # Save scaler for inference later
    np.savez(
        os.path.join(MODEL_DIR, "scaler_params.npz"),
        mean=train_mean, std=train_std,
    )

    # Convert to PyTorch tensors and move to device
    X_train_t = torch.tensor(X_train).to(device)
    y_train_t = torch.tensor(y_train).reshape(-1, 1).to(device)
    X_val_t = torch.tensor(X_val).to(device)
    y_val_t = torch.tensor(y_val).reshape(-1, 1).to(device)
    X_test_t = torch.tensor(X_test).to(device)
    y_test_t = torch.tensor(y_test).reshape(-1, 1).to(device)

    print(f"  Data loaded:")
    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    print(f"Podium ratio — train: {y_train.mean():.1%}, "
          f"val: {y_val.mean():.1%}, test: {y_test.mean():.1%}")

    data = {
        "X_train": X_train_t, "y_train": y_train_t,
        "X_val": X_val_t, "y_val": y_val_t,
        "X_test": X_test_t, "y_test": y_test_t,
        # Keep DataFrames for later analysis (driver names, etc.)
        "test_df": test_df,
    }

    return data

class F1PodiumNet(nn.Module):

    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_layers):
            # Linear layer: z = x·W^T + b
            layers.append(nn.Linear(prev_dim, h_dim))

            # ReLU activation: max(0, z)
            layers.append(nn.ReLU())

            # Dropout on all hidden layers except the last one
            if i < len(hidden_layers) - 1:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = h_dim

        # Output layer: single neuron with sigmoid activation
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        # nn.Sequential chains all layers into a single callable
        self.network = nn.Sequential(*layers)

        # Apply He initialisation to all linear layers
        self._init_weights()

    def _init_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)

class WeightedBCELoss(nn.Module):
    """
    Binary cross-entropy with per-sample class weighting.

    Podium samples (y=1) are weighted by pos_weight to counteract
    the 85/15 class imbalance. Without this, the model would learn
    to predict 0 for everything.
    """

    def __init__(self, pos_weight=5.5):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, y_pred, y_true):
        eps = 1e-12
        y_pred = torch.clamp(y_pred, eps, 1 - eps)

        # Per-sample weights: pos_weight for podiums, 1.0 for non-podiums
        weights = torch.where(y_true == 1, self.pos_weight, 1.0)

        # Standard BCE formula with weighting
        bce = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

        return (bce * weights).mean()


# METRICS
def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """Compute accuracy, precision, recall, F1 from tensors or arrays."""
    # Move to CPU and convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.cpu().numpy()

    y_true = y_true.flatten().astype(int)
    y_pred = (y_pred_probs.flatten() >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# TRAINING LOOP
def train(model, data, device, hyperparams=HYPERPARAMS):
    """
    Train the model with mini-batch SGD, class weighting, and early stopping.
    """
    lr = hyperparams["learning_rate"]
    epochs = hyperparams["epochs"]
    batch_size = hyperparams["batch_size"]
    patience = hyperparams["patience"]
    pos_weight = hyperparams["pos_weight"]

    # DataLoader handles batching and shuffling
    train_dataset = TensorDataset(data["X_train"], data["y_train"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimiser
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Training history
    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "val_precision": [], "val_recall": [], "lr": [],
    }

    # Early stopping state
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {model.__class__.__name__}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Optimiser: Adam (lr={lr})")
    print(f"  Loss: Weighted BCE (pos_weight={pos_weight})")
    print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")

    print(f"\n  {'Epoch':>6}  {'T.Loss':>8}  {'V.Loss':>8}  "
          f"{'V.F1':>6}  {'V.Prec':>7}  {'V.Rec':>6}  {'LR':>10}  {'':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*10}  {'─'*8}")

    for epoch in range(1, epochs + 1):
        # ── Training phase ──
        model.train()  # enable dropout
        batch_losses = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()          # clear previous gradients
            predictions = model(X_batch)   # forward pass
            loss = criterion(predictions, y_batch)  # compute loss
            loss.backward()                # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()               # update weights
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)

        # ── Validation phase ──
        model.eval()  # disable dropout
        with torch.no_grad():  # don't track gradients (saves memory)
            val_preds = model(data["X_val"])
            val_loss = criterion(val_preds, data["y_val"]).item()
            train_preds_full = model(data["X_train"])

        # Compute metrics
        val_metrics = compute_metrics(data["y_val"], val_preds)
        train_metrics = compute_metrics(data["y_train"], train_preds_full)

        # Step the scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["lr"].append(current_lr)

        # ── Early stopping ──
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # .state_dict() returns all learnable parameters
            # We clone to CPU so training can continue on GPU
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            status = "best"
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                status = "stop"

        # Print progress
        if epoch % 10 == 0 or epoch == 1 or status == "stop":
            print(f"  {epoch:6d}  {train_loss:8.4f}  {val_loss:8.4f}  "
                  f"{val_metrics['f1']:6.3f}  {val_metrics['precision']:7.3f}  "
                  f"{val_metrics['recall']:6.3f}  {current_lr:10.6f}  {status:>8}")

        if epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)
        print(f"  Restored best model (val_loss = {best_val_loss:.4f})")

    return model, history

def evaluate(model, X, y, label="Test"):
    """Run model on a dataset and print detailed metrics."""
    model.eval()
    with torch.no_grad():
        preds = model(X)

    metrics = compute_metrics(y, preds)

    print(f"\n  📊 {label} Set Results:")
    print(f"     Accuracy:  {metrics['accuracy']:.1%}")
    print(f"     Precision: {metrics['precision']:.1%} "
          f"(of predicted podiums, {metrics['precision']:.0%} were correct)")
    print(f"     Recall:    {metrics['recall']:.1%} "
          f"(of actual podiums, we found {metrics['recall']:.0%})")
    print(f"     F1 Score:  {metrics['f1']:.3f}")
    print(f"\n     Confusion Matrix:")
    print(f"       Predicted:    No Podium  |  Podium")
    print(f"       Actual No:    {metrics['tn']:5d}      |  {metrics['fp']:5d}   (false alarms)")
    print(f"       Actual Yes:   {metrics['fn']:5d}      |  {metrics['tp']:5d}   (correct catches)")

    return metrics, preds


def show_predictions(model, data, device, n=20):
    """Show individual predictions on the test set with driver names."""
    model.eval()
    test_df = data["test_df"]

    with torch.no_grad():
        preds = model(data["X_test"]).cpu().numpy().flatten()

    test_df = test_df.copy()
    test_df["PodiumProb"] = preds
    test_df["Predicted"] = (preds >= 0.5).astype(int)
    test_df["Correct"] = (test_df["Predicted"] == test_df["IsPodium"])

    # Show a sample of predictions
    cols = ["Driver", "Team", "EventName", "GridPosition", "IsPodium",
            "PodiumProb", "Predicted", "Correct"]
    available_cols = [c for c in cols if c in test_df.columns]

    print(f"\n  🏁 Sample Predictions (Test Set):")
    print(f"  {'─' * 80}")

    # Show some correct podium predictions and some misses
    podiums = test_df[test_df["IsPodium"] == 1].sort_values("PodiumProb", ascending=False)
    non_podiums = test_df[test_df["IsPodium"] == 0].sort_values("PodiumProb", ascending=False)

    print(f"\n  Top predicted podiums (actual podium finishers):")
    if len(podiums) > 0:
        print(podiums[available_cols].head(n // 2).to_string(index=False))

    print(f"\n  Highest false alarms (predicted podium but didn't):")
    false_alarms = non_podiums[non_podiums["Predicted"] == 1]
    if len(false_alarms) > 0:
        print(false_alarms[available_cols].head(n // 4).to_string(index=False))

    print(f"\n  Missed podiums (actual podium but predicted no):")
    misses = podiums[podiums["Predicted"] == 0]
    if len(misses) > 0:
        print(misses[available_cols].head(n // 4).to_string(index=False))
    else:
        print("    None! All podiums were caught.")


# Main
if __name__ == "__main__":
    print("🧠 Step 3: Model Training (PyTorch)")
    print("=" * 60)

    # Select device
    device = get_device()

    # Load data
    data = load_and_prepare_data(device)

    # Build model
    input_dim = data["X_train"].shape[1]
    model = F1PodiumNet(
        input_dim=input_dim,
        hidden_layers=HYPERPARAMS["hidden_layers"],
        dropout_rate=HYPERPARAMS["dropout_rate"],
    ).to(device)

    print(f"\n  Architecture:")
    print(f"  {model}")

    # Train
    model, history = train(model, data, device)

    # Save model
    model_path = os.path.join(MODEL_DIR, "f1_podium_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "hyperparams": HYPERPARAMS,
        "feature_columns": FEATURE_COLUMNS,
    }, model_path)
    print(f"  Model saved to {model_path}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(MODEL_DIR, "training_history.csv"), index=False)

    # Evaluate on all sets
    print("\n" + "═" * 60)
    evaluate(model, data["X_train"], data["y_train"], label="Train")
    evaluate(model, data["X_val"], data["y_val"], label="Validation")
    test_metrics, test_preds = evaluate(model, data["X_test"], data["y_test"], label="Test")

    # Show individual predictions
    print("\n" + "═" * 60)
    show_predictions(model, data, device)
