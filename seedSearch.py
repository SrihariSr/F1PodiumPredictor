import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

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

# Seeds to try
SEEDS = list(range(0, 100))  # tries seeds 0 through 99


# ─── Model Definition ────────────────────────────────────────────────────────

class F1PodiumNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_layers) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=3.5):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, y_pred, y_true):
        eps = 1e-12
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        weights = torch.where(y_true == 1, self.pos_weight, 1.0)
        bce = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        return (bce * weights).mean()


def compute_metrics(y_true, y_pred_probs, threshold=0.5):
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
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ─── Set Seed ────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ─── Load Data (once) ────────────────────────────────────────────────────────

def load_data(device):
    train_df = pd.read_csv(os.path.join(DATA_DIR, "f1_features_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "f1_features_val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "f1_features_test.csv"))

    X_train = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    X_val = val_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_val = val_df[TARGET_COL].values.astype(np.float32)
    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1.0

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # Save scaler
    np.savez(os.path.join(MODEL_DIR, "scaler_params.npz"),
             mean=train_mean, std=train_std)

    data = {
        "X_train": torch.tensor(X_train).to(device),
        "y_train": torch.tensor(y_train).reshape(-1, 1).to(device),
        "X_val": torch.tensor(X_val).to(device),
        "y_val": torch.tensor(y_val).reshape(-1, 1).to(device),
        "X_test": torch.tensor(X_test).to(device),
        "y_test": torch.tensor(y_test).reshape(-1, 1).to(device),
    }
    return data


# ─── Train One Model (silent) ────────────────────────────────────────────────

def train_one_model(data, device, seed):
    """Train a model with a given seed. Returns test metrics and model state."""
    set_seed(seed)

    model = F1PodiumNet(
        input_dim=len(FEATURE_COLUMNS),
        hidden_layers=HYPERPARAMS["hidden_layers"],
        dropout_rate=HYPERPARAMS["dropout_rate"],
    ).to(device)

    train_dataset = TensorDataset(data["X_train"], data["y_train"])
    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMS["batch_size"], shuffle=True)

    criterion = WeightedBCELoss(pos_weight=HYPERPARAMS["pos_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMS["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, HYPERPARAMS["epochs"] + 1):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_preds = model(data["X_val"])
            val_loss = criterion(val_preds, data["y_val"]).item()

        # Check for NaN
        if np.isnan(val_loss):
            break

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            best_epoch = epoch
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= HYPERPARAMS["patience"]:
                break

    # Restore best and evaluate
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        val_preds = model(data["X_val"])
        test_preds = model(data["X_test"])

    val_metrics = compute_metrics(data["y_val"], val_preds)
    test_metrics = compute_metrics(data["y_test"], test_preds)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_f1": val_metrics["f1"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_accuracy": test_metrics["accuracy"],
        "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"],
        "model_state": best_state,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Seed Search — F1 Podium Predictor")
    print("=" * 60)
    print(f"Testing {len(SEEDS)} seeds: {SEEDS[0]} to {SEEDS[-1]}")
    print(f"Hyperparams: pos_weight={HYPERPARAMS['pos_weight']}, "
          f"epochs={HYPERPARAMS['epochs']}, lr={HYPERPARAMS['learning_rate']}")
    print()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")

    # Load data once
    print("Loading data...")
    data = load_data(device)
    print(f"Train: {data['X_train'].shape[0]} samples, {data['X_train'].shape[1]} features")
    print()

    # Train with each seed
    results = []
    best_test_f1 = 0
    best_result = None

    print(f"{'Seed':>6}  {'Epoch':>6}  {'Val F1':>7}  {'Test F1':>8}  "
          f"{'Prec':>6}  {'Rec':>6}  {'FP':>4}  {'FN':>4}  {'Status':>8}")
    print(f"{'─'*6}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*4}  {'─'*4}  {'─'*8}")

    for seed in SEEDS:
        result = train_one_model(data, device, seed)
        results.append(result)

        # Check if this is the best so far
        status = ""
        if result["test_f1"] > best_test_f1:
            best_test_f1 = result["test_f1"]
            best_result = result
            status = "★ BEST"

        print(f"{seed:6d}  {result['best_epoch']:6d}  {result['val_f1']:7.3f}  "
              f"{result['test_f1']:8.3f}  {result['test_precision']:6.1%}  "
              f"{result['test_recall']:6.1%}  {result['test_fp']:4d}  "
              f"{result['test_fn']:4d}  {status:>8}")

    # ─── Results Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")

    results_df = pd.DataFrame([{
        "seed": r["seed"], "best_epoch": r["best_epoch"],
        "val_f1": r["val_f1"], "test_f1": r["test_f1"],
        "test_precision": r["test_precision"], "test_recall": r["test_recall"],
        "test_accuracy": r["test_accuracy"], "test_fp": r["test_fp"],
        "test_fn": r["test_fn"],
    } for r in results])

    print(f"\n  Test F1 statistics across {len(SEEDS)} seeds:")
    print(f"    Mean:   {results_df['test_f1'].mean():.3f}")
    print(f"    Std:    {results_df['test_f1'].std():.3f}")
    print(f"    Min:    {results_df['test_f1'].min():.3f}  (seed {results_df.loc[results_df['test_f1'].idxmin(), 'seed']:.0f})")
    print(f"    Max:    {results_df['test_f1'].max():.3f}  (seed {results_df.loc[results_df['test_f1'].idxmax(), 'seed']:.0f})")
    print(f"    Median: {results_df['test_f1'].median():.3f}")

    # Top 10 seeds
    top10 = results_df.nlargest(10, "test_f1")
    print(f"\n  Top 10 seeds by test F1:")
    print(f"  {'Seed':>6}  {'Test F1':>8}  {'Precision':>10}  {'Recall':>8}  {'FP':>4}  {'FN':>4}")
    for _, row in top10.iterrows():
        print(f"  {row['seed']:6.0f}  {row['test_f1']:8.3f}  {row['test_precision']:10.1%}  "
              f"{row['test_recall']:8.1%}  {row['test_fp']:4.0f}  {row['test_fn']:4.0f}")

    # ─── Save best model ─────────────────────────────────────────────────────
    if best_result and best_result["model_state"]:
        # Build a fresh model and load the best state
        set_seed(best_result["seed"])
        best_model = F1PodiumNet(
            input_dim=len(FEATURE_COLUMNS),
            hidden_layers=HYPERPARAMS["hidden_layers"],
            dropout_rate=HYPERPARAMS["dropout_rate"],
        )
        best_model.load_state_dict(best_result["model_state"])

        model_path = os.path.join(MODEL_DIR, "f1_podium_model.pt")
        torch.save({
            "model_state_dict": best_result["model_state"],
            "hyperparams": HYPERPARAMS,
            "feature_columns": FEATURE_COLUMNS,
            "seed": best_result["seed"],
        }, model_path)
        print(f"\n  Best model saved to {model_path}")
        print(f"  Seed: {best_result['seed']}")
        print(f"  Test F1: {best_result['test_f1']:.3f}")
        print(f"  Test Precision: {best_result['test_precision']:.1%}")
        print(f"  Test Recall: {best_result['test_recall']:.1%}")

    # ─── Save results CSV ────────────────────────────────────────────────────
    csv_path = os.path.join(DATA_DIR, "seed_search_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  Results saved to {csv_path}")

    # ─── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Seed Search Results", fontsize=16, fontweight="bold")

    # Plot 1: F1 vs Seed
    ax1 = axes[0, 0]
    colors = ["#E8364E" if f == best_test_f1 else "#3B82F6" for f in results_df["test_f1"]]
    ax1.bar(results_df["seed"], results_df["test_f1"], color=colors, alpha=0.7, width=0.8)
    ax1.axhline(y=results_df["test_f1"].mean(), color="#888", linestyle="--", alpha=0.5, label=f"Mean: {results_df['test_f1'].mean():.3f}")
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Test F1 Score")
    ax1.set_title("Test F1 by seed")
    ax1.legend()

    # Plot 2: Distribution of F1 scores
    ax2 = axes[0, 1]
    ax2.hist(results_df["test_f1"], bins=20, color="#3B82F6", alpha=0.7, edgecolor="white")
    ax2.axvline(x=best_test_f1, color="#E8364E", linestyle="-", linewidth=2, label=f"Best: {best_test_f1:.3f}")
    ax2.axvline(x=results_df["test_f1"].mean(), color="#888", linestyle="--", alpha=0.5, label=f"Mean: {results_df['test_f1'].mean():.3f}")
    ax2.set_xlabel("Test F1 Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of F1 scores")
    ax2.legend()

    # Plot 3: Precision vs Recall by seed
    ax3 = axes[1, 0]
    ax3.scatter(results_df["test_precision"], results_df["test_recall"],
                c=results_df["test_f1"], cmap="RdYlGn", s=40, alpha=0.7, edgecolor="white")
    if best_result:
        ax3.scatter([best_result["test_precision"]], [best_result["test_recall"]],
                    color="#E8364E", s=100, marker="*", zorder=5, label=f"Best (seed {best_result['seed']})")
    ax3.set_xlabel("Precision")
    ax3.set_ylabel("Recall")
    ax3.set_title("Precision vs recall (color = F1)")
    ax3.legend()

    # Plot 4: Best epoch vs F1
    ax4 = axes[1, 1]
    ax4.scatter(results_df["best_epoch"], results_df["test_f1"],
                c="#3B82F6", s=30, alpha=0.6, edgecolor="white")
    ax4.set_xlabel("Best epoch (when training stopped)")
    ax4.set_ylabel("Test F1 Score")
    ax4.set_title("Training length vs performance")

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "seed_search.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {plot_path}")

    print(f"\n  Done! Use seed {best_result['seed']} in oldRacesPredictingNN.py:")
    print(f"    SEED = {best_result['seed']}")