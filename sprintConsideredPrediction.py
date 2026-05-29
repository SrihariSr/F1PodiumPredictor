import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
 
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
 
SPRINT_RESULTS = {
    "RUS": 1,
    "NOR": 2,
    "ANT": 3,
    "PIA": 4,
    "LEC": 5,
    "HAM": 6,
    "VER": 7,
    "LIN": 8,
    "COL": 9,
    "SAI": 10,
    "LAW": 11,
    "BOR": 12,
    "OCO": 13,
    "PER": 14,
    "HUL": 15,
    "STR": 16,
    "BOT": 17,
    "BEA": 18,
    "ALB": 19,
    "GAS": 20,
    "HAD": 21,
    "ALO": 99, # DNF
}

TARGET_CIRCUIT = "Canada"

SPRINT_WEIGHT = 0.5

# Neural Network model
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
 
    def forward(self, x):
        return self.network(x)

# Convert sprint finishing position to a podium probability
def sprint_to_probability(position, total_drivers=22):
    if position >= 99:
        return 0.01
    return max(0.01, 0.95 * (1 - (position - 1) / (total_drivers - 1)))
 

# Load the trained PyTorch neural network
def load_nn_model(device):
    model_path = os.path.join(MODEL_DIR, "f1_podium_model.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
 
    model = F1PodiumNet(
        input_dim=len(FEATURE_COLUMNS),
        hidden_layers=checkpoint["hyperparams"]["hidden_layers"],
        dropout_rate=checkpoint["hyperparams"]["dropout_rate"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
 
    # Load scaler
    scaler_data = np.load(os.path.join(MODEL_DIR, "scaler_params.npz"))
    scaler = {"mean": scaler_data["mean"], "std": scaler_data["std"]}
 
    return model, scaler

# Load the trained XGBoost model
def load_xgb_model():
    model_path = os.path.join(MODEL_DIR, "f1_xgboost_model.json")
    model = xgb.Booster()
    model.load_model(model_path)
    return model
 
# Run neural network inference on race features
def get_nn_predictions(model, scaler, features, device):
    X = features[FEATURE_COLUMNS].values.astype(np.float32)
    X = (X - scaler["mean"].astype(np.float32)) / scaler["std"].astype(np.float32)
    X_t = torch.tensor(X).to(device)
 
    with torch.no_grad():
        probs = model(X_t).cpu().numpy().flatten()
 
    return probs

# Run XGBoost inference on race features
def get_xgb_predictions(model, features):
    X = features[FEATURE_COLUMNS].values
    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
    probs = model.predict(dmatrix)
    return probs


if __name__ == "__main__":
    print("Sprint adjusted Predictions")
    print("-" * 80)
 
    # Import race data and feature builder
    from futurePredictingNN import (
        RACES_2026, build_race_features, load_historical_stats,
        DEFAULT_DRIVER_STATS, DEFAULT_TEAM_STATS, DEFAULT_PRACTICE,
    )
 
    # Find the target race
    target_race = None
    for race in RACES_2026:
        if race["circuit"] == TARGET_CIRCUIT:
            target_race = race
            break
 
    if target_race is None:
        print(f"Race '{TARGET_CIRCUIT}' not found in RACES_2026")
        exit(1)
 
    print(f"Circuit: {TARGET_CIRCUIT}")
    print(f"Sprint weight: {SPRINT_WEIGHT:.0%} sprint, {1-SPRINT_WEIGHT:.0%} model")
 
    # Load historical stats
    print("Loading historical stats...")
    driver_stats, team_stats, circuit_stats = load_historical_stats()
 
    # Build race features
    features = build_race_features(
        qualifying_results=target_race["qualifying"],
        circuit_name=target_race["circuit"],
        weather=target_race["weather"],
        driver_stats=driver_stats,
        team_stats=team_stats,
        circuit_stats=circuit_stats,
    )
 
    drivers = features["Driver"].tolist()
 
    # Device for neural network
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
 
    # Neural network predictions
    print("  Loading neural network...")
    nn_model, scaler = load_nn_model(device)
    nn_probs = get_nn_predictions(nn_model, scaler, features, device)
 
    # XGBoost predictions
    print("  Loading XGBoost model...")
    xgb_model = load_xgb_model()
    xgb_probs = get_xgb_predictions(xgb_model, features)
 
    # Average both models for a combined prediction
    model_probs = (nn_probs + xgb_probs) / 2
 
    # Blend with sprint results
    results = []
    for i, driver in enumerate(drivers):
        sprint_pos = SPRINT_RESULTS.get(driver, 22)
        sprint_prob = sprint_to_probability(sprint_pos)
        grid_pos = features.iloc[i]["GridPosition"]
 
        blended = (1 - SPRINT_WEIGHT) * model_probs[i] + SPRINT_WEIGHT * sprint_prob
 
        results.append({
            "Driver": driver,
            "Grid": int(grid_pos),
            "NN_Prob": nn_probs[i],
            "XGB_Prob": xgb_probs[i],
            "Model_Avg": model_probs[i],
            "Sprint_Pos": sprint_pos,
            "Sprint_Prob": sprint_prob,
            "Blended": blended,
        })
 
    df = pd.DataFrame(results).sort_values("Blended", ascending=False)
 
    # Print Results
 
    print(f"\n  🏁 {TARGET_CIRCUIT} Grand Prix with Sprint-adjusted Predictions")
    print(f"{'─' * 80}")
    print(f"{'Rank':>4}  {'Driver':<6}  {'Grid':>4}  {'NN':>6}  {'XGB':>6}  "
          f"{'Avg':>6}  {'Sprint':>6}  {'Final':>7}")
    print(f"{'─' * 80}")
 
    for i, (_, row) in enumerate(df.iterrows()):
        medal = ""
        if i < 3:
            medal = ["🥇", "🥈", "🥉"][i]
 
        sprint_str = f"P{int(row['Sprint_Pos'])}" if row["Sprint_Pos"] < 99 else "DNS"
 
        print(f"{i+1:4d}  {row['Driver']:<6s}  P{row['Grid']:<3d}"
              f"{row['NN_Prob']:5.1%}  {row['XGB_Prob']:5.1%}  "
              f"{row['Model_Avg']:5.1%}  {sprint_str:>6s}  "
              f"{row['Blended']:6.1%}  {medal}")
 
    top3 = df.head(3)["Driver"].tolist()
    print(f"\n  Predicted podium: {', '.join(top3)}")
 
    #Compare all methods
 
    nn_top3 = df.sort_values("NN_Prob", ascending=False).head(3)["Driver"].tolist()
    xgb_top3 = df.sort_values("XGB_Prob", ascending=False).head(3)["Driver"].tolist()
    avg_top3 = df.sort_values("Model_Avg", ascending=False).head(3)["Driver"].tolist()
    blended_top3 = top3
 
    print(f"\n{'=' * 70}")
    print(f"Method Comparison")
    print(f"{'=' * 70}")
    print(f"Neural Network only: {', '.join(nn_top3)}")
    print(f"XGBoost only: {', '.join(xgb_top3)}")
    print(f"Model average: {', '.join(avg_top3)}")
    print(f"Sprint-adjusted ({SPRINT_WEIGHT:.0%}): {', '.join(blended_top3)}")
 
    # Check actual podium if available
    actual = target_race.get("actual_podium", [])
    if actual:
        print(f"\n  Actual podium: {', '.join(actual)}")
        for method, pred, name in [
            (nn_top3, actual, "Neural Network"),
            (xgb_top3, actual, "XGBoost"),
            (avg_top3, actual, "Model average"),
            (blended_top3, actual, "Sprint-adjusted"),
        ]:
            correct = len([d for d in method if d in actual])
            print(f"  {name:25s} {correct}/3")
 
 
    # Save results
    output_path = os.path.join(DATA_DIR, "sprint_adjusted_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"\n  Results saved to {output_path}")
