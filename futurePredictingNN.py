import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import fastf1

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

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

GRID_2026 = {
    "McLaren":        ["NOR", "PIA"],
    "Mercedes":       ["RUS", "ANT"],
    "Ferrari":        ["LEC", "HAM"],
    "Red Bull":       ["VER", "HAD"],
    "Aston Martin":   ["ALO", "STR"],
    "Alpine":         ["GAS", "DOO"],
    "Williams":       ["ALB", "SAI"],
    "Racing Bulls":   ["LAW", "LIN"],
    "Haas":           ["OCO", "BEA"],
    "Audi":           ["HUL", "BOR"],
    "Cadillac":       ["PER", "BOT"],
}
 
STREET_CIRCUITS = [
    "Monaco", "Singapore", "Azerbaijan", "Saudi Arabia",
    "Las Vegas", "Miami", "Australia",
]
 
NUM_DRIVERS = 22

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

def load_model(device):
    
    checkpoint_path = os.path.join(MODEL_DIR, "f1_podium_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
 
    hyperparams = checkpoint["hyperparams"]
 
    model = F1PodiumNet(
        input_dim=len(FEATURE_COLUMNS),
        hidden_layers=hyperparams["hidden_layers"],
        dropout_rate=hyperparams["dropout_rate"],
    ).to(device)
 
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # permanently in evaluation mode (no dropout)
 
    # Load scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler_params.npz")
    scaler_data = np.load(scaler_path)
    scaler = {"mean": scaler_data["mean"], "std": scaler_data["std"]}
 
    print(f"  Model loaded from {checkpoint_path}")
    print(f"  Scaler loaded from {scaler_path}")
 
    return model, scaler


def fetch_practice_data(year: int, round_num: int) -> dict:
    """
    Fetch FP2 pace data from FastF1 for a specific race weekend.
    Returns a dict mapping driver abbreviation to their practice stats.

    """
    print(f"Fetching FP2 data for {year} Round {round_num}...")
 
    try:
        session = fastf1.get_session(year, round_num, "FP2")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps
 
        # Collect raw stats per driver
        driver_data = {}
 
        for driver in laps["Driver"].unique():
            driver_laps = laps.pick_drivers(driver).pick_accurate()
            if driver_laps.empty:
                continue
 
            # Convert lap times to seconds
            lap_secs = driver_laps["LapTime"].apply(
                lambda td: td.total_seconds() if pd.notna(td) and hasattr(td, "total_seconds") else np.nan
            ).dropna()
 
            if len(lap_secs) == 0:
                continue
 
            best_lap = lap_secs.min()
 
            # Long run pace: find stints of 5+ laps and average them
            long_run_times = []
            if "Stint" in driver_laps.columns:
                for stint_id, stint in driver_laps.groupby("Stint"):
                    stint_secs = stint["LapTime"].apply(
                        lambda td: td.total_seconds() if pd.notna(td) and hasattr(td, "total_seconds") else np.nan
                    ).dropna()
                    if len(stint_secs) >= 5:
                        long_run_times.extend(stint_secs.tolist())
 
            long_run_pace = np.mean(long_run_times) if long_run_times else np.nan
 
            driver_data[driver] = {
                "fp2_best_lap": best_lap,
                "fp2_long_run": long_run_pace,
            }
 
        if not driver_data:
            print(f"No FP2 data available")
            return {}
 
        # Compute deltas relative to the fastest
        fastest_best = min(d["fp2_best_lap"] for d in driver_data.values())
 
        long_runs = [d["fp2_long_run"] for d in driver_data.values() if pd.notna(d["fp2_long_run"])]
        fastest_long = min(long_runs) if long_runs else np.nan
 
        practice_stats = {}
        for driver, data in driver_data.items():
            practice_stats[driver] = {
                "fp2_best_delta": data["fp2_best_lap"] - fastest_best,
                "fp2_longrun_delta": (
                    data["fp2_long_run"] - fastest_long
                    if pd.notna(data["fp2_long_run"]) and pd.notna(fastest_long)
                    else 2.0  # default: assume 2 seconds off pace
                ),
            }
 
        fastest_driver = min(driver_data, key=lambda d: driver_data[d]["fp2_best_lap"])
        print(f"FP2 data loaded for {len(practice_stats)} drivers")
        print(f"Fastest: {fastest_driver} ({fastest_best:.3f}s)")
 
        return practice_stats
 
    except Exception as e:
        print(f"FP2 data unavailable: {e}")
        return {}

def load_historical_stats():
    # Loads driver and team rolling stats going into 2026
    train_path = os.path.join(DATA_DIR, "f1_features_train.csv")
    val_path = os.path.join(DATA_DIR, "f1_features_val.csv")
    test_path = os.path.join(DATA_DIR, "f1_features_test.csv")

    dataFrames = []

    for path in [train_path, val_path, test_path]:
        if os.path.exists(path):
            dataFrames.append(pd.read_csv(path))
    
    if not dataFrames:
        print("No historical data found.")
        return {}, {}, {}
    
    history = pd.concat(dataFrames, ignore_index=True)
    history = history.sort_values(["Year", "Round"]) # Storing only the most recent year

    driver_stats = {}

    for driver, group in history.groupby("Driver"):
        latest = group.iloc[-1]
        driver_stats[driver] = {
            "rolling_avg_finish_5": latest.get("rolling_avg_finish_5", 10.0),
            "rolling_avg_grid_5": latest.get("rolling_avg_grid_5", 10.0),
            "rolling_podium_rate_5": latest.get("rolling_podium_rate_5", 0.0),
            "rolling_dnf_rate_5": latest.get("rolling_dnf_rate_5", 0.1),
            "rolling_avg_points_5": latest.get("rolling_avg_points_5", 0.0),
            "rolling_avg_finish_3": latest.get("rolling_avg_finish_3", 10.0),
            "rolling_podium_rate_3": latest.get("rolling_podium_rate_3", 0.0),
            "rolling_avg_finish_10": latest.get("rolling_avg_finish_10", 10.0),
            "rolling_podium_rate_10": latest.get("rolling_podium_rate_10", 0.0),
            "driver_encoded": latest.get("driver_encoded", 0.15),
            "streak_podiums": latest.get("streak_podiums", 0),
            "streak_points": latest.get("streak_points", 0),
            "best_finish_last_5": latest.get("best_finish_last_5", 12.0),
            "positions_gained_avg_5": latest.get("positions_gained_avg_5", 0.0),
            "first_lap_avg_gain_5": latest.get("first_lap_avg_gain_5", 0.0),
        }

    team_stats = {}
    for team, group in history.groupby("Team"):
        team_stats[team] = {
            "team_encoded": group["team_encoded"].iloc[-1]
            if "team_encoded" in group.columns else 0.15, # Using an estimate of 15% podium rate as a fallback value
        }
    circuit_stats = {}

    # Load upset rates for the circuits
    for circuit, group in history.groupby("EventName"):
        circuit_stats[circuit] = {
            "circuit_encoded": group["circuit_encoded"].iloc[-1]
            if "circuit_encoded" in group.columns else 0.05,
            "circuit_sc_rate": group["circuit_sc_rate"].iloc[-1]
            if "circuit_sc_rate" in group.columns else 0.3,
        }

    return driver_stats, team_stats, circuit_stats

DEFAULT_PRACTICE = {
    "fp2_best_delta": 2.0,
    "fp2_longrun_delta": 2.0,
}

# For new drivers/teams with no historical data stored
DEFAULT_DRIVER_STATS = {
    "rolling_avg_finish_5": 13.0,
    "rolling_avg_grid_5": 13.0,
    "rolling_podium_rate_5": 0.0,
    "rolling_dnf_rate_5": 0.1,
    "rolling_avg_points_5": 2.0,
    "rolling_avg_finish_3": 13.0,
    "rolling_podium_rate_3": 0.0,
    "rolling_avg_finish_10": 13.0,
    "rolling_podium_rate_10": 0.0,
    "driver_encoded": 0.05,
}

DEFAULT_TEAM_STATS = {
    "team_encoded": 0.05,
}

def build_race_features(
    qualifying_results: list[dict],
    circuit_name: str,
    weather: dict,
    driver_stats: dict,
    team_stats: dict,
    circuit_stats: dict,
    practice_data=None,
) -> pd.DataFrame:

    rows = []

    # Calculate qualifying times, pole time, and median time
    qualifying_times = [r["q_time_sec"] for r in qualifying_results if r.get("q_time_sec")]
    pole_time = min(qualifying_times) if qualifying_times else None
    median_time = np.median(qualifying_times) if qualifying_times else None

    # Check if the circuit is a street circuit
    is_street = 1 if any(
        sc.lower() in circuit_name.lower() for sc in STREET_CIRCUITS
    ) else 0

    c_stats = circuit_stats.get(circuit_name, {"circuit_encoded": 0.05, "circuit_sc_rate": 0.3})

    # Normalizing the weather data
    air_norm = np.clip((weather.get("air_temp", 25) - 10) / 35, 0, 1)
    track_norm = np.clip((weather.get("track_temp", 35) - 15) / 50, 0, 1)
    humid_norm = np.clip((weather.get("humidity", 50) - 10) / 90, 0, 1)
    wind_norm = np.clip((weather.get("wind_speed", 3) - 0) / 15, 0, 1)
    rainfall = weather.get("rainfall", 0)

    # Compute teammate grid deltas from qualifying results
    teammate_deltas = {}
    team_drivers = {}
    for result in qualifying_results:
        team = result["team"]
        if team not in team_drivers:
            team_drivers[team] = []
        team_drivers[team].append((result["driver"], result["grid_position"]))

    for team, drivers in team_drivers.items():
        if len(drivers) == 2:
            d1, g1 = drivers[0]
            d2, g2 = drivers[1]
            teammate_deltas[d1] = g1 - g2
            teammate_deltas[d2] = g2 - g1

    # Practice data defaults
    if practice_data is None:
        practice_data = {}

    for result in qualifying_results:
        driver = result["driver"]
        team = result["team"]
        grid_pos = result["grid_position"]
        q_time = result.get("q_time_sec")

        # Look up this driver's historical form
        d_stats = driver_stats.get(driver, DEFAULT_DRIVER_STATS)
        t_stats = team_stats.get(team, DEFAULT_TEAM_STATS)

        if q_time and pole_time:
            quali_delta_pole = q_time - pole_time
        else:
            quali_delta_pole = 5.0

        if q_time and median_time:
            quali_delta_median = q_time - median_time
        else:
            quali_delta_median = 3.0

        # Practice stats for this driver
        if not practice_data:
            fp2_best = quali_delta_pole * 1.2 if quali_delta_pole else 2.0
            fp2_long = quali_delta_pole * 1.5 if quali_delta_pole else 2.0
            p_stats = {"fp2_best_delta": fp2_best, "fp2_longrun_delta": fp2_long}
        else:
            p_stats = practice_data.get(driver, DEFAULT_PRACTICE)

        row = {
            "Driver": driver,
            "Team": team,

            # Practice
            "FP2_BestLap_Delta": p_stats.get("fp2_best_delta", 2.0),
            "FP2_LongRun_Delta": p_stats.get("fp2_longrun_delta", 2.0),

            # Grid & qualifying
            "GridPosition": grid_pos,
            "GridPosition_norm": grid_pos / NUM_DRIVERS,
            "quali_delta_to_pole": quali_delta_pole,
            "quali_delta_to_median": quali_delta_median,

            # Rolling form (primary window)
            "rolling_avg_finish_5": d_stats["rolling_avg_finish_5"],
            "rolling_avg_grid_5": d_stats["rolling_avg_grid_5"],
            "rolling_podium_rate_5": d_stats["rolling_podium_rate_5"],
            "rolling_dnf_rate_5": d_stats["rolling_dnf_rate_5"],
            "rolling_avg_points_5": d_stats["rolling_avg_points_5"],

            # Rolling form (short window)
            "rolling_avg_finish_3": d_stats["rolling_avg_finish_3"],
            "rolling_podium_rate_3": d_stats["rolling_podium_rate_3"],

            # Rolling form (long window)
            "rolling_avg_finish_10": d_stats["rolling_avg_finish_10"],
            "rolling_podium_rate_10": d_stats["rolling_podium_rate_10"],

            # Encodings
            "driver_encoded": d_stats["driver_encoded"],
            "team_encoded": t_stats["team_encoded"],

            # Circuit
            "is_street_circuit": is_street,
            "circuit_encoded": c_stats.get("circuit_encoded", 0.05),

            # Weather
            "air_temp_norm": air_norm,
            "track_temp_norm": track_norm,
            "humidity_norm": humid_norm,
            "wind_speed_norm": wind_norm,
            "Rainfall": rainfall,

            # Interactions
            "grid_x_rainfall": grid_pos * rainfall,
            "driver_team_form": d_stats["driver_encoded"] * t_stats["team_encoded"],
            "form_vs_grid_gap": (
                d_stats["rolling_avg_grid_5"] - d_stats["rolling_avg_finish_5"]
            ),

            # Driver momentum
            "streak_podiums": d_stats.get("streak_podiums", 0),
            "streak_points": d_stats.get("streak_points", 0),
            "best_finish_last_5": d_stats.get("best_finish_last_5", 12.0),
            "positions_gained_avg_5": d_stats.get("positions_gained_avg_5", 0.0),

            # Teammate comparison
            "teammate_grid_delta": teammate_deltas.get(driver, 0.0),

            # Circuit-specific
            "driver_circuit_avg_finish": d_stats.get("driver_circuit_avg_finish", 10.0),
            "circuit_sc_rate": c_stats.get("circuit_sc_rate", 0.3),

            # Race start
            "first_lap_avg_gain_5": d_stats.get("first_lap_avg_gain_5", 0.0),
        }

        rows.append(row)

    return pd.DataFrame(rows)

def predict_race_results(model, scaler, race_features: pd.DataFrame, device, circuit_name: str = "") -> pd.DataFrame:
    
    x = race_features[FEATURE_COLUMNS].values.astype(np.float32)

    x = (x - scaler["mean"])/scaler["std"] # Standardizing the features

    x_tensor = torch.tensor(x).to(device)

    model.eval()
    with torch.no_grad():
        probabilities = model(x_tensor).cpu().numpy().flatten()
    
    results = race_features[["Driver", "Team", "GridPosition"]].copy()
    results["PodiumProb"] = probabilities
    results["PodiumProb%"] = (probabilities * 100).round(1)
    results["Predicted"] = (probabilities >= 0.5).astype(int)
    
    results = results.sort_values("PodiumProb", ascending=False).reset_index(drop=True)
    results["Rank"] = range(1, len(results) + 1)

    return results

def print_predictions(results: pd.DataFrame, circuit_name: str):

    print(f"\n{circuit_name} Grand Prix Podium Predictions")
    print(f"  {'─' * 65}")
    print(f"  {'Rank':>4}  {'Driver':<6}  {'Team':<16}  {'Grid':>4}  "
          f"{'Prob':>7}  {'Podium?':>7}")
    print(f"  {'─' * 65}")
 
    for _, row in results.iterrows():
        rank = row["Rank"]
        driver = row["Driver"]
        team = row["Team"]
        grid = int(row["GridPosition"])
        prob = row["PodiumProb%"]
 
        # Probability bar
        bar_len = int(prob / 100 * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
 
        print(f"{rank:4d}  {driver:<6}  {team:<16}  P{grid:<3d}  "
              f"{prob:6.1f}%  {bar}")
 
    # Summary
    predicted_podiums = results[results["Predicted"] == 1]
    print(f"\n  Predicted podium finishers: {len(predicted_podiums)}")
    if len(predicted_podiums) > 0:
        top3 = results.head(3)
        names = ", ".join(top3["Driver"].tolist())
        print(f"  Most likely podium: {names}")
 

RACES_2026 = [
    {
        "circuit": "Australia",
        "round": 1,
        "weather": {"air_temp": 22, "track_temp": 32, "humidity": 55, "wind_speed": 3, "rainfall": 0},
        "qualifying": [
            {"driver": "RUS", "team": "Mercedes",      "grid_position": 1,  "q_time_sec": 78.518},
            {"driver": "ANT", "team": "Mercedes",      "grid_position": 2,  "q_time_sec": 78.756},
            {"driver": "HAD", "team": "Red Bull",       "grid_position": 3,  "q_time_sec": 79.303},
            {"driver": "LEC", "team": "Ferrari",        "grid_position": 4,  "q_time_sec": 79.350},
            {"driver": "NOR", "team": "McLaren",        "grid_position": 5,  "q_time_sec": 79.502},
            {"driver": "PIA", "team": "McLaren",        "grid_position": 6,  "q_time_sec": 79.610},
            {"driver": "HAM", "team": "Ferrari",        "grid_position": 7,  "q_time_sec": 79.650},
            {"driver": "BEA", "team": "Haas",           "grid_position": 8,  "q_time_sec": 79.850},
            {"driver": "LIN", "team": "Racing Bulls",   "grid_position": 9,  "q_time_sec": 79.900},
            {"driver": "GAS", "team": "Alpine",         "grid_position": 10, "q_time_sec": 80.050},
            {"driver": "LAW", "team": "Racing Bulls",   "grid_position": 11, "q_time_sec": 80.100},
            {"driver": "HUL", "team": "Audi",           "grid_position": 12, "q_time_sec": 80.200},
            {"driver": "DOO", "team": "Alpine",         "grid_position": 13, "q_time_sec": 80.300},
            {"driver": "BOR", "team": "Audi",           "grid_position": 14, "q_time_sec": 80.400},
            {"driver": "ALB", "team": "Williams",       "grid_position": 15, "q_time_sec": 80.500},
            {"driver": "OCO", "team": "Haas",           "grid_position": 16, "q_time_sec": 80.600},
            {"driver": "PER", "team": "Cadillac",       "grid_position": 17, "q_time_sec": 80.700},
            {"driver": "BOT", "team": "Cadillac",       "grid_position": 18, "q_time_sec": 80.800},
            {"driver": "SAI", "team": "Williams",       "grid_position": 19, "q_time_sec": 80.900},
            # VER, STR, ALO did not set qualifying times — started from back
            {"driver": "VER", "team": "Red Bull",       "grid_position": 20, "q_time_sec": None},
            {"driver": "STR", "team": "Aston Martin",   "grid_position": 21, "q_time_sec": None},
            {"driver": "ALO", "team": "Aston Martin",   "grid_position": 22, "q_time_sec": None},
        ],
        # Actual result: RUS P1, ANT P2, LEC P3 | HAM P4, NOR P5, VER P6
        "actual_podium": ["RUS", "ANT", "LEC"],
    },
    {
        "circuit": "China",
        "round": 2,
        "weather": {"air_temp": 16, "track_temp": 24, "humidity": 65, "wind_speed": 4, "rainfall": 0},
        "qualifying": [
            {"driver": "ANT", "team": "Mercedes",      "grid_position": 1,  "q_time_sec": 93.200},
            {"driver": "RUS", "team": "Mercedes",      "grid_position": 2,  "q_time_sec": 93.450},
            {"driver": "LEC", "team": "Ferrari",        "grid_position": 3,  "q_time_sec": 93.700},
            {"driver": "HAM", "team": "Ferrari",        "grid_position": 4,  "q_time_sec": 93.750},
            {"driver": "BEA", "team": "Haas",           "grid_position": 5,  "q_time_sec": 94.100},
            {"driver": "GAS", "team": "Alpine",         "grid_position": 6,  "q_time_sec": 94.200},
            {"driver": "HAD", "team": "Red Bull",       "grid_position": 7,  "q_time_sec": 94.300},
            {"driver": "VER", "team": "Red Bull",       "grid_position": 8,  "q_time_sec": 94.400},
            {"driver": "LIN", "team": "Racing Bulls",   "grid_position": 9,  "q_time_sec": 94.500},
            {"driver": "LAW", "team": "Racing Bulls",   "grid_position": 10, "q_time_sec": 94.600},
            {"driver": "HUL", "team": "Audi",           "grid_position": 11, "q_time_sec": 94.700},
            {"driver": "DOO", "team": "Alpine",         "grid_position": 12, "q_time_sec": 94.800},
            {"driver": "OCO", "team": "Haas",           "grid_position": 13, "q_time_sec": 94.900},
            {"driver": "PER", "team": "Cadillac",       "grid_position": 14, "q_time_sec": 95.000},
            {"driver": "ALB", "team": "Williams",       "grid_position": 15, "q_time_sec": 95.100},
            {"driver": "BOT", "team": "Cadillac",       "grid_position": 16, "q_time_sec": 95.200},
            {"driver": "SAI", "team": "Williams",       "grid_position": 17, "q_time_sec": 95.300},
            {"driver": "BOR", "team": "Audi",           "grid_position": 18, "q_time_sec": 95.400},
            {"driver": "ALO", "team": "Aston Martin",   "grid_position": 19, "q_time_sec": 95.500},
            {"driver": "STR", "team": "Aston Martin",   "grid_position": 20, "q_time_sec": 95.600},
            # NOR and PIA did not start
            {"driver": "NOR", "team": "McLaren",        "grid_position": 21, "q_time_sec": None},
            {"driver": "PIA", "team": "McLaren",        "grid_position": 22, "q_time_sec": None},
        ],
        "actual_podium": ["ANT", "RUS", "HAM"],
    },
    {
    "circuit": "Japan",
    "round": 3,
    "weather": {"air_temp": 20, "track_temp": 30, "humidity": 55, "wind_speed": 3, "rainfall": 0},
    "qualifying": [
        {"driver": "ANT", "team": "Mercedes",      "grid_position": 1,  "q_time_sec": 88.778},
        {"driver": "RUS", "team": "Mercedes",      "grid_position": 2,  "q_time_sec": 89.076},
        {"driver": "PIA", "team": "McLaren",        "grid_position": 3,  "q_time_sec": 89.132},
        {"driver": "LEC", "team": "Ferrari",        "grid_position": 4,  "q_time_sec": 89.200},
        {"driver": "NOR", "team": "McLaren",        "grid_position": 5,  "q_time_sec": 89.250},
        {"driver": "HAM", "team": "Ferrari",        "grid_position": 6,  "q_time_sec": 89.350},
        {"driver": "GAS", "team": "Alpine",         "grid_position": 7,  "q_time_sec": 89.500},
        {"driver": "HAD", "team": "Red Bull",       "grid_position": 8,  "q_time_sec": 89.600},
        {"driver": "BOR", "team": "Audi",           "grid_position": 9,  "q_time_sec": 89.700},
        {"driver": "LIN", "team": "Racing Bulls",   "grid_position": 10, "q_time_sec": 89.800},
        {"driver": "VER", "team": "Red Bull",       "grid_position": 11, "q_time_sec": 89.900},
        {"driver": "LAW", "team": "Racing Bulls",   "grid_position": 12, "q_time_sec": 90.000},
        {"driver": "DOO", "team": "Alpine",         "grid_position": 13, "q_time_sec": 90.100},
        {"driver": "OCO", "team": "Haas",           "grid_position": 14, "q_time_sec": 90.200},
        {"driver": "HUL", "team": "Audi",           "grid_position": 15, "q_time_sec": 90.300},
        {"driver": "SAI", "team": "Williams",       "grid_position": 16, "q_time_sec": 90.400},
        {"driver": "ALB", "team": "Williams",       "grid_position": 17, "q_time_sec": 90.500},
        {"driver": "BEA", "team": "Haas",           "grid_position": 18, "q_time_sec": 90.600},
        {"driver": "PER", "team": "Cadillac",       "grid_position": 19, "q_time_sec": 90.700},
        {"driver": "BOT", "team": "Cadillac",       "grid_position": 20, "q_time_sec": 90.800},
        {"driver": "ALO", "team": "Aston Martin",   "grid_position": 21, "q_time_sec": 90.900},
        {"driver": "STR", "team": "Aston Martin",   "grid_position": 22, "q_time_sec": 91.000},
    ],
    "actual_podium": ["ANT", "PIA", "LEC"],
},
{
    "circuit": "Miami",
    "round": 4,
    "weather": {
        "air_temp": 26,
        "track_temp": 35,
        "humidity": 79,  
        "wind_speed": 1.3,
        "rainfall": 1,
    },
    "qualifying": [
        {"driver": "ANT", "team": "Mercedes",      "grid_position": 1,  "q_time_sec": 87.798},
        {"driver": "VER", "team": "Red Bull",       "grid_position": 2,  "q_time_sec": 87.964},
        {"driver": "LEC", "team": "Ferrari",        "grid_position": 3,  "q_time_sec": 88.143},
        {"driver": "NOR", "team": "McLaren",        "grid_position": 4,  "q_time_sec": 88.183},
        {"driver": "RUS", "team": "Mercedes",       "grid_position": 5,  "q_time_sec": 88.197},
        {"driver": "HAM", "team": "Ferrari",        "grid_position": 6,  "q_time_sec": 88.319},
        {"driver": "PIA", "team": "McLaren",        "grid_position": 7,  "q_time_sec": 88.500},
        {"driver": "COL", "team": "Alpine",         "grid_position": 8,  "q_time_sec": 88.762},
        {"driver": "HAD", "team": "Racing Bulls",   "grid_position": 9,  "q_time_sec": 88.789},
        {"driver": "GAS", "team": "Alpine",         "grid_position": 10, "q_time_sec": 88.810},
        {"driver": "HUL", "team": "Audi",           "grid_position": 11, "q_time_sec": 89.121},
        {"driver": "LAW", "team": "Racing Bulls",   "grid_position": 12, "q_time_sec": 89.181},
        {"driver": "BEA", "team": "Haas",           "grid_position": 13, "q_time_sec": 89.249},
        {"driver": "SAI", "team": "Williams",       "grid_position": 14, "q_time_sec": 89.250},
        {"driver": "OCO", "team": "Haas",           "grid_position": 15, "q_time_sec": 89.454},
        {"driver": "ALB", "team": "Williams",       "grid_position": 16, "q_time_sec": 89.628},
        {"driver": "LIN", "team": "Racing Bulls",   "grid_position": 17, "q_time_sec": 89.278},
        {"driver": "ALO", "team": "Aston Martin",   "grid_position": 18, "q_time_sec": 90.243},
        {"driver": "STR", "team": "Aston Martin",   "grid_position": 19, "q_time_sec": 90.309},
        {"driver": "BOT", "team": "Cadillac",       "grid_position": 20, "q_time_sec": 90.774},
        {"driver": "PER", "team": "Cadillac",       "grid_position": 21, "q_time_sec": 91.112},
        {"driver": "BOR", "team": "Audi",           "grid_position": 22, "q_time_sec": 92.882},
    ],
    "actual_podium": ["ANT", "NOR", "PIA"],
},
{
    "circuit": "Canada",
    "round": 5,
    "weather": {
        "air_temp": 12,
        "track_temp": 18,
        "humidity": 85,
        "wind_speed": 3,
        "rainfall": 1,
    },
    "qualifying": [
        {"driver": "RUS", "team": "Mercedes",      "grid_position": 1,  "q_time_sec": 72.578},
        {"driver": "ANT", "team": "Mercedes",      "grid_position": 2,  "q_time_sec": 72.646},
        {"driver": "NOR", "team": "McLaren",        "grid_position": 3,  "q_time_sec": 72.729},
        {"driver": "PIA", "team": "McLaren",        "grid_position": 4,  "q_time_sec": 72.781},
        {"driver": "HAM", "team": "Ferrari",        "grid_position": 5,  "q_time_sec": 72.868},
        {"driver": "VER", "team": "Red Bull",       "grid_position": 6,  "q_time_sec": 72.907},
        {"driver": "HAD", "team": "Red Bull",       "grid_position": 7,  "q_time_sec": 72.935},
        {"driver": "LEC", "team": "Ferrari",        "grid_position": 8,  "q_time_sec": 72.976},
        {"driver": "LIN", "team": "Racing Bulls",   "grid_position": 9,  "q_time_sec": 73.280},
        {"driver": "COL", "team": "Alpine",         "grid_position": 10, "q_time_sec": 73.688},
        {"driver": "HUL", "team": "Audi",           "grid_position": 11, "q_time_sec": 73.717},
        {"driver": "LAW", "team": "Racing Bulls",   "grid_position": 12, "q_time_sec": 73.728},
        {"driver": "BOR", "team": "Audi",           "grid_position": 13, "q_time_sec": 73.902},
        {"driver": "GAS", "team": "Alpine",         "grid_position": 14, "q_time_sec": 74.018},
        {"driver": "SAI", "team": "Williams",       "grid_position": 15, "q_time_sec": 74.104},
        {"driver": "BEA", "team": "Haas",           "grid_position": 16, "q_time_sec": 74.247},
        {"driver": "OCO", "team": "Haas",           "grid_position": 17, "q_time_sec": 74.317},
        {"driver": "ALB", "team": "Williams",       "grid_position": 18, "q_time_sec": 74.323},
        {"driver": "ALO", "team": "Aston Martin",   "grid_position": 19, "q_time_sec": 74.668},
        {"driver": "PER", "team": "Cadillac",       "grid_position": 20, "q_time_sec": 74.901},
        {"driver": "STR", "team": "Aston Martin",   "grid_position": 21, "q_time_sec": 75.667},
        {"driver": "BOT", "team": "Cadillac",       "grid_position": 22, "q_time_sec": 75.744},
    ],
    "actual_podium": ["ANT", "HAM", "VER"],
},
{
    "circuit": "Monaco",
    "round": 6,
    "weather": {
        "air_temp": 21,
        "track_temp": 40,
        "humidity": 55,
        "wind_speed": 2,
        "rainfall": 0,
    },
    "qualifying": [
        {"driver": "ANT", "team": "Mercedes",      "grid_position": 1,  "q_time_sec": 72.051},
        {"driver": "VER", "team": "Red Bull",       "grid_position": 2,  "q_time_sec": 72.094},
        {"driver": "HAM", "team": "Ferrari",        "grid_position": 3,  "q_time_sec": 72.264},
        {"driver": "LEC", "team": "Ferrari",        "grid_position": 4,  "q_time_sec": 72.351},
        {"driver": "HAD", "team": "Red Bull",       "grid_position": 5,  "q_time_sec": 72.401},
        {"driver": "RUS", "team": "Mercedes",       "grid_position": 6,  "q_time_sec": 72.501},
        {"driver": "PIA", "team": "McLaren",        "grid_position": 7,  "q_time_sec": 72.601},
        {"driver": "NOR", "team": "McLaren",        "grid_position": 8,  "q_time_sec": 72.651},
        {"driver": "GAS", "team": "Alpine",         "grid_position": 9,  "q_time_sec": 72.901},
        {"driver": "LAW", "team": "Racing Bulls",   "grid_position": 10, "q_time_sec": 73.001},
        {"driver": "ALB", "team": "Williams",       "grid_position": 11, "q_time_sec": 73.101},
        {"driver": "SAI", "team": "Williams",       "grid_position": 12, "q_time_sec": 73.151},
        {"driver": "HUL", "team": "Audi",           "grid_position": 13, "q_time_sec": 73.201},
        {"driver": "COL", "team": "Alpine",         "grid_position": 14, "q_time_sec": 73.301},
        {"driver": "LIN", "team": "Racing Bulls",   "grid_position": 15, "q_time_sec": 73.401},
        {"driver": "BOR", "team": "Audi",           "grid_position": 16, "q_time_sec": 73.551},
        {"driver": "OCO", "team": "Haas",           "grid_position": 17, "q_time_sec": 73.601},
        {"driver": "PER", "team": "Cadillac",       "grid_position": 18, "q_time_sec": 73.751},
        {"driver": "BEA", "team": "Haas",           "grid_position": 19, "q_time_sec": 73.901},
        {"driver": "BOT", "team": "Cadillac",       "grid_position": 20, "q_time_sec": 74.051},
        {"driver": "ALO", "team": "Aston Martin",   "grid_position": 21, "q_time_sec": 74.151},
        {"driver": "STR", "team": "Aston Martin",   "grid_position": 22, "q_time_sec": 74.251},
    ],
    "actual_podium": ["ANT", "HAM", "GAS"],
},
{
    "circuit": "Spain",
    "round": 7,
    "weather": {
        "air_temp": 32,
        "track_temp": 56,
        "humidity": 30,
        "wind_speed": 4,
        "rainfall": 0,
    },
    "qualifying": [
        {"driver": "RUS", "team": "Mercedes",      "grid_position": 1,  "q_time_sec": 74.679},
        {"driver": "HAM", "team": "Ferrari",        "grid_position": 2,  "q_time_sec": 74.743},
        {"driver": "ANT", "team": "Mercedes",       "grid_position": 3,  "q_time_sec": 74.998},
        {"driver": "NOR", "team": "McLaren",        "grid_position": 4,  "q_time_sec": 75.001},
        {"driver": "VER", "team": "Red Bull",       "grid_position": 5,  "q_time_sec": 75.021},
        {"driver": "HAD", "team": "Red Bull",       "grid_position": 6,  "q_time_sec": 75.077},
        {"driver": "PIA", "team": "McLaren",        "grid_position": 7,  "q_time_sec": 75.090},
        {"driver": "LAW", "team": "Racing Bulls",   "grid_position": 8,  "q_time_sec": 76.542},
        {"driver": "HUL", "team": "Audi",           "grid_position": 9,  "q_time_sec": 76.657},
        {"driver": "LEC", "team": "Ferrari",        "grid_position": 10, "q_time_sec": 75.400},
        {"driver": "LIN", "team": "Racing Bulls",   "grid_position": 11, "q_time_sec": 75.840},
        {"driver": "BOR", "team": "Audi",           "grid_position": 12, "q_time_sec": 76.001},
        {"driver": "COL", "team": "Alpine",         "grid_position": 13, "q_time_sec": 76.191},
        {"driver": "GAS", "team": "Alpine",         "grid_position": 14, "q_time_sec": 76.261},
        {"driver": "BEA", "team": "Haas",           "grid_position": 15, "q_time_sec": 76.389},
        {"driver": "SAI", "team": "Williams",       "grid_position": 16, "q_time_sec": 77.827},
        {"driver": "OCO", "team": "Haas",           "grid_position": 17, "q_time_sec": 77.073},
        {"driver": "ALB", "team": "Williams",       "grid_position": 18, "q_time_sec": 77.424},
        {"driver": "PER", "team": "Cadillac",       "grid_position": 19, "q_time_sec": 77.545},
        {"driver": "BOT", "team": "Cadillac",       "grid_position": 20, "q_time_sec": 77.757},
        {"driver": "STR", "team": "Aston Martin",   "grid_position": 21, "q_time_sec": 78.758},
        {"driver": "ALO", "team": "Aston Martin",   "grid_position": 22, "q_time_sec": 78.815},
    ],
    "actual_podium": ["HAM", "RUS", "NOR"],
},
]

if __name__ == "__main__":
    print("2026 Race Predictions")
    print("=" * 60)
 
    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(f"  Using CPU")
 
    # Load model and scaler
    print("\n  Loading model...")
    model, scaler = load_model(device)
 
    # Load historical stats
    print("\n  Loading historical stats...")
    driver_stats, team_stats, circuit_stats = load_historical_stats()
 
    # Predict each 2026 race using actual qualifying data
    all_predictions = []
 
    for race in RACES_2026:
        circuit = race["circuit"]
        weather = race["weather"]
        qualifying = race["qualifying"]
        actual_podium = race.get("actual_podium", [])
        round_num = race["round"]
 
        # Fetch FP2 practice data for this race
        # practice_data = fetch_practice_data(2026, round_num)
        # Will use once enough races are done

        # Build features (now includes practice data)
        features = build_race_features(
            qualifying_results=qualifying,
            circuit_name=circuit,
            weather=weather,
            driver_stats=driver_stats,
            team_stats=team_stats,
            circuit_stats=circuit_stats,
            # practice_data=practice_data,
        )
 
        # Predict
        results = predict_race_results(model, scaler, features, device, circuit)
        print_predictions(results, circuit)

        if actual_podium:
            predicted_podium = results.head(3)["Driver"].tolist()
            correct_predictions = [i for i in predicted_podium if i in actual_podium]
            missed = [i for i in actual_podium if i not in predicted_podium]
        
            print(f"\nActual podium: {', '.join(actual_podium)}")
            print(f"Predicted top 3: {', '.join(predicted_podium)}")
            print(f"Correct:         {len(correct_predictions)}/3 ({', '.join(correct_predictions) if correct_predictions else 'none'})")
            if missed:
                print(f"Missed: {', '.join(missed)}")
 
        # Store for CSV
        results["Circuit"] = circuit
        results["Round"] = race["round"]
        all_predictions.append(results)
        
    # Save all predictions
    all_preds = pd.concat(all_predictions, ignore_index=True)
    output_path = os.path.join(DATA_DIR, "predictions_2026.csv")
    all_preds.to_csv(output_path, index=False)
    print(f"\nAll predictions saved to {output_path}")

        # Season accuracy so far
    print(f"\n{'-' * 60}")
    print(f"2026 Season Prediction Accuracy (Neural Network)")
    print(f"{'-' * 60}")
 
    total_podium_spots = 0
    total_correct = 0
 
    for race in RACES_2026:
        if race.get("actual_podium"):
            circuit = race["circuit"]
            actual = race["actual_podium"]
 
            # Get this race's predictions
            race_preds = all_preds[all_preds["Circuit"] == circuit]
            predicted_top3 = race_preds.head(3)["Driver"].tolist()
            correct = len([d for d in predicted_top3 if d in actual])
 
            total_podium_spots += 3
            total_correct += correct
 
            print(f"Round {race['round']:2d} {circuit:15s}  "
                  f"Predicted: {', '.join(predicted_top3):20s}  "
                  f"Actual: {', '.join(actual):20s}  "
                  f"Score: {correct}/3")
        
    if total_podium_spots > 0:
        accuracy = total_correct / total_podium_spots
        print(f"\nOverall: {total_correct}/{total_podium_spots} podium "
              f"finishers correctly predicted ({accuracy:.0%})")

    # Driver rankings
    print(f"\n{'-' * 70}")
    print(f"Driver Podium Probability Rankings (2026 so far)")
    print(f"{'-' * 70}")
 
    summary = all_preds.groupby("Driver").agg(
        AvgPodiumProb=("PodiumProb", "mean"),
        MaxPodiumProb=("PodiumProb", "max"),
        PredictedPodiums=("Predicted", "sum"),
        RacesEntered=("Predicted", "count"),
    ).sort_values("AvgPodiumProb", ascending=False)
 
    print(f"\n  {'Driver':<6}  {'Avg Prob':>8}  {'Max Prob':>8}"
          f"{'Pred. Podiums':>13}  {'Races':>5}")
    print(f"  {'─' * 50}")
 
    for driver, row in summary.iterrows():
        print(f"{driver:<6}  {row['AvgPodiumProb']:7.1%}"
              f"{row['MaxPodiumProb']:7.1%}"
              f"{int(row['PredictedPodiums']):>13}"
              f"{int(row['RacesEntered']):>5}")
            
            

