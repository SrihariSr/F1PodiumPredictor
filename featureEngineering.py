import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Constants for races
SHORT_WINDOW = 3
PRIMARY_WINDOW = 5
LONG_WINDOW = 10

STREET_RACES = ["Monaco", "Singapore", "Azerbaijan", "Saudi Arabia", "Las Vegas", "Miami", "Australia"]

TRAINING_YEARS = [2022, 2023, 2024, 2025]
VALIDATING_SPLIT = {"year": 2025, "max_round": 12}
TESTING_SPLIT = {"year": 2025, "min_round": 13}

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)

    for window in [SHORT_WINDOW, PRIMARY_WINDOW, LONG_WINDOW]:

        # New columns for the different window sizes
        df[f"rolling_avg_finish_{window}"] = np.nan
        df[f"rolling_avg_grid_{window}"] = np.nan
        df[f"rolling_podium_rate_{window}"] = np.nan
        df[f"rolling_dnf_rate_{window}"] = np.nan
        df[f"rolling_avg_points_{window}"] = np.nan

        for driver, group in df.groupby("Driver"):
            
            indices = group.index.tolist()

            finishes = group["FinishPosition"].values
            grids = group["GridPosition"].values
            podiums = group["IsPodium"].values
            # IsClassified=1 means finished, so flip to get 1 for DNF and 0 if finished
            dnfs = 1 - group["IsClassified"].values
            points = group["Points"].values

            for i, index in enumerate(indices):
                for window in [SHORT_WINDOW, PRIMARY_WINDOW, LONG_WINDOW]:
                    start = max(0, i - window)
                    end = i

                    # No previous races
                    if end <= start:
                        continue

                    history_finishes = finishes[start:end]
                    history_grids = grids[start:end]
                    history_podiums = podiums[start:end]
                    history_dnfs = dnfs[start:end]
                    history_points = points[start:end]

                    df.at[index, f"rolling_avg_finish_{window}"] = np.nanmean(history_finishes)
                    df.at[index, f"rolling_avg_grid_{window}"] = np.nanmean(history_grids)
                    df.at[index, f"rolling_podium_rate_{window}"] = np.nanmean(history_podiums)
                    df.at[index, f"rolling_dnf_rate_{window}"] = np.nanmean(history_dnfs)
                    df.at[index, f"rolling_avg_points_{window}"] = np.nanmean(history_points)

    return df

def compute_driver_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum features: streaks and best recent result.
    """
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)
 
    df["streak_podiums"] = 0
    df["streak_points"] = 0
    df["best_finish_last_5"] = np.nan
    df["positions_gained_avg_5"] = np.nan
 
    for driver, group in df.groupby("Driver"):
        indices = group.index.tolist()
        finishes = group["FinishPosition"].values
        grids = group["GridPosition"].values
        podiums = group["IsPodium"].values
        points_scored = group["Points"].values
 
        for i, idx in enumerate(indices):
            if i == 0:
                continue
 
            # Podium streak - count consecutive podiums going backwards
            streak_p = 0
            for j in range(i - 1, -1, -1):
                if podiums[j] == 1:
                    streak_p += 1
                else:
                    break
            df.at[idx, "streak_podiums"] = streak_p
 
            # Points streak - count consecutive points finishes going backwards
            streak_pts = 0
            for j in range(i - 1, -1, -1):
                if points_scored[j] > 0:
                    streak_pts += 1
                else:
                    break
            df.at[idx, "streak_points"] = streak_pts
 
            # Best finish in last 5 races
            start = max(0, i - 5)
            history = finishes[start:i]
            if len(history) > 0:
                df.at[idx, "best_finish_last_5"] = np.nanmin(history)
 
            # Average positions gained per race over last 5
            gains = grids[start:i] - finishes[start:i]
            if len(gains) > 0:
                df.at[idx, "positions_gained_avg_5"] = np.nanmean(gains)
 
    return df


def compute_teammate_features(df: pd.DataFrame) -> pd.DataFrame:

    # Compute how each driver performs relative to their teammate.
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)
    df["teammate_grid_delta"] = np.nan
 
    for (year, rnd), race_group in df.groupby(["Year", "Round"]):
        # Group drivers by team within this race
        for team, team_group in race_group.groupby("Team"):
            # Skip if team does not have 2 drivers
            if len(team_group) != 2:
                continue
 
            drivers = team_group.index.tolist()
            grid_0 = team_group.loc[drivers[0], "GridPosition"]
            grid_1 = team_group.loc[drivers[1], "GridPosition"]
 
            if pd.notna(grid_0) and pd.notna(grid_1):
                # Negative = qualified ahead of teammate
                df.at[drivers[0], "teammate_grid_delta"] = grid_0 - grid_1
                df.at[drivers[1], "teammate_grid_delta"] = grid_1 - grid_0
 
    return df

def compute_circuit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute driver-specific history at each circuit, and circuit-level
    safety car and overtake rates.

    """
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)
    df["driver_circuit_avg_finish"] = np.nan
    df["circuit_sc_rate"] = np.nan
 
    # Track history per driver per circuit
    driver_circuit_history = {}  # (driver, circuit) -> [finishes]
    circuit_sc_history = {}      # circuit -> [had_sc values]
 
    for (year, rnd), race_group in df.groupby(["Year", "Round"], sort=True):
        event_name = race_group["EventName"].iloc[0]
 
        for idx, row in race_group.iterrows():
            driver = row["Driver"]
            key = (driver, event_name)
 
            # Encode using only past visits to this circuit
            if key in driver_circuit_history and len(driver_circuit_history[key]) > 0:
                df.at[idx, "driver_circuit_avg_finish"] = np.nanmean(driver_circuit_history[key])
 
            # Circuit safety car rate from past races here
            if event_name in circuit_sc_history and len(circuit_sc_history[event_name]) > 0:
                df.at[idx, "circuit_sc_rate"] = np.mean(circuit_sc_history[event_name])
 
        # After encoding, record this race's data
        for idx, row in race_group.iterrows():
            driver = row["Driver"]
            key = (driver, event_name)
            finish = row["FinishPosition"]
 
            if key not in driver_circuit_history:
                driver_circuit_history[key] = []
            if pd.notna(finish):
                driver_circuit_history[key].append(finish)
 
        # Record safety car occurrence for this circuit
        had_sc = race_group["HadSafetyCar"].iloc[0] if "HadSafetyCar" in race_group.columns else 0
        if event_name not in circuit_sc_history:
            circuit_sc_history[event_name] = []
        circuit_sc_history[event_name].append(had_sc)
 
    return df

def compute_start_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling average of first-lap position gains.
    Uses PositionsGainedLap1 from dataCollection.py.

    """
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)
    df["first_lap_avg_gain_5"] = np.nan
 
    if "PositionsGainedLap1" not in df.columns:
        return df
 
    for driver, group in df.groupby("Driver"):
        indices = group.index.tolist()
        gains = group["PositionsGainedLap1"].values
 
        for i, idx in enumerate(indices):
            start = max(0, i - 5)
            end = i
 
            if end <= start:
                continue
 
            history = gains[start:end]
            if len(history) > 0:
                df.at[idx, "first_lap_avg_gain_5"] = np.nanmean(history)
 
    return df

def compute_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    
    for (year, rnd), race_group in df.groupby(["Year", "Round"]):
        race_indices = race_group.index
        
        best_q_times = race_group[["Q3_sec", "Q2_sec", "Q1_sec"]].bfill(axis=1).iloc[:, 0]

        pole_time = best_q_times.min()
        median_time = best_q_times.median()
        if pd.notna(pole_time):
            df.loc[race_indices, "quali_delta_to_pole"] = best_q_times - pole_time

        else:
            df.loc[race_indices, "quali_delta_to_pole"] = np.nan

        if pd.notna(median_time):
            df.loc[race_indices, "quali_delta_to_median"] = best_q_times - median_time
        else:
            df.loc[race_indices, "quali_delta_to_median"] = np.nan
        
        # Normalizing grid on a scale from 0 to 1
        num_starters = race_group["GridPosition"].notna().sum()

        if num_starters > 0:
            df.loc[race_indices, "GridPosition_norm"] = race_group["GridPosition"] / num_starters
        else:
            df.loc[race_indices, "GridPosition_norm"] = np.nan
    
    return df

def compute_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)

    df["driver_encoded"] = np.nan
    df["team_encoded"] = np.nan
 
    # Build a running record of podiums per driver and team
    driver_podiums = {}   # driver -> [list of IsPodium values]
    team_podiums = {}     # team -> [list of IsPodium values]
 
    # We need to process race by race to avoid look-ahead
    for (year, rnd), race_group in df.groupby(["Year", "Round"], sort=True):
        race_indices = race_group.index
 
        for idx, row in race_group.iterrows():
            driver = row["Driver"]
            team = row["Team"]
 
            # Encode using ONLY past data
            if driver in driver_podiums and len(driver_podiums[driver]) > 0:
                df.at[idx, "driver_encoded"] = np.mean(driver_podiums[driver])
            # else: stays NaN (first race for this driver)
 
            if team in team_podiums and len(team_podiums[team]) > 0:
                df.at[idx, "team_encoded"] = np.mean(team_podiums[team])
            # else: stays NaN (first race for this team)
 
        # AFTER encoding this race, add its results to the history
        for idx, row in race_group.iterrows():
            driver = row["Driver"]
            team = row["Team"]
            podium = row["IsPodium"]
 
            if driver not in driver_podiums:
                driver_podiums[driver] = []
            driver_podiums[driver].append(podium)
 
            if team not in team_podiums:
                team_podiums[team] = []
            team_podiums[team].append(podium)
 
    return df

def compute_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_street_circuit"] = df["EventName"].apply(lambda name: 1 if any(sc.lower() in name.lower() for sc in STREET_RACES) else 0)
    
    circuit_history = {}

    for (year, rnd), race_group in df.groupby(["Year", "Round"], sort=True):
        race_indices = race_group.index
        event_name = race_group["EventName"].iloc[0]

        if event_name in circuit_history and len(circuit_history[event_name]) > 0:
            df.loc[race_indices, "circuit_encoded"] = np.mean(circuit_history[event_name])

        podium_drivers = race_group[race_group["IsPodium"] == 1]
        if len(podium_drivers) > 0:
            upset_rate = (podium_drivers["GridPosition"] > 5).mean()
        else:
            upset_rate = 0.0
        
        if event_name not in circuit_history:
            circuit_history[event_name] = []
        circuit_history[event_name].append(upset_rate)
 
    return df

def normalise_weather(df: pd.DataFrame) -> pd.DataFrame:

    weather_cols = {
        "AirTemp_mean":   "air_temp_norm",
        "TrackTemp_mean":  "track_temp_norm",
        "Humidity_mean":   "humidity_norm",
        "WindSpeed_mean":  "wind_speed_norm",
    }
 
    # Compute min/max from training years only
    train_mask = df["Year"].isin(TRAINING_YEARS)
 
    for raw_col, norm_col in weather_cols.items():
        if raw_col not in df.columns:
            df[norm_col] = np.nan
            continue
 
        train_min = df.loc[train_mask, raw_col].min()
        train_max = df.loc[train_mask, raw_col].max()
        range_val = train_max - train_min
 
        if range_val > 0:
            df[norm_col] = (df[raw_col] - train_min) / range_val
            # Clip to [0, 1] — val/test values might exceed training range
            df[norm_col] = df[norm_col].clip(0.0, 1.0)
        else:
            df[norm_col] = 0.0
 
    return df
        

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:

    # Grid × Rainfall: rain diminishes grid advantage
    df["grid_x_rainfall"] = df["GridPosition"] * df["Rainfall"]
 
    # Driver × Team form: captures the combined strength
    df["driver_team_form"] = df["driver_encoded"] * df["team_encoded"]
 
    # Form vs Grid gap: does the driver typically finish better or worse
    # than where they qualify? Positive = gains places in races.
    df["form_vs_grid_gap"] = (
        df["rolling_avg_grid_5"] - df["rolling_avg_finish_5"]
    )
 
    return df  

def handle_missing_values(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Fill remaining NaN values with sensible defaults.
 
    Strategy:
        - Rolling features (NaN = no history) → fill with column median
          (assumes rookie/new driver is average until proven otherwise)
        - Qualifying deltas (NaN = no time set) → fill with worst value + margin
          (penalises drivers who didn't set a time)
        - Encodings (NaN = first race) → fill with global mean
          (neutral prior)
        - Everything else → column median
    """
    # Qualifying penalty: fill missing quali deltas with the max observed + 1 second
    for col in ["quali_delta_to_pole", "quali_delta_to_median"]:
        if col in df.columns:
            max_val = df[col].max()
            penalty = max_val + 1.0 if pd.notna(max_val) else 5.0
            df[col] = df[col].fillna(penalty)

    # FP2 data: fill with 2.0 (assume 2 seconds off pace for missing sessions)
    for col in ["FP2_BestLap_Delta", "FP2_LongRun_Delta"]:
        if col in df.columns:
            df[col] = df[col].fillna(2.0)
 
    # All other features: fill with median
    for col in feature_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
 
    return df        

def temporal_split(df, feature_cols, target_col="IsPodium"):
    
    train_mask = df["Year"].isin(TRAINING_YEARS) | (df["Year"] == 2026)
    val_mask = (df["Year"] == VALIDATING_SPLIT["year"]) & (df["Round"] <= VALIDATING_SPLIT["max_round"])
    test_mask = (df["Year"] == TESTING_SPLIT["year"]) & (df["Round"] >= TESTING_SPLIT["min_round"])

    cols_to_keep = feature_cols + [target_col, "Year", "Round", "Driver", "Team", "EventName"]
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]

    train_df = df.loc[train_mask, cols_to_keep].copy()
    val_df = df.loc[val_mask, cols_to_keep].copy()
    test_df = df.loc[test_mask, cols_to_keep].copy()

    return train_df, val_df, test_df

FEATURE_COLUMNS = [
    "GridPosition",
    "GridPosition_norm",
    "quali_delta_to_pole",
    "quali_delta_to_median",
    "rolling_avg_finish_5",
    "rolling_avg_grid_5",
    "rolling_podium_rate_5",
    "rolling_dnf_rate_5",
    "rolling_avg_points_5",
    "rolling_avg_finish_3",
    "rolling_podium_rate_3",
    "rolling_avg_finish_10",
    "rolling_podium_rate_10",
    "driver_encoded",
    "team_encoded",
    "is_street_circuit",
    "circuit_encoded",
    "air_temp_norm",
    "track_temp_norm",
    "humidity_norm",
    "wind_speed_norm",
    "Rainfall",
    "grid_x_rainfall",
    "driver_team_form",
    "form_vs_grid_gap",
]

def run_data_engineering(input_path: str = None) -> tuple:
    if input_path is None:
        input_path = os.path.join(DATA_DIR, "f1_historical_data.csv")
    
    print("Data Engineering")
    print("=" * 60)

    print("Loading raw data...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    print("Computing rolling performance features...")
    df = compute_rolling_features(df)
    print("Added rolling finish, grid, podium, DNF, and points averages")

    print("Computing driver momentum features...")
    df = compute_driver_momentum_features(df)
    print("Added driver momentum features")

    print("Computing relative performance features...")
    df = compute_relative_features(df)
    print("Added quali deltas and grid normalisation")

    print("Computing target encodings (driver & team)...")
    df = compute_target_encoding(df)
    print("Encoded drivers and teams by historical podium rate")

    print("Computing teammate features")
    df = compute_teammate_features(df)
    print("Computed teammate features")
 
    print("Computing circuit features...")
    df = compute_circuit_features(df)
    print("Computed street circuit flag and circuit upset encoding")

    print("Computing circuit history features...")
    df = compute_circuit_history_features(df)
    print("Computed circuit history features")

    print("Computing start features...")
    df = compute_start_features(df)
    print("Computed start features")

    print("Normalising weather features...")
    df = normalise_weather(df)
    print("Min-max normalised weather to [0, 1]")
 
    print("Computing interaction features...")
    df = compute_interaction_features(df)
    print("Added gridxrainfall, driverxteam form, form-vs-grid gap")
 
    print("Handling missing values...")
    df = handle_missing_values(df, FEATURE_COLUMNS)
    remaining_nans = df[FEATURE_COLUMNS].isna().sum().sum()
    print(f"Filled NaNs (remaining: {remaining_nans})")
 
    print("Splitting data temporally...")
    train_df, val_df, test_df = temporal_split(df, FEATURE_COLUMNS)
    print(f"Train: {len(train_df)} rows ({TRAINING_YEARS})")
    print(f"Val:   {len(val_df)} rows (2025 R1-R{VALIDATING_SPLIT['max_round']})")
    print(f"Test:  {len(test_df)} rows (2025 R{TESTING_SPLIT['min_round']}+)")
 
    # Save
    train_path = os.path.join(DATA_DIR, "f1_features_train.csv")
    val_path = os.path.join(DATA_DIR, "f1_features_val.csv")
    test_path = os.path.join(DATA_DIR, "f1_features_test.csv")
 
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
 
    print(f"\n  Saved to:")
    print(f"    {train_path}")
    print(f"    {val_path}")
    print(f"    {test_path}")
 
    return train_df, val_df, test_df, FEATURE_COLUMNS


# Main

if __name__ == "__main__":
    train_df, val_df, test_df, features = run_data_engineering()
 
    print("\n📊 Feature Summary:")
    print(f"   Number of features: {len(features)}")
    print(f"   Feature names:")
    for i, f in enumerate(features):
        col_data = train_df[f] if f in train_df.columns else None
        if col_data is not None:
            print(f"     {i+1:2d}. {f:30s}  mean={col_data.mean():8.3f}  std={col_data.std():8.3f}")
        else:
            print(f"     {i+1:2d}. {f:30s}  (not found)")
 
    print(f"\n   Target distribution (train):")
    print(f"     Podium:     {train_df['IsPodium'].sum():4d} ({train_df['IsPodium'].mean():.1%})")
    print(f"     Non-podium: {(train_df['IsPodium']==0).sum():4d} ({(train_df['IsPodium']==0).mean():.1%})")







        
    
    