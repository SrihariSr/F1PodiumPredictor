import fastf1
import pandas as pd
import numpy as np
import os
import warnings
import logging
import time
from datetime import timedelta

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.WARNING)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

YEARS = [2022, 2023, 2024, 2025, 2026]

# Helper method - Convert timedelta to seconds
def td_to_sec(td):
    # Safely convert a pandas Timedelta to float seconds
    if pd.isna(td):
        return np.nan
    if isinstance(td, timedelta):
        return td.total_seconds()
    return np.nan


# Collect Race Results
def collect_race_results(session, year: int, round_num: int, event_name: str) -> pd.DataFrame:
    # Extract race results from a pre-loaded race session.
    try:
        results = session.results

        df = pd.DataFrame({
            "Year":           year,
            "Round":          round_num,
            "EventName":      event_name,
            "Driver":         results["Abbreviation"],
            "Team":           results["TeamName"],
            "GridPosition":   pd.to_numeric(results["GridPosition"], errors="coerce"),
            "FinishPosition": pd.to_numeric(results["Position"], errors="coerce"),
            "Points":         pd.to_numeric(results["Points"], errors="coerce"),
            "Status":         results["Status"],
        })

        # Binary: did the driver finish the race?
        df["IsClassified"] = df["Status"].apply(
            lambda s: 1 if s in ["Finished", "+1 Lap", "+2 Laps", "+3 Laps"] else 0
        )

        # TARGET: Was the driver on the podium? (top 3)
        df["IsPodium"] = (df["FinishPosition"] <= 3).astype(int)

        return df

    except Exception as e:
        print(f"Race results failed for {year} R{round_num}: {e}")
        return pd.DataFrame()

def collect_practice_data(year: int, round_num: int) -> pd.DataFrame:
    try:
        session = fastf1.get_session(year, round_num, "FP2")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps

        records = []
        for driver in laps["Driver"].unique():
            driver_laps = laps.pick_drivers(driver).pick_accurate()
            if driver_laps.empty:
                continue

            lap_secs = driver_laps["LapTime"].apply(td_to_sec).dropna()
            if len(lap_secs) == 0:
                continue

            # Best single lap
            best_lap = lap_secs.min()

            # Long run pace: stints of 5+ consecutive laps
            # Group by stint (gaps between pit stops)
            tyre_stints = driver_laps.groupby("Stint")
            long_run_times = []
            for stint_id, stint in tyre_stints:
                stint_secs = stint["LapTime"].apply(td_to_sec).dropna()
                if len(stint_secs) >= 5:
                    long_run_times.extend(stint_secs.tolist())

            long_run_pace = np.mean(long_run_times) if long_run_times else np.nan

            records.append({
                "Year": year,
                "Round": round_num,
                "Driver": driver,
                "FP2_BestLap_sec": best_lap,
                "FP2_LongRunPace_sec": long_run_pace,
            })

        df = pd.DataFrame(records)

        # Compute delta to fastest
        if not df.empty and "FP2_BestLap_sec" in df.columns:
            fastest = df["FP2_BestLap_sec"].min()
            df["FP2_BestLap_Delta"] = df["FP2_BestLap_sec"] - fastest

            if df["FP2_LongRunPace_sec"].notna().any():
                fastest_lr = df["FP2_LongRunPace_sec"].min()
                df["FP2_LongRun_Delta"] = df["FP2_LongRunPace_sec"] - fastest_lr
            else:
                df["FP2_LongRun_Delta"] = np.nan

        return df

    except Exception as e:
        print(f"Practice data failed for {year} R{round_num}: {e}")
        return pd.DataFrame()

# Collect Qualifying Data
def collect_qualifying(year: int, round_num: int) -> pd.DataFrame:
    """
    Fetch qualifying session times (Q1, Q2, Q3).

    Returns a DataFrame with columns:
        Year, Round, Driver, Q1_sec, Q2_sec, Q3_sec, QualifyingPosition
    """
    try:
        session = fastf1.get_session(year, round_num, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results

        df = pd.DataFrame({
            "Year":    year,
            "Round":   round_num,
            "Driver":  results["Abbreviation"],
            "Q1_sec":  results["Q1"].apply(td_to_sec),
            "Q2_sec":  results["Q2"].apply(td_to_sec),
            "Q3_sec":  results["Q3"].apply(td_to_sec),
            "QualifyingPosition": pd.to_numeric(results["Position"], errors="coerce"),
        })

        return df

    except Exception as e:
        print(f"  ⚠ Qualifying failed for {year} R{round_num}: {e}")
        return pd.DataFrame()


# Collect Weather Data
def collect_weather(session, year: int, round_num: int) -> pd.DataFrame:
    """
    Extract weather data from a pre-loaded race session.
    """
    try:
        weather = session.weather_data

        if weather is None or weather.empty:
            return pd.DataFrame()

        row = {
            "Year":            year,
            "Round":           round_num,
            "AirTemp_mean":    weather["AirTemp"].mean(),
            "TrackTemp_mean":  weather["TrackTemp"].mean(),
            "Humidity_mean":   weather["Humidity"].mean(),
            "WindSpeed_mean":  weather["WindSpeed"].mean(),
            "Rainfall":        int(weather["Rainfall"].any()),
        }

        return pd.DataFrame([row])

    except Exception as e:
        print(f"Weather failed for {year} R{round_num}: {e}")
        return pd.DataFrame()


# Collect Lap Pace Data
def collect_pace_data(session, year: int, round_num: int) -> pd.DataFrame:
    """
    Extract lap-level pace stats from a pre-loaded race session.
    """
    try:
        laps = session.laps

        records = []
        for driver in laps["Driver"].unique():
            driver_laps = laps.pick_drivers(driver).pick_accurate()

            if driver_laps.empty:
                continue

            # Convert times to seconds
            lap_secs = driver_laps["LapTime"].apply(td_to_sec).dropna()
            s1_secs = driver_laps["Sector1Time"].apply(td_to_sec).dropna()
            s2_secs = driver_laps["Sector2Time"].apply(td_to_sec).dropna()
            s3_secs = driver_laps["Sector3Time"].apply(td_to_sec).dropna()

            # Count pit stops (laps where PitInTime is not NaT)
            pit_stops = driver_laps["PitInTime"].notna().sum()

            # Most used tyre compound
            compound_counts = driver_laps["Compound"].value_counts()
            dominant_compound = compound_counts.index[0] if not compound_counts.empty else "UNKNOWN"

            records.append({
                "Year":               year,
                "Round":              round_num,
                "Driver":             driver,
                "MeanLapTime_sec":    lap_secs.mean() if len(lap_secs) > 0 else np.nan,
                "StdLapTime_sec":     lap_secs.std() if len(lap_secs) > 1 else np.nan,
                "BestLapTime_sec":    lap_secs.min() if len(lap_secs) > 0 else np.nan,
                "MeanS1_sec":         s1_secs.mean() if len(s1_secs) > 0 else np.nan,
                "MeanS2_sec":         s2_secs.mean() if len(s2_secs) > 0 else np.nan,
                "MeanS3_sec":         s3_secs.mean() if len(s3_secs) > 0 else np.nan,
                "NumPitStops":        pit_stops,
                "DominantCompound":   dominant_compound,
            })

        return pd.DataFrame(records)

    except Exception as e:
        print(f"Pace data failed for {year} R{round_num}: {e}")
        return pd.DataFrame()


# Master Collection Pipeline
def collect_all_data(years: list[int] = YEARS) -> pd.DataFrame:

    all_data = []

    for year in years:
        print(f"\n{'='*60}")
        print(f"Collecting {year} season...")
        print(f"{'='*60}")

        # Get the schedule for this year
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"Could not load {year} schedule: {e}")
            continue

        for _, event in schedule.iterrows():
            round_num = event["RoundNumber"]
            event_name = event["EventName"]

            if round_num == 0:  # Skip testing
                continue

            print(f"Round {round_num}: {event_name}")

            # Load race session once (with weather) and reuse across all extractors
            try:
                race_session = fastf1.get_session(year, round_num, "R")
                race_session.load(telemetry=False, weather=True, messages=False)
            except Exception as e:
                print(f"Failed to load race session for {year} R{round_num}: {e}")
                continue

            # Collect each data source
            race_df = collect_race_results(race_session, year, round_num, event_name)
            if race_df.empty:
                continue

            quali_df = collect_qualifying(year, round_num)
            time.sleep(10)  # delay between the two session loads (R and Q)

            weather_df = collect_weather(race_session, year, round_num)
            pace_df = collect_pace_data(race_session, year, round_num)
            practice_df = collect_practice_data(year, round_num)

            # Merge everything onto the race results
            merged = race_df.copy()

            if not quali_df.empty:
                merged = merged.merge(
                    quali_df, on=["Year", "Round", "Driver"], how="left"
                )

            
            if not practice_df.empty:
                merged = merged.merge(
                    practice_df, on=["Year", "Round", "Driver"], how="left"
                )

            if not weather_df.empty:
                merged = merged.merge(
                    weather_df, on=["Year", "Round"], how="left"
                )

            if not pace_df.empty:
                merged = merged.merge(
                    pace_df, on=["Year", "Round", "Driver"], how="left"
                )
            
            first_lap_df = collect_first_lap_data(race_session, year, round_num)
            if not first_lap_df.empty:
                merged = merged.merge(
                    first_lap_df, on=["Year", "Round", "Driver"], how="left"
                )
 
            # Safety car data (per-race, same for all drivers)
            sc_df = collect_safety_car_data(race_session, year, round_num)
            if not sc_df.empty:
                merged = merged.merge(
                    sc_df, on=["Year", "Round"], how="left"
                )

            all_data.append(merged)
            time.sleep(5)

    if not all_data:
        print("No data collected!")
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)

    # Save to CSV
    output_path = os.path.join(DATA_DIR, "f1_historical_data.csv")
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(final_df)} rows to {output_path}")
    print(f"   Columns: {list(final_df.columns)}")
    print(f"   Podium ratio: {final_df['IsPodium'].mean():.2%}")

    return final_df

def collect_first_lap_data(session, year: int, round_num: int) -> pd.DataFrame:
    """
    Extract first-lap position changes and safety car data from a race session.
 
    Returns a DataFrame with columns:
        Year, Round, Driver, LapOnePosition, PositionsGainedLap1
    """
    try:
        laps = session.laps
 
        records = []
        for driver in laps["Driver"].unique():
            driver_laps = laps.pick_drivers(driver)
 
            if driver_laps.empty:
                continue
 
            # Get lap 1 data
            lap1 = driver_laps[driver_laps["LapNumber"] == 1]
            if lap1.empty:
                continue
 
            lap1_pos = lap1["Position"].iloc[0]
 
            # Grid position from session results
            results = session.results
            driver_result = results[results["Abbreviation"] == driver]
            if driver_result.empty:
                continue
 
            grid_pos = pd.to_numeric(driver_result["GridPosition"].iloc[0], errors="coerce")
 
            # Positions gained on lap 1 (positive = gained, negative = lost)
            if pd.notna(grid_pos) and pd.notna(lap1_pos):
                positions_gained = grid_pos - lap1_pos
            else:
                positions_gained = 0
 
            records.append({
                "Year": year,
                "Round": round_num,
                "Driver": driver,
                "LapOnePosition": lap1_pos,
                "PositionsGainedLap1": positions_gained,
            })
 
        return pd.DataFrame(records)
 
    except Exception as e:
        print(f"  ⚠ First lap data failed for {year} R{round_num}: {e}")
        return pd.DataFrame()
 
 
def collect_safety_car_data(session, year: int, round_num: int) -> pd.DataFrame:
    """
    Check if a safety car was deployed during the race.
 
    Returns a single-row DataFrame with:
        Year, Round, HadSafetyCar (1 or 0)
    """
    try:
        laps = session.laps
 
        # Safety car laps have TrackStatus containing '4' (SC) or '6' (VSC)
        if "TrackStatus" in laps.columns:
            sc_laps = laps["TrackStatus"].astype(str).str.contains("4|6", na=False).any()
            had_sc = int(sc_laps)
        else:
            had_sc = 0
 
        return pd.DataFrame([{
            "Year": year,
            "Round": round_num,
            "HadSafetyCar": had_sc,
        }])
 
    except Exception as e:
        print(f"Safety car data failed for {year} R{round_num}: {e}")
        return pd.DataFrame()


# Main
if __name__ == "__main__":
    print("Data Collection")
    print("=" * 60)
    df = collect_all_data()

    if not df.empty:
        print("\nData Summary:")
        print(f"   Total samples:     {len(df)}")
        print(f"   Unique drivers:    {df['Driver'].nunique()}")
        print(f"   Unique circuits:   {df['EventName'].nunique()}")
        print(f"   Seasons covered:   {sorted(df['Year'].unique())}")
        print(f"   Podium finishes:   {df['IsPodium'].sum()}")
        print(f"   Non-podium:        {(df['IsPodium'] == 0).sum()}")