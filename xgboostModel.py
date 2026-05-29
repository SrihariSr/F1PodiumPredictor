import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import xgboost as xgb
import numpy as np
import pandas as pd


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


# Metrics
def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# Main

if __name__ == "__main__":
    print("XGBoost Model Training")
    print("=" * 60)

    # ── Load Data ─────────────────────────────────────────────────────────
    train_df = pd.read_csv(os.path.join(DATA_DIR, "f1_features_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "f1_features_val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "f1_features_test.csv"))

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COL].values

    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")
    print(f"Podium ratio — train: {y_train.mean():.1%}, "
          f"val: {y_val.mean():.1%}, test: {y_test.mean():.1%}")

    # Class imbalance weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / n_pos
    print(f"scale_pos_weight: {scale_pos:.2f}")

    # ── XGBoost Parameters ────────────────────────────────────────────────
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "scale_pos_weight": scale_pos,
        "min_child_weight": 5,
        "gamma": 0.3,
        "seed": 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLUMNS)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLUMNS)

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\nTraining...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    val_probs = model.predict(dval)
    test_probs = model.predict(dtest)

    val_metrics = compute_metrics(y_val, val_probs)
    test_metrics = compute_metrics(y_test, test_probs)

    print(f"\n{'=' * 60}")
    print(f"Model Results")
    print(f"{'=' * 60}")

    print(f"\n  Validation Set:")
    print(f"    Accuracy:  {val_metrics['accuracy']:.1%}")
    print(f"    Precision: {val_metrics['precision']:.1%}")
    print(f"    Recall:    {val_metrics['recall']:.1%}")
    print(f"    F1 Score:  {val_metrics['f1']:.3f}")
    print(f"    TP: {val_metrics['tp']}  FP: {val_metrics['fp']}  "
          f"FN: {val_metrics['fn']}  TN: {val_metrics['tn']}")

    print(f"\n  Test Set:")
    print(f"    Accuracy:  {test_metrics['accuracy']:.1%}")
    print(f"    Precision: {test_metrics['precision']:.1%}")
    print(f"    Recall:    {test_metrics['recall']:.1%}")
    print(f"    F1 Score:  {test_metrics['f1']:.3f}")
    print(f"    TP: {test_metrics['tp']}  FP: {test_metrics['fp']}  "
          f"FN: {test_metrics['fn']}  TN: {test_metrics['tn']}")

    # ── Feature Importance ────────────────────────────────────────────────
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 features by importance:")
    for feat, score in sorted_imp[:10]:
        print(f"    {feat:30s}  {score:.1f}")

    # ── Save Model ────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "f1_xgboost_model.json")
    model.save_model(model_path)
    print(f"\n  Model saved to {model_path}")

    # ══════════════════════════════════════════════════════════════════════
    # 2026 PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("2026 Race Predictions (XGBoost)")
    print(f"{'=' * 60}")

    # Import prediction helpers from existing file
    from futurePredictingNN import (
        RACES_2026, build_race_features, load_historical_stats,
        DEFAULT_DRIVER_STATS, DEFAULT_TEAM_STATS, DEFAULT_PRACTICE,
    )

    # Load historical stats
    print("\n  Loading historical stats...")
    driver_stats, team_stats, circuit_stats = load_historical_stats()

    all_predictions = []

    for race in RACES_2026:
        circuit = race["circuit"]
        weather = race["weather"]
        qualifying = race["qualifying"]
        actual_podium = race.get("actual_podium", [])

        # Build features using the same function as the neural network
        features = build_race_features(
            qualifying_results=qualifying,
            circuit_name=circuit,
            weather=weather,
            driver_stats=driver_stats,
            team_stats=team_stats,
            circuit_stats=circuit_stats,
        )

        # XGBoost prediction
        X_race = features[FEATURE_COLUMNS].values
        drace = xgb.DMatrix(X_race, feature_names=FEATURE_COLUMNS)
        probs = model.predict(drace)

        features["PodiumProb"] = probs
        features["Predicted"] = (probs >= 0.5).astype(int)
        results = features.sort_values("PodiumProb", ascending=False)

        # Print predictions
        print(f"\n  🏁 {circuit} Grand Prix — Podium Predictions")
        print(f"  {'─' * 65}")
        print(f"  {'Rank':>4}  {'Driver':<6}  {'Team':<18}  {'Grid':<6}  "
              f"{'Prob':>6}  Podium?")
        print(f"  {'─' * 65}")

        for i, (_, row) in enumerate(results.iterrows()):
            prob = row["PodiumProb"]
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            medal = ""
            if i < 3 and prob > 0:
                medal = ["🥇", "🥈", "🥉"][i]

            print(f"  {i+1:4d}  {row['Driver']:<6s}  {row['Team']:<18s}  "
                  f"P{int(row['GridPosition']):<5d}  {prob:5.1%}  {bar}  {medal}")

        predicted_top3 = results.head(3)["Driver"].tolist()
        num_predicted = (results["PodiumProb"] >= 0.5).sum()
        print(f"\n  Predicted podium finishers: {num_predicted}")
        print(f"  Most likely podium: {', '.join(predicted_top3)}")

        if actual_podium:
            correct = [d for d in predicted_top3 if d in actual_podium]
            missed = [d for d in actual_podium if d not in predicted_top3]

            print(f"\n  Actual podium:    {', '.join(actual_podium)}")
            print(f"  Predicted top 3:  {', '.join(predicted_top3)}")
            print(f"  Correct:          {len(correct)}/3 "
                  f"({', '.join(correct) if correct else 'none'})")
            if missed:
                print(f"  Missed:           {', '.join(missed)}")

        # Store for CSV
        results["Circuit"] = circuit
        results["Round"] = race["round"]
        all_predictions.append(results)

    # ── Save Predictions ──────────────────────────────────────────────────
    all_preds = pd.concat(all_predictions, ignore_index=True)
    output_path = os.path.join(DATA_DIR, "xgb_predictions_2026.csv")
    all_preds.to_csv(output_path, index=False)
    print(f"\n  All predictions saved to {output_path}")

    # ── Season Accuracy ───────────────────────────────────────────────────
    print(f"\n{'-' * 60}")
    print(f"2026 Season Prediction Accuracy (XGBoost)")
    print(f"{'-' * 60}")

    total_podium_spots = 0
    total_correct = 0

    for race in RACES_2026:
        if race.get("actual_podium"):
            circuit = race["circuit"]
            actual = race["actual_podium"]

            race_preds = all_preds[all_preds["Circuit"] == circuit]
            predicted_top3 = race_preds.head(3)["Driver"].tolist()
            correct = len([d for d in predicted_top3 if d in actual])

            total_podium_spots += 3
            total_correct += correct

            print(f"  Round {race['round']:2d} {circuit:15s}  "
                  f"Predicted: {', '.join(predicted_top3):20s}  "
                  f"Actual: {', '.join(actual):20s}  "
                  f"Score: {correct}/3")

    if total_podium_spots > 0:
        accuracy = total_correct / total_podium_spots
        print(f"\n  Overall: {total_correct}/{total_podium_spots} podium "
              f"finishers correctly predicted ({accuracy:.0%})")

    # ── Driver Rankings ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  📊 Driver Podium Probability Rankings (2026 — XGBoost)")
    print(f"{'=' * 60}")

    summary = all_preds.groupby("Driver").agg(
        AvgPodiumProb=("PodiumProb", "mean"),
        MaxPodiumProb=("PodiumProb", "max"),
        PredictedPodiums=("Predicted", "sum"),
        RacesEntered=("Predicted", "count"),
    ).sort_values("AvgPodiumProb", ascending=False)

    print(f"\n  {'Driver':<6}  {'Avg Prob':>8}  {'Max Prob':>8}"
          f"  {'Pred. Podiums':>13}  {'Races':>5}")
    print(f"  {'─' * 50}")

    for driver, row in summary.iterrows():
        print(f"  {driver:<6}  {row['AvgPodiumProb']:7.1%}"
              f"  {row['MaxPodiumProb']:7.1%}"
              f"  {int(row['PredictedPodiums']):>13}"
              f"  {int(row['RacesEntered']):>5}")
