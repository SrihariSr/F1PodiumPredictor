# F1 Podium Predictor using Machine Learning 🏎️

A Machine Learning pipeline that predicts Formula 1 podium finishers using historical race data. Built using PyTorch and XGBoost and trained on 4 seasons (2022–2025) of F1 data collected using the FastF1 API.

**2026 season accuracy (using the first 3 races):** XGBoost: 78%, Neural Network: 67%

## How it works

The pipeline collects race results, qualifying times, weather data, and lap pace statistics from every F1 race since 2022. It engineers 25 features per driver per race such as rolling performance averages, qualifying deltas, team/driver strength encodings, circuit characteristics, and weather conditions. Two models are trained on this data: a feedforward neural network using PyTorch, and gradient boosted trees using XGBoost. Given a race weekend's qualifying results, the models output podium probabilities for all 22 drivers.

## Results

### 2026 Season Predictions (Gradient Boosted Trees)

| Race | Predicted | Actual | Score |
|------|-----------|--------|-------|
| Australia | RUS, ANT, LEC | RUS, ANT, LEC | 3/3 |
| China | ANT, RUS, LEC | ANT, RUS, HAM | 2/3 |
| Japan | ANT, RUS, LEC | ANT, PIA, LEC | 2/3 |
| **Overall** | | | **7/9 (78%)** |

### Model Comparison

| Metric | Neural Network | Gradient Boosted Trees |
|--------|---------------|---------|
| 2026 accuracy | 6/9 (67%) | 7/9 (78%) |
| Test set F1 | 0.923 | 0.973 |
| Test precision | 85.7% | 94.7% |
| Test recall | 100% | 100% |
| Training time | ~2 min | ~5 sec |

## Pipeline

```
Step 1: Data Collection     → FastF1 API → race results, qualifying, weather, lap data
Step 2: Feature Engineering → 25 features per driver per race
Step 3: Model Training      → PyTorch neural network and XGBoost
Step 4: Prediction          → Podium probabilities for upcoming races
Step 5: Evaluation          → Performance charts and accuracy tracking
```

## Features

The model uses 25 engineered features grouped into 7 categories:

**Grid & qualifying**: grid position, normalised grid position, qualifying delta to pole, qualifying delta to median

**Rolling driver form**: average finish position and podium rate over 3 and 10-race windows; average finish position, grid position, podium rate, DNF rate, and points average over a 5-race window

**Driver & team strength**: target-encoded historical podium rates for each driver and team, updated race by race without data leakage

**Circuit characteristics**: street circuit flag, circuit upset rate encoding

**Weather**: normalised air temperature, track temperature, humidity, wind speed, rainfall flag

**Interaction features**: grid position × rainfall, driver × team form product, form vs grid gap

## Usage

### Full pipeline (first run)

```bash
# Step 1: Collect historical data (takes a while due to FastF1 API's rate limits)
python dataCollection.py

# Step 2: Engineer features
python featureEngineering.py

# Step 3: Train models
python oldRacesPredictingNN.py    # Neural network
python xgboostModel.py            # XGBoost

# Step 4: Predict 2026 races
python futurePredictingNN.py      # Neural network predictions
python xgboostModel.py            # XGBoost predictions (included in training script)
```

### Predicting a new race

After Saturday qualifying, add the results to the `RACES_2026` list in `futurePredictingNN.py`:

Example:
```python
{
    "circuit": "Miami",
    "round": 4,
    "weather": {"air_temp": 30, "track_temp": 45, "humidity": 70, "wind_speed": 4, "rainfall": 0},
    "qualifying": [
        {"driver": "ANT", "team": "Mercedes", "grid_position": 1, "q_time_sec": 87.500},
        {"driver": "RUS", "team": "Mercedes", "grid_position": 2, "q_time_sec": 87.800},
        # ... all 22 drivers
    ],
    "actual_podium": [],  # fill in after the race
},
```

Then run the prediction script.

### Retraining after new races

As the 2026 season progresses, retrain the model to incorporate new data:

```bash
python dataCollection.py        # downloads new race data
python featureEngineering.py    # rebuilds features
python oldRacesPredictingNN.py  # retrains neural network
python xgboostModel.py          # retrains XGBoost
```

## Architecture

### Neural Network (PyTorch)

```
Input (25 features)
  → Linear(25, 64) → ReLU → Dropout(0.35)
  → Linear(64, 32) → ReLU → Dropout(0.35)
  → Linear(32, 16) → ReLU
  → Linear(16, 1)  → Sigmoid
  → Podium probability in the range [0, 1]
```

- Parameters: ~5,000
- Loss: Weighted binary cross-entropy (pos_weight=3.5)
- Optimiser: Adam (lr=0.001)
- Regularisation: dropout, early stopping, learning rate scheduling
- Training: up to 200 epochs (early stopping typically halts around epoch 140–150)

### Gradient Boosted Trees (XGBoost)

- 500 gradient boosted trees (max_depth=3)
- Learning rate: 0.1
- Regularisation: subsample=0.7, colsample_bytree=0.7, min_child_weight=5, gamma=0.3
- Class weighting: scale_pos_weight=5.67 (handles 85/15 class imbalance)

## Key Design Decisions

**Temporal splitting over random splitting.** Training data uses all four seasons (2022–2025), with 2025 rounds 1–12 as the validation set and 2025 rounds 13+ as the test set. Completed 2026 races are also added to the training set after predictions are made and evaluated. Splitting by time (rather than randomly shuffling) avoids the worst form of leakage: if rows were shuffled, a model trained on a 2025 round-15 feature set could be "tested" on round-2 of the same year, where the rolling averages for both draws from the same pool. The temporal ordering ensures that rolling features computed for any race only draw on races that occurred before it.

**Rolling features use past data only.** When computing a driver's average finish over their last 5 races, only races before the current one are used. This prevents data leakage where the model could learn the result it is trying to predict.

**Weighted loss function.** Only 15% of race entries result in podiums. Without class weighting, the model could achieve 85% accuracy by always predicting "no podium." The pos_weight parameter penalises missed podiums 3.5x more than false alarms.

## Limitations

- **Small sample size.** ~1,900 total samples across 4 seasons. Better models use way more samples to train.
- **Regulation changes.** The 2026 regulations fundamentally changed car performance. Models trained on 2022-2025 data don't fully capture the 2026 competitive order.
- **No race-day dynamics.** The model predicts from qualifying results and historical form. It can't account for first-lap incidents, safety cars, mechanical failures, or tyre strategy decisions that happen during the race.
- **Grid position dominance.** ~80% of XGBoost's decision-making relies on grid position. The model essentially learns "front row starters usually podium" with minor adjustments from other features.

## Acknowledgements

- [FastF1](https://github.com/theOehrly/Fast-F1) for the F1 data API.
- [PyTorch](https://pytorch.org/) for the neural network framework.
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosted trees.