"""
Student 19 - Electricity Demand Forecasting
Machine Learning Lab Mini Project

Algorithm: Time-Series Regression using Linear Regression
Target: electricity_demand_mw
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "electricity_demand.csv"


def main():
    # 1. Load the dataset
    df = pd.read_csv(DATA_FILE)

    # 2. Prepare date/time column
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 3. Create time-series features
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["lag_1"] = df["electricity_demand_mw"].shift(1)
    df["lag_24"] = df["electricity_demand_mw"].shift(24)
    df["rolling_24"] = (
        df["electricity_demand_mw"].shift(1).rolling(24).mean()
    )

    df = df.dropna().reset_index(drop=True)

    features = [
        "hour",
        "day_of_week",
        "month",
        "lag_1",
        "lag_24",
        "rolling_24",
    ]
    target = "electricity_demand_mw"

    # 4. Time-based train/test split
    # Time-series data should not be randomly shuffled.
    split = int(len(df) * 0.80)
    train = df.iloc[:split]
    test = df.iloc[split:]

    # 5. Train the regression model
    model = LinearRegression()
    model.fit(train[features], train[target])

    # 6. Predict test data
    predictions = model.predict(test[features])

    # 7. Evaluate the model
    mae = mean_absolute_error(test[target], predictions)
    rmse = np.sqrt(mean_squared_error(test[target], predictions))
    r2 = r2_score(test[target], predictions)

    print("=" * 60)
    print("ELECTRICITY DEMAND FORECASTING - STUDENT 19")
    print("=" * 60)
    print(f"Training records : {len(train)}")
    print(f"Testing records  : {len(test)}")
    print(f"MAE              : {mae:.3f} MW")
    print(f"RMSE             : {rmse:.3f} MW")
    print(f"R2 Score         : {r2:.4f}")
    print("=" * 60)

    # 8. Display sample predictions
    result = test[["datetime", target]].copy()
    result["predicted_demand_mw"] = np.round(predictions, 2)

    print("\nSample predictions:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
