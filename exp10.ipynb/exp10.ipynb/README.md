# Electricity Demand Forecasting - Student 19

Machine Learning Lab Mini Project.

## Project Details

- **Project:** Electricity Demand Forecasting
- **Algorithm:** Time-Series Regression
- **Implementation:** Linear Regression with time-series features
- **Target / Output:** Demand
- **Expected Result:** Forecast electricity demand

> This project contains a **sample/demo dataset** for GitHub and testing. Replace `electricity_demand.csv` with the exact dataset provided by your faculty/Kaggle if your lab requires the original dataset.

## Files

```text
Project_19_Electricity_Demand_Forecasting/
├── electricity_demand_forecasting.py
├── electricity_demand.csv
├── README.md
└── requirements.txt
```

## Dataset

The sample CSV contains two columns:

| Column | Description |
|---|---|
| `datetime` | Date and time of the electricity-demand observation |
| `electricity_demand_mw` | Electricity demand in MW |

The sample contains hourly observations for 30 days.

## How the Program Works

1. Load the electricity-demand CSV file.
2. Convert `datetime` into a proper date/time value.
3. Create time-series features:
   - Hour
   - Day of week
   - Month
   - Previous-hour demand (`lag_1`)
   - Previous-day demand (`lag_24`)
   - 24-hour rolling average
4. Split the data chronologically into 80% training and 20% testing data.
5. Train a Linear Regression model.
6. Predict electricity demand for the test period.
7. Calculate MAE, RMSE, and R² score.
8. Display sample actual and predicted demand values.

## Installation

Open Command Prompt/PowerShell inside this project folder and run:

```bash
pip install -r requirements.txt
```

## Run

```bash
python electricity_demand_forecasting.py
```

## Sample Output

Using the included sample dataset:

```text
============================================================
ELECTRICITY DEMAND FORECASTING - STUDENT 19
============================================================
Training records : 556
Testing records  : 140
MAE              : 2.421 MW
RMSE             : 2.965 MW
R2 Score         : 0.8645
============================================================
```

The exact values can change slightly if the sample dataset is replaced with another dataset.

## GitHub Upload

Upload these four files to your repository:

- `electricity_demand_forecasting.py`
- `electricity_demand.csv`
- `README.md`
- `requirements.txt`

Then run the Python program from the repository folder.

## Note

The project allocation specifies **Time-Series Regression** for Project 19. The provided program implements this using **Linear Regression** and lag/rolling time-series features.
