import pandas as pd
import numpy as np
from pathlib import Path
import holidays


# Import data
CSV_PATH = Path(__file__).parents[2] / "spiges_meteo_joined.csv"
df = pd.read_csv(CSV_PATH, parse_dates=["date"])


# Add column holidays
df['is_holidays'] = df.apply(lambda row: holidays.Switzerland(subdiv=row['kanton_hospital'], years=row['year']).get(row['date']) is not None, axis=1)


############## Manage data ############## 

df['day_of_week'] = df['day_of_week'].astype('category')
df["is_weekend"] = df["is_weekend"].astype("category")
df["is_winter"] = df["is_winter"].astype("category")
df["is_summer"] = df["is_summer"].astype("category")
df['is_holidays'] = df['is_holidays'].astype(int) # transform the True/False into 0/1
df["is_holidays"] = df["is_holidays"].astype("category")

# Create lag for pct_elderly and mean_age columns
df = df.sort_values(['kanton_hospital', 'date'])
df['pct_elderly_lag1'] = df.groupby('kanton_hospital')['pct_elderly'].shift(1)
df['mean_age_lag1'] = df.groupby('kanton_hospital')['mean_age'].shift(1)
df['mean_nems_lag1'] = df.groupby('kanton_hospital')['mean_nems'].shift(1)
df['ips_cases_lag1'] = df.groupby('kanton_hospital')['ips_cases'].shift(1)
df['mean_severity_lag1'] = df.groupby('kanton_hospital')['mean_severity'].shift(1)

df = df.sort_values("date").reset_index(drop=True)

# Create const
FEATURE_COLS = [
    "month", "day_of_week", "is_weekend", "is_winter", "is_summer", 'is_holidays',
    "notfall_lag1", "notfall_lag7", "notfall_roll7",
    "mean_severity_lag1", "mean_nems_lag1", "ips_cases_lag1", 'mean_age_lag1', "pct_elderly_lag1",
    "temperature_avg", "temperature_max", "temperature_min"]

TARGET_COL = "target_notfall_next24h"

df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS)


############## Create the train, val, test sets ##############

n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]

train_features = train[FEATURE_COLS].reset_index(drop=True)
train_target   = train[TARGET_COL].reset_index(drop=True)

val_features   = val[FEATURE_COLS].reset_index(drop=True)
val_target     = val[TARGET_COL].reset_index(drop=True)

test_features  = test[FEATURE_COLS].reset_index(drop=True)
test_target    = test[TARGET_COL].reset_index(drop=True)
