import pandas as pd
from pathlib import Path

meteo = pd.read_csv("../data/meteo_daily_by_canton.csv", parse_dates=["date"])
spiges = pd.read_csv("../data/spiges_daily_aggregated.csv", parse_dates=["eintrittsdatum_dt"])

merged = spiges.merge(
    meteo,
    left_on=["kanton_hospital", "eintrittsdatum_dt"],
    right_on=["canton_abbr", "date"],
    how="inner",   # use "left" if you want to keep spiges rows without meteo matches
)

Path("data").mkdir(exist_ok=True)
merged.to_csv("../data/spiges_meteo_joined.csv", index=False)

print(f"Rows: spiges={len(spiges)}, meteo={len(meteo)}, merged={len(merged)}")
