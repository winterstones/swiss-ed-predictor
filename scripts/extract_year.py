from pathlib import Path
import pandas as pd

YEAR = 2025
SRC  = Path("data/historical")
DST  = SRC / str(YEAR)
DST.mkdir(parents=True, exist_ok=True)

for csv_path in sorted(SRC.glob("*.csv")):
    df = pd.read_csv(csv_path)

    ts_col = next((c for c in df.columns if "reference" in c.lower()), None)
    if ts_col is None:
        print(f"  skip {csv_path.name}: no timestamp column")
        continue

    df[ts_col] = pd.to_datetime(df[ts_col], format="%d.%m.%Y %H:%M", errors="coerce")
    year_df = df[df[ts_col].dt.year == YEAR]

    if year_df.empty:
        print(f"  skip {csv_path.name}: no data for {YEAR}")
        continue

    out_path = DST / f"{csv_path.stem}_{YEAR}.csv"
    if out_path.exists():
        continue                                 
    year_df.to_csv(out_path, index=False)
    print(f"  {csv_path.name} → {out_path.relative_to(SRC)} ({len(year_df)} rows)")
