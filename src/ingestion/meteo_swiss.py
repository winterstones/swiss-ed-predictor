"""
MeteoSwiss Open Data Connector — v2
====================================
Fetches and engineers weather features from 3 MétéoSuisse STAC collections:

  1. SMN  — Stations météorologiques automatiques (ch.meteoschweiz.ogd-smn)
            → température, précipitations, vent, humidité, ensoleillement
            → granularité journalière (fichiers *_d.csv)
            → PRIORITÉ 1 — corrélation directe avec pics urgences

  2. POLLEN — Stations pollen (ch.meteoschweiz.ogd-pollen)
              → concentration pollinique horaire/journalière
              → PRIORITÉ 2 — crises asthme/allergies → urgences respiratoires

  3. NBCN — Stations climatologiques homogènes (ch.meteoschweiz.ogd-nbcn)
            → normales historiques → calcul d'anomalies thermiques
            → PRIORITÉ 3 — "il fait 5°C de plus que la normale" est plus
                            prédictif que la valeur absolue

Mapping stations ↔ hôpitaux
-----------------------------
  BE  → BER (Bern / Zollikofen)
  ZH  → SMA (Zürich / Fluntern)
  GE  → GVE (Genève-Cointrin)
  VD  → PUY (Lausanne / La Pully)
  BS  → BAS (Basel / Binningen)
  AG  → SHA (Schaffhausen) ou AIO (Altdorf)
  SG  → STG (St. Gallen)
  VS  → SIO (Sion)
  NE  → NEU (Neuchâtel)
  TI  → LUG (Lugano)

Source: MétéoSuisse Open Data — Licence CC-BY
Référence: https://opendatadocs.meteoswiss.ch/a-data-groundbased/a1-automatic-weather-stations
"""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
import pandas as pd
from loguru import logger

# ── STAC Collection base URLs ─────────────────────────────────────
_STAC_BASE   = "https://data.geo.admin.ch/api/stac/v1/collections"
_DATA_BASE   = "https://data.geo.admin.ch"

SMN_COLLECTION    = "ch.meteoschweiz.ogd-smn"
POLLEN_COLLECTION = "ch.meteoschweiz.ogd-pollen"
NBCN_COLLECTION   = "ch.meteoschweiz.ogd-nbcn"

# ── Station → Canton mapping (closest to hospital) ───────────────
HOSPITAL_STATIONS: dict[str, str] = {
    "BE": "BER",   # Bern / Zollikofen   → Inselspital
    "ZH": "SMA",   # Zürich Fluntern     → USZ
    "GE": "GVE",   # Genève-Cointrin     → HUG
    "VD": "PUY",   # Lausanne La Pully   → CHUV
    "BS": "BAS",   # Basel Binningen     → USB
    "AG": "ALT",   # Altdorf / Aarau     → KSA
    "SG": "STG",   # St. Gallen          → KSSG
    "VS": "SIO",   # Sion                → Hôpital Valais
    "NE": "NEU",   # Neuchâtel           → HNE
    "TI": "LUG",   # Lugano              → EOC
}

# ── SMN parameters we actually need ──────────────────────────────
# Full parameter list: ogd-smn_meta_parameters.csv
SMN_PARAMS = {
    "tre200d0": "temp_mean_c",         # Température moyenne journalière 2m
    "tre200dn": "temp_min_c",          # Température minimale journalière 2m
    "tre200dx": "temp_max_c",          # Température maximale journalière 2m
    "rre150d0": "precipitation_mm",    # Précipitations journalières totales
    "sre000d0": "sunshine_min",        # Durée d'ensoleillement (minutes)
    "ure200d0": "humidity_pct",        # Humidité relative moyenne
    "fu3010d0": "wind_speed_ms",       # Vitesse du vent moyen 10m
    "fu3010dx": "wind_gust_ms",        # Rafale max journalière
    "prestas0": "pressure_hpa",        # Pression atmosphérique au niveau station
}


# ═══════════════════════════════════════════════════════════════════
# 1.  SMN — Stations météorologiques automatiques
# ═══════════════════════════════════════════════════════════════════

def fetch_smn_daily(
    station_id: str,
    start_date: date,
    end_date: date,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch daily SMN data for one station via STAC API.

    File pattern:
      {DATA_BASE}/{collection}/{station_abbr.lower()}/
        ogd-smn_{station_abbr.lower()}_d_{period}.csv

    period = "historical" | "recent" | "now"
    We use "historical" for past data and "recent" for current year.

    Args:
        station_id:  MétéoSuisse station abbreviation (e.g. "BER")
        start_date:  First day of interest
        end_date:    Last day of interest
        timeout:     HTTP timeout in seconds

    Returns:
        DataFrame with one row per day, columns renamed to English ML names.
    """
    station = station_id.upper()
    sid     = station.lower()
    frames  = []

    # Choose periods to fetch
    current_year = date.today().year
    periods = []
    if start_date.year < current_year:
        periods.append("historical")
    if end_date.year >= current_year:
        periods.append("recent")

    for period in periods:
        url = (
            f"{_DATA_BASE}/{SMN_COLLECTION}/{sid}/"
            f"ogd-smn_{sid}_d_{period}.csv"
        )
        logger.info(f"Fetching SMN daily [{station}] {period}: {url}")
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()
            df = _parse_smn_csv(r.text, station)
            frames.append(df)
        except httpx.HTTPStatusError as e:
            logger.warning(f"SMN {station} {period}: HTTP {e.response.status_code} — skipping")
        except Exception as e:
            logger.warning(f"SMN {station} {period}: {e} — skipping")

    if not frames:
        logger.error(f"No SMN data retrieved for {station}")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")

    # Filter to requested window
    df = df[(df["date"] >= pd.Timestamp(start_date)) &
            (df["date"] <= pd.Timestamp(end_date))].reset_index(drop=True)

    logger.success(f"SMN {station}: {len(df)} daily records ({start_date} → {end_date})")
    return df


def fetch_smn_all_hospitals(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch SMN daily data for all hospital cantons and merge into one DataFrame.

    Returns:
        DataFrame with columns: date, canton, station_id, + all weather features
    """
    frames = []
    for canton, station in HOSPITAL_STATIONS.items():
        df = fetch_smn_daily(station, start_date, end_date)
        if not df.empty:
            df["canton"]     = canton
            df["station_id"] = station
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values(["canton", "date"])


def _parse_smn_csv(text: str, station_id: str) -> pd.DataFrame:
    """
    Parse SMN CSV file.
    Format: semicolon-separated, first column = date (YYYYMMDD or ISO),
            parameter columns named by MeteoSwiss parameter codes.
    """
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=";",
            na_values=["-", "", " "],
            low_memory=False,
        )
    except Exception as e:
        logger.error(f"CSV parse error for {station_id}: {e}")
        return pd.DataFrame()

    # Normalise date column (MeteoSwiss uses 'date' or 'time')
    date_col = next((c for c in df.columns if c.lower() in ["date", "time", "datum"]), None)
    if date_col is None:
        logger.warning(f"No date column found in SMN file for {station_id}")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])

    # Rename parameter columns
    rename = {}
    for smn_code, ml_name in SMN_PARAMS.items():
        if smn_code in df.columns:
            rename[smn_code] = ml_name

    df = df.rename(columns=rename)

    # Keep only renamed + date
    keep = ["date"] + [v for v in SMN_PARAMS.values() if v in df.columns]
    return df[keep].copy()


# ═══════════════════════════════════════════════════════════════════
# 2.  POLLEN — Stations pollen
# ═══════════════════════════════════════════════════════════════════

# Pollen stations closest to hospital cantons
POLLEN_STATIONS: dict[str, str] = {
    "BE": "BER",
    "ZH": "ZUE",
    "GE": "GEN",
    "VD": "LAU",
    "BS": "BAS",
    "SG": "STG",
    "TI": "LUG",
}

# Pollen types most correlated with ED respiratory admissions
POLLEN_TYPES = [
    "Betula",      # Bouleau   — allergies respiratoires majeures (mars-avril)
    "Poaceae",     # Graminées — asthme estival (mai-juillet)
    "Fraxinus",    # Frêne     — printemps précoce
    "Alnus",       # Aulne     — fin hiver
]


def fetch_pollen_daily(
    station_id: str,
    start_date: date,
    end_date: date,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch daily pollen concentration for one station.

    Returns:
        DataFrame with date + one column per pollen type (grains/m³)
    """
    station = station_id.upper()
    sid     = station.lower()

    url = (
        f"{_DATA_BASE}/{POLLEN_COLLECTION}/{sid}/"
        f"ogd-pollen_{sid}_d_historical.csv"
    )
    logger.info(f"Fetching pollen [{station}]: {url}")

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()

        df = pd.read_csv(io.StringIO(r.text), sep=";", na_values=["-", ""])
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")

        # Keep pollen type columns
        pollen_cols = [c for c in df.columns if any(p in c for p in POLLEN_TYPES)]
        keep = ["date"] + pollen_cols
        df = df[[c for c in keep if c in df.columns]].copy()

        # Rename to standard names
        df.columns = [
            c if c == "date"
            else f"pollen_{c.split('_')[0].lower()}_grains_m3"
            for c in df.columns
        ]

        # Add aggregate pollen index (0-5 scale)
        pollen_value_cols = [c for c in df.columns if c.startswith("pollen_")]
        if pollen_value_cols:
            df["pollen_index"] = df[pollen_value_cols].sum(axis=1).clip(0, 5000) / 1000

        df = df[
            (df["date"] >= pd.Timestamp(start_date)) &
            (df["date"] <= pd.Timestamp(end_date))
        ].reset_index(drop=True)

        logger.success(f"Pollen {station}: {len(df)} daily records")
        return df

    except httpx.HTTPStatusError as e:
        logger.warning(f"Pollen {station}: HTTP {e.response.status_code}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Pollen {station}: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# 3.  NBCN — Normales climatologiques (anomalies thermiques)
# ═══════════════════════════════════════════════════════════════════

def fetch_climate_normals(
    station_id: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch homogeneous climate series (NBCN) to compute historical normals.
    Used to derive temperature anomaly = observed - historical_mean.

    Returns:
        DataFrame with month (1-12) + temp_normal_c per station.
    """
    station = station_id.upper()
    sid     = station.lower()

    url = (
        f"{_DATA_BASE}/{NBCN_COLLECTION}/{sid}/"
        f"ogd-nbcn_{sid}_d_historical.csv"
    )
    logger.info(f"Fetching NBCN normals [{station}]: {url}")

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()

        df = pd.read_csv(io.StringIO(r.text), sep=";", na_values=["-", ""])
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        temp_col = next((c for c in df.columns if "tre200" in c.lower()), None)

        if not date_col or not temp_col:
            return pd.DataFrame()

        df["date"]  = pd.to_datetime(df[date_col], errors="coerce")
        df["temp_c"] = pd.to_numeric(df[temp_col], errors="coerce")
        df = df.dropna(subset=["date", "temp_c"])
        df["month"] = df["date"].dt.month

        # Monthly normals over 1981-2010 reference period
        normals = (
            df[df["date"].dt.year.between(1981, 2010)]
            .groupby("month")["temp_c"]
            .mean()
            .reset_index()
            .rename(columns={"temp_c": "temp_normal_c"})
        )
        normals["station_id"] = station
        logger.success(f"NBCN {station}: computed monthly normals")
        return normals

    except Exception as e:
        logger.warning(f"NBCN {station}: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# 4.  FEATURE ENGINEERING — pipeline principal
# ═══════════════════════════════════════════════════════════════════

def build_weather_features(
    smn_df: pd.DataFrame,
    pollen_df: Optional[pd.DataFrame] = None,
    normals_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build all weather ML features from raw MétéoSuisse data.

    Features engineered:
    ─────────────────────────────────────────────────────
    Température
      temp_mean_c, temp_min_c, temp_max_c
      temp_cold        → 1 si temp_mean < 5°C  (hypothermie, chutes)
      temp_freezing    → 1 si temp_min  < 0°C  (verglas)
      temp_hot         → 1 si temp_max  > 30°C (canicule)
      temp_anomaly_c   → écart vs normale mensuelle (NBCN)
      cold_streak      → nb jours consécutifs temp_mean < 5°C

    Précipitations
      precipitation_mm
      is_precipitation → 1 si > 0.5mm
      heavy_rain       → 1 si > 10mm  (accidents, chutes)
      is_snow          → 1 si précip + temp_min < 1°C (verglas)

    Vent
      wind_speed_ms, wind_gust_ms
      strong_wind      → 1 si rafale > 15 m/s

    Humidité & Ensoleillement
      humidity_pct
      high_humidity    → 1 si > 80% (infections respiratoires)
      sunshine_min
      low_sunshine     → 1 si < 60 min (dépression saisonnière)

    Pollen
      pollen_index     → index agrégé 0-5 (asthme, allergies → urgences)

    Interactions (features croisées)
      cold_x_humid     → froid + humide → infections respi
      rain_x_wind      → tempête → traumatismes
    ─────────────────────────────────────────────────────

    Args:
        smn_df:      Output de fetch_smn_daily() ou fetch_smn_all_hospitals()
        pollen_df:   Output de fetch_pollen_daily() — optionnel
        normals_df:  Output de fetch_climate_normals() — optionnel

    Returns:
        DataFrame enrichi avec toutes les features météo.
    """
    if smn_df.empty:
        logger.warning("Empty SMN DataFrame — returning empty features")
        return pd.DataFrame()

    df = smn_df.copy()

    # ── Température ──────────────────────────────────────────────
    if "temp_mean_c" in df.columns:
        df["temp_cold"]    = (df["temp_mean_c"] < 5).astype(int)
        df["temp_hot"]     = (df["temp_max_c"].fillna(df["temp_mean_c"]) > 30).astype(int)
        df["temp_range_c"] = (
            df.get("temp_max_c", df["temp_mean_c"]) -
            df.get("temp_min_c", df["temp_mean_c"])
        )

    if "temp_min_c" in df.columns:
        df["temp_freezing"] = (df["temp_min_c"] < 0).astype(int)

    # Température anomalie vs normale mensuelle
    if normals_df is not None and not normals_df.empty and "temp_mean_c" in df.columns:
        df["month"] = df["date"].dt.month
        df = df.merge(normals_df[["month", "temp_normal_c"]], on="month", how="left")
        df["temp_anomaly_c"] = df["temp_mean_c"] - df["temp_normal_c"]
        df["temp_above_normal"] = (df["temp_anomaly_c"] > 3).astype(int)
        df["temp_below_normal"] = (df["temp_anomaly_c"] < -3).astype(int)

    # Cold streak — nb jours consécutifs froids (effet différé sur urgences)
    if "temp_cold" in df.columns:
        group_col = "canton" if "canton" in df.columns else None
        if group_col:
            df["cold_streak"] = (
                df.groupby(group_col)["temp_cold"]
                .transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
            )
        else:
            df["cold_streak"] = (
                df["temp_cold"] *
                (df["temp_cold"].groupby((df["temp_cold"] != df["temp_cold"].shift()).cumsum()).cumcount() + 1)
            )

    # ── Précipitations ───────────────────────────────────────────
    if "precipitation_mm" in df.columns:
        df["is_precipitation"] = (df["precipitation_mm"] > 0.5).astype(int)
        df["heavy_rain"]       = (df["precipitation_mm"] > 10).astype(int)
        df["is_snow"] = (
            (df["precipitation_mm"] > 0.5) &
            (df.get("temp_min_c", pd.Series(10, index=df.index)) < 1)
        ).astype(int)

    # ── Vent ─────────────────────────────────────────────────────
    if "wind_gust_ms" in df.columns:
        df["strong_wind"] = (df["wind_gust_ms"] > 15).astype(int)
    elif "wind_speed_ms" in df.columns:
        df["strong_wind"] = (df["wind_speed_ms"] > 10).astype(int)

    # ── Humidité & Ensoleillement ─────────────────────────────────
    if "humidity_pct" in df.columns:
        df["high_humidity"] = (df["humidity_pct"] > 80).astype(int)

    if "sunshine_min" in df.columns:
        df["low_sunshine"] = (df["sunshine_min"] < 60).astype(int)

    # ── Pollen ───────────────────────────────────────────────────
    if pollen_df is not None and not pollen_df.empty:
        merge_on = ["date", "canton"] if "canton" in pollen_df.columns else ["date"]
        df = df.merge(
            pollen_df[merge_on + ["pollen_index"]],
            on=merge_on,
            how="left",
        )
        df["pollen_index"] = df["pollen_index"].fillna(0)
        df["high_pollen"]  = (df["pollen_index"] > 2.0).astype(int)
    else:
        df["pollen_index"] = 0
        df["high_pollen"]  = 0

    # ── Interactions ─────────────────────────────────────────────
    if "temp_cold" in df.columns and "high_humidity" in df.columns:
        df["cold_x_humid"] = df["temp_cold"] * df["high_humidity"]

    if "is_precipitation" in df.columns and "strong_wind" in df.columns:
        df["rain_x_wind"] = df["is_precipitation"] * df.get("strong_wind", 0)

    logger.success(f"Weather features built: {df.shape[1]} columns, {len(df)} rows")
    return df


def get_feature_names() -> list[str]:
    """Return the list of all weather ML feature names (for XGBoost column selection)."""
    return [
        # Raw measurements
        "temp_mean_c", "temp_min_c", "temp_max_c",
        "precipitation_mm", "sunshine_min", "humidity_pct",
        "wind_speed_ms", "wind_gust_ms", "pressure_hpa",
        # Engineered binary flags
        "temp_cold", "temp_freezing", "temp_hot",
        "is_precipitation", "heavy_rain", "is_snow", "strong_wind",
        "high_humidity", "low_sunshine",
        # Anomalie et streak
        "temp_anomaly_c", "temp_above_normal", "temp_below_normal",
        "cold_streak", "temp_range_c",
        # Pollen
        "pollen_index", "high_pollen",
        # Interactions
        "cold_x_humid", "rain_x_wind",
    ]


# ═══════════════════════════════════════════════════════════════════
# 5.  PIPELINE COMPLET — fonction d'entrée principale
# ═══════════════════════════════════════════════════════════════════

def build_meteo_dataset(
    start_date: date,
    end_date: date,
    include_pollen: bool = True,
    include_normals: bool = True,
) -> pd.DataFrame:
    """
    Pipeline complet : fetch + engineer toutes les features météo
    pour tous les cantons hospitaliers.

    Usage:
        from src.ingestion.meteo_swiss import build_meteo_dataset
        meteo = build_meteo_dataset(date(2021,1,1), date(2024,12,31))
        # Merge avec SpiGes daily sur ['date', 'canton']

    Returns:
        DataFrame avec date + canton + toutes les features météo.
        Prêt pour merge avec spiges_daily_aggregated.
    """
    logger.info(f"Building meteo dataset {start_date} → {end_date}")

    # 1. SMN pour tous les hôpitaux
    smn = fetch_smn_all_hospitals(start_date, end_date)
    if smn.empty:
        logger.error("No SMN data — aborting")
        return pd.DataFrame()

    # 2. Pollen (optionnel)
    pollen_frames = []
    if include_pollen:
        for canton, station in POLLEN_STATIONS.items():
            p = fetch_pollen_daily(station, start_date, end_date)
            if not p.empty:
                p["canton"] = canton
                pollen_frames.append(p)
    pollen = pd.concat(pollen_frames) if pollen_frames else None

    # 3. Normales climatologiques (optionnel)
    normals_frames = []
    if include_normals:
        for canton, station in HOSPITAL_STATIONS.items():
            n = fetch_climate_normals(station)
            if not n.empty:
                n["canton"] = canton
                normals_frames.append(n)
    normals = pd.concat(normals_frames) if normals_frames else None

    # 4. Feature engineering
    result = build_weather_features(smn, pollen, normals)

    logger.success(
        f"Meteo dataset ready: {len(result):,} rows × {result.shape[1]} cols "
        f"| {result['canton'].nunique()} cantons"
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# CLI — test rapide
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    end   = date.today()
    start = end - timedelta(days=30)

    if len(sys.argv) > 1:
        canton = sys.argv[1].upper()
        station = HOSPITAL_STATIONS.get(canton, "BER")
        df = fetch_smn_daily(station, start, end)
        df = build_weather_features(df)
        print(f"\n{canton} — {station} — derniers 30 jours")
        print(df[["date"] + [c for c in get_feature_names() if c in df.columns]].tail(10).to_string())
    else:
        print("Usage: python meteo_swiss.py [CANTON]")
        print("       python meteo_swiss.py BE")
        print(f"\nStation mapping: {HOSPITAL_STATIONS}")
        print(f"\nFeatures disponibles ({len(get_feature_names())}):")
        for f in get_feature_names():
            print(f"  {f}")