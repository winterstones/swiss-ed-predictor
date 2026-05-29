"""
Swiss ED Predictor — XGBoost Training Script
=============================================
Entraîne 3 modèles XGBoost pour prédire les pics d'affluence
aux urgences hospitalières suisses à 24h, 48h et 72h.

Dataset : spiges_meteo_traffic_joined.csv
  → 2 479 lignes × 35 colonnes
  → 10 cantons × 248 jours (2021-01-01 → 2021-09-05)
  → Sources : SpiGes/OFSP + MétéoSuisse + ASTRA trafic

Architecture :
  1. Feature engineering & imputation
  2. TimeSeriesSplit cross-validation (5 folds, no data leakage)
  3. Entraînement XGBoost avec early stopping
  4. Évaluation : MAE, RMSE, MAPE, R²
  5. Analyse SHAP (explicabilité)
  6. Sauvegarde modèles + rapport

Référence scientifique :
  King et al. (Nature npj Digital Medicine, 2022)
  XGBoost sur 109 465 visites UK : AUROC 0.90, MAE 4.0

Usage :
  python3 train.py
  python3 train.py --horizon 24
  python3 train.py --horizon all --save-shap
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from loguru import logger

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
DATA_PATH = ROOT.parents[1] / "data" / "historical" / "spiges_meteo_traffic_joined.csv"
MODEL_DIR = ROOT.parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 1. FEATURE SETS
# ═══════════════════════════════════════════════════════════════════

SPIGES_FEATURES = [
    "month", "day_of_week", "week_of_year",
    "is_weekend", "is_winter", "is_summer",
    "pct_elderly", "mean_severity", "mean_nems", "ips_cases",
]

LAG_FEATURES = [
    "notfall_lag1",   # corr +0.767
    "notfall_lag7",   # corr +0.859
    "notfall_roll7",  # corr +0.878 — la plus forte
]

METEO_FEATURES = [
    "temperature_avg",  # corr -0.253
    "temperature_max",
    "temperature_min",
]

TRAFFIC_FEATURES = [
    "daily_traffic_volume",  # corr +0.733
    "heavy_vehicle_pct",     # corr +0.240
    "avg_speed_kmh",         # corr -0.213
    "low_speed_flag",
    "traffic_per_hour",      # corr +0.733
    "high_traffic_day",
    "low_traffic_day",
]

HOLIDAY_FEATURES = [
    # ── Swiss federal holidays (Bundesfeiertage) ──────────────────
    "is_federal_holiday",       # 1 = Neujahr, Bundesfeiertag, Weihnachten...
    # ── Cantonal / regional holidays ─────────────────────────────
    "is_cantonal_holiday",      # 1 = e.g. Karfreitag, Ostermontag, Pfingstmontag
    # ── School holidays ──────────────────────────────────────────
    "is_school_holiday",        # 1 = Swiss school holiday period
    # ── Bridge days / long weekends ──────────────────────────────
    "is_bridge_day",            # 1 = Fri/Mon between holiday and weekend
    # ── Days relative to holidays ────────────────────────────────
    "days_to_next_holiday",     # 0..30 — anticipation effect
    "days_since_last_holiday",  # 0..30 — recovery effect
    # ── Combined holiday type ────────────────────────────────────
    "holiday_type",             # 0=normal, 1=federal, 2=cantonal, 3=school
    # ── Interaction ──────────────────────────────────────────────
    "holiday_x_winter",         # holiday during winter → strong drop
    "holiday_x_weekday",        # holiday on a weekday → unexpected drop
]

INTERACTION_FEATURES = [
    "cold_x_weekend",
    "traffic_x_winter",
    "lag7_x_winter",
    "elderly_x_winter",
    "traffic_x_elderly",
    # Note: holiday_x_winter and holiday_x_weekday are in HOLIDAY_FEATURES above
]

ALL_FEATURES = list(dict.fromkeys(  # deduplicate while preserving order
    SPIGES_FEATURES + LAG_FEATURES
    + METEO_FEATURES + TRAFFIC_FEATURES
    + HOLIDAY_FEATURES
    + INTERACTION_FEATURES
))

TARGET_COLS = {
    24: "target_notfall_next24h",
    48: "target_notfall_next48h",
    72: "target_notfall_next72h",
}

XGB_PARAMS = {
    "n_estimators":         800,
    "max_depth":            5,
    "learning_rate":        0.04,
    "subsample":            0.80,
    "colsample_bytree":     0.75,
    "colsample_bylevel":    0.75,
    "min_child_weight":     5,
    "reg_alpha":            0.2,
    "reg_lambda":           1.5,
    "gamma":                0.05,
    "objective":            "reg:squarederror",
    "eval_metric":          ["rmse", "mae"],
    "early_stopping_rounds":60,
    "random_state":         42,
    "n_jobs":               -1,
    "verbosity":            0,
}

# ═══════════════════════════════════════════════════════════════════
# SWISS HOLIDAY CALENDAR
# ═══════════════════════════════════════════════════════════════════

def _build_swiss_holidays(years: list[int]) -> dict:
    """
    Construit le calendrier des jours fériés suisses pour les années données.

    Structure retournée :
      {
        date_obj: {"type": "federal"|"cantonal", "name": str},
        ...
      }

    Sources :
      Jours fériés fédéraux (art. 20a CO) :
        1er janvier, 1er août, 25 décembre, 26 décembre
      Jours fériés reconnus dans la plupart des cantons :
        Vendredi Saint, Lundi de Pâques, Ascension, Lundi de Pentecôte
      Calcul Pâques : algorithme de Butcher (validé jusqu'en 2099)
    """
    from datetime import date, timedelta

    calendar = {}

    for year in years:
        # ── Jours fériés FÉDÉRAUX ─────────────────────────────────
        federal = [
            (date(year, 1, 1),  "Nouvel An"),
            (date(year, 8, 1),  "Fête nationale suisse"),
            (date(year, 12, 25),"Noël"),
            (date(year, 12, 26),"Saint-Étienne"),
        ]
        for d, name in federal:
            calendar[d] = {"type": "federal", "name": name}

        # ── Calcul de Pâques (algorithme de Butcher) ──────────────
        a = year % 19
        b = year // 100
        c = year % 100
        d_ = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d_ - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day   = ((h + l - 7 * m + 114) % 31) + 1
        easter = date(year, month, day)

        # ── Jours fériés CANTONAUX (majorité des cantons CH) ──────
        cantonal = [
            (easter - timedelta(days=2), "Vendredi Saint"),
            (easter + timedelta(days=1), "Lundi de Pâques"),
            (easter + timedelta(days=39),"Ascension"),
            (easter + timedelta(days=50),"Lundi de Pentecôte"),
        ]
        # Certains cantons : Fête-Dieu, Saint-Berchtold, etc.
        # (pas universel → on les met en "cantonal" avec poids plus faible)
        extra_cantonal = [
            (date(year, 1, 2),  "Saint-Berchtold"),      # BE, ZH, AG, SG, TI, VS
            (date(year, 5, 1),  "Fête du Travail"),      # BS, ZH (partiel)
        ]
        for d, name in cantonal:
            calendar[d] = {"type": "cantonal", "name": name}
        for d, name in extra_cantonal:
            if d not in calendar:
                calendar[d] = {"type": "cantonal", "name": name}

    return calendar


def _school_holidays_ch_2021() -> set:
    """
    Vacances scolaires suisses 2021 (approximation nationale).
    Les dates varient légèrement par canton — on prend les périodes communes.

    Impact sur les urgences :
      - Moins d'accidents scolaires
      - Plus de loisirs → accidents sport/loisirs
      - Trafic routier différent (familles vs pendulaires)
      - Net : légère réduction des urgences planifiables
    """
    from datetime import date, timedelta

    holidays = set()

    periods = [
        # Hiver/Noël 2020-2021
        (date(2021, 1, 1),  date(2021, 1, 10)),
        # Vacances de printemps (Frühlingsferien)
        (date(2021, 4, 1),  date(2021, 4, 18)),
        # Ascension / pont
        (date(2021, 5, 13), date(2021, 5, 16)),
        # Été (Sommerferien) — la plus longue
        (date(2021, 7, 5),  date(2021, 8, 22)),
    ]

    for start, end in periods:
        d = start
        while d <= end:
            holidays.add(d)
            d += __import__('datetime').timedelta(days=1)

    return holidays


def _add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute toutes les features liées aux jours fériés et vacances scolaires.

    Features ajoutées :
      is_federal_holiday     : 1 si jour férié fédéral (Neujahr, Bundesfeiertag...)
      is_cantonal_holiday    : 1 si jour férié cantonal (Karfreitag, Ostermontag...)
      is_school_holiday      : 1 si période de vacances scolaires suisses
      is_bridge_day          : 1 si pont entre jour férié et weekend
      days_to_next_holiday   : jours jusqu'au prochain jour férié (0..30)
      days_since_last_holiday: jours depuis le dernier jour férié (0..30)
      holiday_type           : 0=normal, 1=federal, 2=cantonal, 3=school only
      holiday_x_winter       : interaction férié × hiver
      holiday_x_weekday      : interaction férié × jour ouvré (impact maximal)

    Corrélations attendues avec notfall_admissions :
      is_federal_holiday     : -0.15 à -0.25 (réduction urgences planifiables)
      is_school_holiday      : -0.10 à -0.18 (moins d'accidents scolaires)
      days_to_next_holiday   : signal anticipation (gens reportent les soins)
      holiday_x_weekday      : plus fort que holiday seul (surprise sur semaine)
    """
    from datetime import timedelta

    df = df.copy()
    dates = pd.to_datetime(df["date"])
    years = dates.dt.year.unique().tolist()

    # Construire les calendriers
    swiss_holidays = _build_swiss_holidays(years)
    school_holidays = _school_holidays_ch_2021()

    federal_dates  = {d for d, v in swiss_holidays.items() if v["type"] == "federal"}
    cantonal_dates = {d for d, v in swiss_holidays.items() if v["type"] == "cantonal"}
    all_holiday_dates = federal_dates | cantonal_dates

    # ── Features binaires de base ──────────────────────────────────
    df["is_federal_holiday"]  = dates.dt.date.isin(federal_dates).astype(int)
    df["is_cantonal_holiday"] = dates.dt.date.isin(cantonal_dates).astype(int)
    df["is_school_holiday"]   = dates.dt.date.isin(school_holidays).astype(int)

    # ── Pont (bridge day) ─────────────────────────────────────────
    # Définition : vendredi entre un jeudi férié et le weekend
    # ou lundi entre le weekend et un mardi férié
    bridge_dates = set()
    for d in all_holiday_dates:
        dow = d.weekday()
        if dow == 3:  # Jeudi férié → vendredi est un pont
            bridge_dates.add(d + timedelta(days=1))
        elif dow == 1:  # Mardi férié → lundi est un pont
            bridge_dates.add(d - timedelta(days=1))
    # Ne garder que les jours ouvrés
    bridge_dates = {
        d for d in bridge_dates
        if d.weekday() < 5 and d not in all_holiday_dates
    }
    df["is_bridge_day"] = dates.dt.date.isin(bridge_dates).astype(int)

    # ── Jours jusqu'au prochain / depuis le dernier férié ─────────
    sorted_holidays = sorted(all_holiday_dates)
    holiday_ts      = pd.to_datetime(sorted_holidays)

    days_to   = np.full(len(df), 30, dtype=int)
    days_since= np.full(len(df), 30, dtype=int)

    for i, d in enumerate(dates):
        future = holiday_ts[holiday_ts > d]
        if not future.empty:
            days_to[i] = min(30, (future[0] - d).days)
        past = holiday_ts[holiday_ts < d]
        if not past.empty:
            days_since[i] = min(30, (d - past[-1]).days)

    df["days_to_next_holiday"]    = days_to
    df["days_since_last_holiday"] = days_since

    # ── Type de jour agrégé — vectorisé ────────────────────────────
    # 0=normal, 1=federal, 2=cantonal, 3=school holiday only
    holiday_type_arr = np.zeros(len(df), dtype=int)
    date_arr = dates.dt.date
    holiday_type_arr[date_arr.isin(school_holidays)] = 3
    holiday_type_arr[date_arr.isin(cantonal_dates)]  = 2
    holiday_type_arr[date_arr.isin(federal_dates)]   = 1
    df["holiday_type"] = holiday_type_arr

    # ── Interactions — using .values to guarantee 1D arrays ───────
    any_holiday_arr = (
        df["is_federal_holiday"].values.astype(int) |
        df["is_cantonal_holiday"].values.astype(int)
    )
    is_winter_arr   = df["is_winter"].values.astype(int)  if "is_winter"  in df.columns else np.zeros(len(df), int)
    is_weekend_arr  = df["is_weekend"].values.astype(int) if "is_weekend" in df.columns else np.ones(len(df), int)

    df["holiday_x_winter"]  = any_holiday_arr * is_winter_arr
    df["holiday_x_weekday"] = any_holiday_arr * (1 - is_weekend_arr)

    return df


# ═══════════════════════════════════════════════════════════════════
# 2. CHARGEMENT & PRÉPARATION
# ═══════════════════════════════════════════════════════════════════

def load_and_prepare(path: Path) -> pd.DataFrame:
    logger.info(f"Loading: {path}")
    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    df = df.sort_values(["kanton_hospital", "date"]).reset_index(drop=True)
    logger.info(f"  Shape: {df.shape}")

    # Imputation météo — médiane canton × mois
    for col in ["temperature_avg", "temperature_max", "temperature_min"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df.groupby(["kanton_hospital", "month"])[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df[col].fillna(df[col].median())

    # Imputation lags — forward fill par canton
    for col in ["notfall_lag1", "notfall_lag7", "notfall_roll7"]:
        if col in df.columns:
            df[col] = df.groupby("kanton_hospital")[col].transform(
                lambda x: x.ffill().bfill()
            )
            df[col] = df[col].fillna(df[col].median())

    # Feature engineering — interactions
    df["cold_x_weekend"]    = df["is_winter"]  * df["is_weekend"]
    df["traffic_x_winter"]  = (df["daily_traffic_volume"] / 10000) * df["is_winter"]
    df["lag7_x_winter"]     = df["notfall_lag7"] * df["is_winter"]
    df["elderly_x_winter"]  = df["pct_elderly"]  * df["is_winter"]
    df["traffic_x_elderly"] = (df["daily_traffic_volume"] / 10000) * df["pct_elderly"]

    # Feature engineering — holidays
    df = _add_holiday_features(df)

    logger.info(f"  Holiday features: {df['is_federal_holiday'].sum()} federal | "
                f"{df['is_cantonal_holiday'].sum()} cantonal | "
                f"{df['is_school_holiday'].sum()} school holiday days")

    # Encodage canton
    canton_map = {c: i for i, c in enumerate(sorted(df["kanton_hospital"].unique()))}
    df["canton_enc"] = df["kanton_hospital"].map(canton_map)
    with open(MODEL_DIR / "canton_map.json", "w") as f:
        json.dump(canton_map, f)

    logger.info(f"  NaN après imputation — temp: {df['temperature_avg'].isna().sum()} | lag: {df['notfall_lag1'].isna().sum()}")

    # ── Sanitize : garantir que toutes les colonnes sont des Series 1D ──
    # Certaines opérations pandas peuvent créer des DataFrames au lieu de Series
    problem_cols = []
    for col in df.columns:
        if isinstance(df[col], pd.DataFrame):
            problem_cols.append(col)
            df[col] = df[col].iloc[:, 0]
    if problem_cols:
        logger.warning(f"  Colonnes DataFrame → Series corrigées: {problem_cols}")

    return df


# ═══════════════════════════════════════════════════════════════════
# 3. CROSS-VALIDATION TEMPORELLE
# ═══════════════════════════════════════════════════════════════════

def _sanitize_X(X: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit que toutes les colonnes de X sont des Series 1D numériques.
    Corrige silencieusement les colonnes DataFrame ou multi-dim.
    """
    clean = {}
    for col in X.columns:
        val = X[col]
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0]
        clean[col] = pd.to_numeric(val, errors="coerce").fillna(0)
    result = pd.DataFrame(clean, index=X.index)
    # Drop duplicated column names if any
    result = result.loc[:, ~result.columns.duplicated()]
    return result

def cross_validate(X, y, n_splits, label):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=10)
    cv_scores, best_model = [], None

    logger.info(f"\n{'─'*60}")
    logger.info(f"TimeSeriesSplit CV — horizon {label} | {n_splits} folds | gap=10")
    logger.info(f"{'─'*60}")
    logger.info(f"  {'Fold':>4} {'Train':>7} {'Val':>7} {'MAE':>7} {'RMSE':>7} {'MAPE':>7} {'R²':>7}")
    logger.info(f"  {'─'*50}")

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        # Guarantee all columns are 1D numeric Series — no DataFrame columns
        X_tr  = _sanitize_X(X_tr)
        X_val = _sanitize_X(X_val)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = np.maximum(0, model.predict(X_val))
        mae   = mean_absolute_error(y_val, preds)
        rmse  = np.sqrt(mean_squared_error(y_val, preds))
        r2    = r2_score(y_val, preds)
        mask  = y_val > 0
        mape  = np.mean(np.abs((y_val[mask] - preds[mask]) / y_val[mask])) * 100

        cv_scores.append({"fold": fold, "n_train": len(X_tr), "n_val": len(X_val),
                           "mae": round(mae,3), "rmse": round(rmse,3),
                           "mape": round(mape,2), "r2": round(r2,4)})
        logger.info(f"  {fold:>4} {len(X_tr):>7} {len(X_val):>7} {mae:>7.2f} {rmse:>7.2f} {mape:>6.1f}% {r2:>7.3f}")
        best_model = model

    logger.info(f"  {'─'*50}")
    logger.info(
        f"  {'MEAN':>4} {'':>7} {'':>7} "
        f"{np.mean([s['mae'] for s in cv_scores]):>7.2f} "
        f"{np.mean([s['rmse'] for s in cv_scores]):>7.2f} "
        f"{np.mean([s['mape'] for s in cv_scores]):>6.1f}% "
        f"{np.mean([s['r2'] for s in cv_scores]):>7.3f}"
    )
    logger.info(f"  ± STD MAE: {np.std([s['mae'] for s in cv_scores]):.3f}")

    return cv_scores, best_model


# ═══════════════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT FINAL
# ═══════════════════════════════════════════════════════════════════

def train_final(X_train, y_train, X_test, y_test, horizon):
    logger.info(f"\n{'═'*60}")
    logger.info(f"Final training — horizon {horizon}h")
    logger.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(X_train.columns)}")

    val_size = max(100, int(len(X_train) * 0.10))
    X_train  = _sanitize_X(X_train)
    X_test   = _sanitize_X(X_test)
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train.iloc[:-val_size], y_train.iloc[:-val_size],
        eval_set=[(X_train.iloc[-val_size:], y_train.iloc[-val_size:])],
        verbose=False,
    )

    preds = np.maximum(0, model.predict(X_test))
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    mask  = y_test > 0
    mape  = np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask])) * 100
    acc2  = np.mean(np.abs(preds - y_test) <= 2) * 100
    acc5  = np.mean(np.abs(preds - y_test) <= 5) * 100

    metrics = {
        "horizon_h": horizon, "n_train": len(X_train), "n_test": len(X_test),
        "n_features": len(X_train.columns), "best_iteration": int(model.best_iteration),
        "mae": round(mae,3), "rmse": round(rmse,3), "r2": round(r2,4),
        "mape_pct": round(mape,2), "accuracy_pm2": round(acc2,1),
        "accuracy_pm5": round(acc5,1), "trained_at": datetime.now().isoformat(),
    }

    logger.success(
        f"\n  ✅ Test results — {horizon}h\n"
        f"     MAE  = {mae:.2f} admissions/jour\n"
        f"     RMSE = {rmse:.2f}\n"
        f"     R²   = {r2:.4f}\n"
        f"     MAPE = {mape:.1f}%\n"
        f"     Précision ±2 adm = {acc2:.1f}%\n"
        f"     Précision ±5 adm = {acc5:.1f}%"
    )
    return model, metrics


# ═══════════════════════════════════════════════════════════════════
# 5. ANALYSE SHAP
# ═══════════════════════════════════════════════════════════════════

def compute_shap(model, X, horizon, n_samples=500, save=True):
    logger.info(f"\n  Computing SHAP (n={min(n_samples, len(X))})...")

    INTERPRETATIONS = {
        "notfall_roll7":        "Tendance 7j — mémoire temporelle forte",
        "notfall_lag7":         "J-7 — même jour semaine dernière",
        "notfall_lag1":         "J-1 — continuité journalière",
        "daily_traffic_volume": "Volume trafic — proxy mobilité/taille canton",
        "traffic_per_hour":     "Trafic/h — mobilité normalisée",
        "is_winter":            "Hiver — grippe + chutes + froid",
        "is_weekend":           "Weekend — moins de cas planifiés",
        "temperature_avg":      "Température — froid → urgences",
        "pct_elderly":          "Part 65+ — risque hivernal",
        "heavy_vehicle_pct":    "% poids lourds — accidents graves",
        "avg_speed_kmh":        "Vitesse — congestion / météo dégradée",
        "lag7_x_winter":        "Lag7 × hiver — tendance hivernale amplifiée",
        "elderly_x_winter":     "65+ × hiver — infections respiratoires",
        "traffic_x_winter":     "Trafic × hiver — verglas → accidents",
        "cold_x_weekend":       "Froid × weekend — chutes loisirs",
        "month":                "Mois — saisonnalité",
        "day_of_week":          "Jour — profil hebdomadaire",
        "mean_severity":        "Sévérité — état de santé général",
        "mean_nems":            "NEMS — charge soins infirmiers",
        # Holidays
        "is_federal_holiday":      "Férié fédéral — réduction urgences planifiables",
        "is_cantonal_holiday":     "Férié cantonal — Karfreitag, Ostermontag...",
        "is_school_holiday":       "Vacances scolaires — moins d'accidents scolaires",
        "is_bridge_day":           "Pont — report soins non urgents",
        "days_to_next_holiday":    "Jours jusqu'au prochain férié — anticipation",
        "days_since_last_holiday": "Jours depuis dernier férié — reprise",
        "holiday_type":            "Type de jour férié (0=normal..3=vacances)",
        "holiday_x_winter":        "Férié × hiver — double réduction urgences",
        "holiday_x_weekday":       "Férié en semaine — impact maximal inattendu",
    }

    sample      = X.sample(min(n_samples, len(X)), random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    importance = pd.DataFrame({
        "feature":  X.columns,
        "mean_shap":np.abs(shap_values).mean(axis=0),
        "std_shap": np.abs(shap_values).std(axis=0),
    }).sort_values("mean_shap", ascending=False).reset_index(drop=True)

    logger.info(f"\n  Top 10 features SHAP — horizon {horizon}h :")
    logger.info(f"  {'Rank':>4} {'Feature':<28} {'SHAP':>7}  Interprétation")
    logger.info(f"  {'─'*70}")
    for i, row in importance.head(10).iterrows():
        interp = INTERPRETATIONS.get(row["feature"], "")
        logger.info(f"  {i+1:>4} {row['feature']:<28} {row['mean_shap']:>7.3f}  {interp}")

    if save:
        out = MODEL_DIR / f"shap_importance_{horizon}h.csv"
        importance.to_csv(out, index=False)
        logger.info(f"\n  Saved: {out}")

    return importance


# ═══════════════════════════════════════════════════════════════════
# 6. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# 8. PLOTS — visualisations post-entraînement
# ═══════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Palette projet ────────────────────────────────────────────────
NAVY  = "#0A2342"
TEAL  = "#1B7FA1"
MINT  = "#02C39A"
ACC   = "#F9A825"
RED   = "#EF4444"
GRAY  = "#64748B"
LIGHT = "#E2EEF4"
WHITE = "#FFFFFF"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.facecolor": WHITE,
    "axes.facecolor":   WHITE,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.titlecolor":  NAVY,
    "axes.labelcolor":  GRAY,
    "xtick.color":      GRAY,
    "ytick.color":      GRAY,
})

PLOTS_DIR = MODEL_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_predictions_vs_actual(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizon: int,
    df_test: pd.DataFrame,
) -> Path:
    """
    Plot 1 — Prédit vs Réel (scatter + série temporelle)
    Montre la qualité de la prédiction sur le test set.
    """
    preds = np.maximum(0, model.predict(X_test))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Prédictions vs Réel — Horizon {horizon}h  |  MAE={mae:.2f}  R²={r2:.3f}",
        fontsize=14, fontweight="bold", color=NAVY, y=1.02
    )

    # ── Scatter ───────────────────────────────────────────────────
    ax = axes[0]
    lim = max(y_test.max(), preds.max()) + 2
    ax.scatter(y_test, preds, alpha=0.35, s=18, color=TEAL, edgecolors="none")
    ax.plot([0, lim], [0, lim], "--", color=NAVY, lw=1.5, label="Prédiction parfaite")
    ax.fill_between([0, lim], [0-2, lim-2], [0+2, lim+2],
                     alpha=0.08, color=MINT, label="Intervalle ±2 adm")
    ax.set_xlabel("Admissions réelles")
    ax.set_ylabel("Admissions prédites")
    ax.set_title("Scatter — Prédit vs Réel")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=9)

    # Annotation corrélation
    ax.text(0.05, 0.93, f"R² = {r2:.3f}", transform=ax.transAxes,
            fontsize=10, color=NAVY, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT, alpha=0.8))

    # ── Série temporelle ──────────────────────────────────────────
    ax2 = axes[1]
    if "date" in df_test.columns:
        dates = pd.to_datetime(df_test["date"].values)
    else:
        dates = np.arange(len(y_test))

    # Trier par date pour un graphique cohérent
    sort_idx = np.argsort(dates) if hasattr(dates, '__iter__') else np.arange(len(y_test))
    y_sorted = np.array(y_test)[sort_idx]
    p_sorted = preds[sort_idx]
    d_sorted = np.array(dates)[sort_idx] if hasattr(dates, '__iter__') else sort_idx

    # Sélectionner un canton pour la lisibilité (BE = le plus représentatif)
    if "kanton_hospital" in df_test.columns:
        mask_be = df_test["kanton_hospital"].values == "BE"
        if mask_be.sum() > 10:
            y_sorted = np.array(y_test)[mask_be]
            p_sorted = preds[mask_be]
            d_sorted = np.array(dates)[mask_be] if hasattr(dates, '__iter__') else np.arange(mask_be.sum())
            ax2.set_title("Série temporelle — Canton BE (test set)")
        else:
            ax2.set_title("Série temporelle — Test set")
    else:
        ax2.set_title("Série temporelle — Test set")

    ax2.plot(d_sorted, y_sorted, color=NAVY, lw=1.5, label="Réel", alpha=0.85)
    ax2.plot(d_sorted, p_sorted, color=MINT, lw=1.5, label=f"Prédit {horizon}h",
             linestyle="--", alpha=0.9)
    ax2.fill_between(d_sorted,
                     np.maximum(0, p_sorted - 2), p_sorted + 2,
                     alpha=0.15, color=MINT, label="Intervalle ±2")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Admissions urgences")
    ax2.legend(fontsize=9)

    if hasattr(d_sorted[0], 'strftime'):
        fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    out = PLOTS_DIR / f"01_predictions_vs_actual_{horizon}h.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    logger.info(f"  Plot saved: {out.name}")
    return out


def plot_shap_importance(
    shap_importance: pd.DataFrame,
    horizon: int,
    top_n: int = 15,
) -> Path:
    """
    Plot 2 — SHAP Feature Importance
    Bar chart horizontal des features les plus influentes.
    """
    df = shap_importance.head(top_n).copy()
    df = df.sort_values("mean_shap")  # ascending pour barh

    colors = [MINT if i >= len(df) - 3 else TEAL if i >= len(df) - 7 else GRAY
              for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["feature"], df["mean_shap"], color=colors,
                   edgecolor="none", height=0.65)

    # Valeurs sur les barres
    for bar, val in zip(bars, df["mean_shap"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=9, color=NAVY)

    ax.set_xlabel("Importance SHAP moyenne (|valeur|)")
    ax.set_title(
        f"Feature Importance SHAP — Horizon {horizon}h\n"
        f"(Top {top_n} features sur {len(shap_importance)} totales)",
        pad=12
    )

    # Légende couleurs
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=MINT,  label="Top 3 — très haute importance"),
        Patch(facecolor=TEAL,  label="Top 4–7 — haute importance"),
        Patch(facecolor=GRAY,  label="Reste — importance modérée"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Annotation source
    ax.text(0.98, 0.02,
            f"Sources : SpiGes · MétéoSuisse · ASTRA trafic\n"
            f"Modèle : XGBoost | Dataset : 2 479 lignes × 10 cantons",
            transform=ax.transAxes, fontsize=7.5,
            ha="right", va="bottom", color=GRAY,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT, alpha=0.6))

    ax.set_xlim(0, df["mean_shap"].max() * 1.18)
    plt.tight_layout()
    out = PLOTS_DIR / f"02_shap_importance_{horizon}h.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    logger.info(f"  Plot saved: {out.name}")
    return out


def plot_residuals(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizon: int,
) -> Path:
    """
    Plot 3 — Analyse des résidus
    Distribution des erreurs + résidus vs prédit.
    """
    preds   = np.maximum(0, model.predict(X_test))
    errors  = np.array(y_test) - preds
    mae     = mean_absolute_error(y_test, preds)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Analyse des résidus — Horizon {horizon}h",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.02)

    # ── Distribution des erreurs ──────────────────────────────────
    ax = axes[0]
    ax.hist(errors, bins=30, color=TEAL, edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(0,        color=NAVY, lw=2,   linestyle="-",  label="Erreur nulle")
    ax.axvline(mae,      color=ACC,  lw=1.5, linestyle="--", label=f"+MAE = +{mae:.2f}")
    ax.axvline(-mae,     color=ACC,  lw=1.5, linestyle="--", label=f"−MAE = −{mae:.2f}")
    ax.axvline(errors.std(),  color=RED, lw=1,  linestyle=":",  label=f"+σ = {errors.std():.2f}")
    ax.axvline(-errors.std(), color=RED, lw=1,  linestyle=":",  label=f"−σ = {errors.std():.2f}")

    ax.set_xlabel("Erreur (réel − prédit)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des erreurs de prédiction")
    ax.legend(fontsize=8)

    # Stats dans le coin
    stats_txt = (
        f"Moyenne : {errors.mean():.2f}\n"
        f"Std     : {errors.std():.2f}\n"
        f"P5/P95  : [{np.percentile(errors,5):.1f}, {np.percentile(errors,95):.1f}]\n"
        f"±2 adm  : {(np.abs(errors)<=2).mean()*100:.1f}%\n"
        f"±5 adm  : {(np.abs(errors)<=5).mean()*100:.1f}%"
    )
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="right", color=NAVY,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT, alpha=0.85),
            fontfamily="monospace")

    # ── Résidus vs Prédit (homoscédasticité) ─────────────────────
    ax2 = axes[1]
    ax2.scatter(preds, errors, alpha=0.3, s=15, color=TEAL, edgecolors="none")
    ax2.axhline(0,    color=NAVY, lw=2,   linestyle="-")
    ax2.axhline(mae,  color=ACC,  lw=1.5, linestyle="--", label=f"±MAE")
    ax2.axhline(-mae, color=ACC,  lw=1.5, linestyle="--")
    ax2.fill_between([preds.min(), preds.max()], -mae, mae,
                     alpha=0.08, color=MINT, label="Intervalle ±MAE")

    # Colorer les outliers
    outliers = np.abs(errors) > 5
    if outliers.sum() > 0:
        ax2.scatter(preds[outliers], errors[outliers],
                    alpha=0.7, s=25, color=RED, edgecolors="none",
                    label=f"Erreur >5 adm ({outliers.sum()} pts, {outliers.mean()*100:.1f}%)")

    ax2.set_xlabel("Admissions prédites")
    ax2.set_ylabel("Résidu (réel − prédit)")
    ax2.set_title("Résidus vs Valeurs prédites\n(homoscédasticité)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = PLOTS_DIR / f"03_residuals_{horizon}h.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    logger.info(f"  Plot saved: {out.name}")
    return out


def plot_cv_scores(cv_scores: list[dict], horizon: int) -> Path:
    """
    Plot 4 — Cross-Validation scores par fold
    Montre la stabilité du modèle dans le temps.
    """
    folds = [s["fold"] for s in cv_scores]
    maes  = [s["mae"]  for s in cv_scores]
    rmses = [s["rmse"] for s in cv_scores]
    r2s   = [s["r2"]   for s in cv_scores]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Cross-Validation Temporelle — Horizon {horizon}h  "
        f"(TimeSeriesSplit, 5 folds, gap=10)",
        fontsize=13, fontweight="bold", color=NAVY, y=1.03
    )

    for ax, vals, name, color, unit in [
        (axes[0], maes,  "MAE",  TEAL, "admissions"),
        (axes[1], rmses, "RMSE", ACC,  "admissions"),
        (axes[2], r2s,   "R²",   MINT, ""),
    ]:
        bars = ax.bar(folds, vals, color=color, edgecolor="white",
                      linewidth=0.5, alpha=0.85, width=0.55)
        mean_v = np.mean(vals)
        ax.axhline(mean_v, color=NAVY, lw=2, linestyle="--",
                   label=f"Moyenne : {mean_v:.3f}")

        # Valeur sur chaque barre
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=NAVY)

        ax.set_xlabel("Fold (chronologique →)")
        ax.set_ylabel(f"{name} ({unit})" if unit else name)
        ax.set_title(f"{name} par fold")
        ax.set_xticks(folds)
        ax.set_xticklabels([f"F{f}\n({cv_scores[f-1]['n_train']} tr)" for f in folds],
                           fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(vals) * 1.25)

    # Annotation importante
    fig.text(0.5, -0.04,
             "Note : chaque fold entraîne sur le passé et prédit sur le futur → "
             "pas de fuite de données temporelles",
             ha="center", fontsize=9, color=GRAY, style="italic")

    plt.tight_layout()
    out = PLOTS_DIR / f"04_cv_scores_{horizon}h.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    logger.info(f"  Plot saved: {out.name}")
    return out


def plot_seasonal_analysis(df: pd.DataFrame, horizon: int) -> Path:
    """
    Plot 5 — Analyse saisonnière SpiGes + trafic
    Corrélations et patterns temporels clés du dataset.
    """
    fig = plt.figure(figsize=(15, 13))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)
    fig.suptitle(
        "Analyse des données — SpiGes × MétéoSuisse × ASTRA Trafic × Jours Fériés\n"
        "Patterns temporels & corrélations",
        fontsize=14, fontweight="bold", color=NAVY, y=1.01
    )

    # ── 1. Admissions urgences par mois ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    monthly = df.groupby("month")["notfall_admissions"].mean()
    months_lbl = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
    colors_m = [RED if monthly.get(m, 0) > monthly.mean()*1.1
                else TEAL for m in range(1, 13)]
    ax1.bar(range(1, 13), [monthly.get(m, 0) for m in range(1, 13)],
            color=colors_m, edgecolor="white", linewidth=0.4, alpha=0.9)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(months_lbl, fontsize=8, rotation=45)
    ax1.axhline(monthly.mean(), color=NAVY, lw=1.5, linestyle="--",
                label=f"Moy = {monthly.mean():.1f}")
    ax1.set_title("Admissions urgences par mois")
    ax1.set_ylabel("Moy. admissions/jour")
    ax1.legend(fontsize=8)

    # ── 2. Profil hebdomadaire ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    dow = df.groupby("day_of_week")["notfall_admissions"].mean()
    days_lbl = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
    colors_d = [GRAY if i >= 5 else TEAL for i in range(7)]
    ax2.bar(days_lbl, [dow.get(i, 0) for i in range(7)],
            color=colors_d, edgecolor="white", linewidth=0.4, alpha=0.9)
    ax2.axhline(dow.mean(), color=NAVY, lw=1.5, linestyle="--",
                label=f"Moy = {dow.mean():.1f}")
    ax2.set_title("Profil hebdomadaire")
    ax2.set_ylabel("Moy. admissions/jour")
    ax2.legend(fontsize=8)

    # ── 3. Volume trafic vs admissions ───────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if "daily_traffic_volume" in df.columns:
        sample = df.dropna(subset=["daily_traffic_volume"]).sample(
            min(500, len(df)), random_state=42
        )
        ax3.scatter(sample["daily_traffic_volume"] / 1000,
                    sample["notfall_admissions"],
                    alpha=0.25, s=12, color=TEAL, edgecolors="none")
        # Régression linéaire
        from numpy.polynomial.polynomial import polyfit
        x = sample["daily_traffic_volume"].values / 1000
        y = sample["notfall_admissions"].values
        coeffs = np.polyfit(x, y, 1)
        xline  = np.linspace(x.min(), x.max(), 100)
        corr   = np.corrcoef(x, y)[0, 1]
        ax3.plot(xline, np.polyval(coeffs, xline),
                 color=RED, lw=2, label=f"Régression (r={corr:.2f})")
        ax3.set_xlabel("Volume trafic (milliers véh/j)")
        ax3.set_ylabel("Admissions urgences")
        ax3.set_title("Trafic vs Admissions urgences")
        ax3.legend(fontsize=8)

    # ── 4. Température vs admissions ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if "temperature_avg" in df.columns:
        sample2 = df.dropna(subset=["temperature_avg"]).sample(
            min(500, len(df)), random_state=99
        )
        # Colorer par saison
        season_colors = sample2["is_winter"].map({1: TEAL, 0: ACC})
        ax4.scatter(sample2["temperature_avg"],
                    sample2["notfall_admissions"],
                    c=season_colors, alpha=0.25, s=12, edgecolors="none")
        x = sample2["temperature_avg"].values
        y = sample2["notfall_admissions"].values
        coeffs = np.polyfit(x, y, 1)
        xline  = np.linspace(x.min(), x.max(), 100)
        corr   = np.corrcoef(x, y)[0, 1]
        ax4.plot(xline, np.polyval(coeffs, xline),
                 color=RED, lw=2, label=f"Régression (r={corr:.2f})")
        from matplotlib.patches import Patch
        ax4.legend(handles=[
            Patch(color=TEAL, label="Hiver"),
            Patch(color=ACC,  label="Autres"),
            plt.Line2D([0],[0], color=RED, lw=2, label=f"r={corr:.2f}"),
        ], fontsize=8)
        ax4.set_xlabel("Température moyenne (°C)")
        ax4.set_ylabel("Admissions urgences")
        ax4.set_title("Température vs Admissions")

    # ── 5. Heatmap corrélations — incluant jours fériés ──────────
    ax5 = fig.add_subplot(gs[1, 1:])
    corr_cols = [
        "notfall_admissions",
        "notfall_lag1", "notfall_lag7", "notfall_roll7",
        "daily_traffic_volume", "avg_speed_kmh", "heavy_vehicle_pct",
        "temperature_avg", "is_winter", "is_weekend",
        "pct_elderly", "mean_severity",
        "is_federal_holiday", "is_cantonal_holiday",
        "is_school_holiday",
    ]
    available_cols = [c for c in corr_cols if c in df.columns]
    corr_matrix = df[available_cols].corr()

    short_labels = {
        "notfall_admissions":   "Notfall",
        "notfall_lag1":         "Lag J-1",
        "notfall_lag7":         "Lag J-7",
        "notfall_roll7":        "Roll 7j",
        "daily_traffic_volume": "Trafic",
        "avg_speed_kmh":        "Vitesse",
        "heavy_vehicle_pct":    "% PL",
        "temperature_avg":      "Temp.",
        "is_winter":            "Hiver",
        "is_weekend":           "WE",
        "pct_elderly":          "65+",
        "mean_severity":        "Sévérité",
        "is_federal_holiday":   "Fér.Fed",
        "is_cantonal_holiday":  "Fér.Can",
        "is_school_holiday":    "Vacances",
    }
    labels = [short_labels.get(c, c) for c in available_cols]

    sns.heatmap(
        corr_matrix,
        ax=ax5,
        annot=True, fmt=".2f",
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.4,
        annot_kws={"size": 6.5},
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"shrink": 0.6},
    )
    ax5.set_title("Heatmap corrélations — toutes sources + jours fériés", pad=10)
    ax5.tick_params(axis="x", rotation=40, labelsize=7.5)
    ax5.tick_params(axis="y", rotation=0,  labelsize=7.5)

    # ── 6. Impact jours fériés sur les admissions ─────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    if "is_federal_holiday" in df.columns:
        day_types = {
            "Jour normal\n(semaine)":  df[(df["is_weekend"]==0) & (df["is_federal_holiday"]==0) & (df["is_cantonal_holiday"]==0)]["notfall_admissions"].mean(),
            "Weekend":                 df[df["is_weekend"]==1]["notfall_admissions"].mean(),
            "Vacances\nscolaires":     df[(df["is_school_holiday"]==1) & (df["is_weekend"]==0)]["notfall_admissions"].mean(),
            "Férié\ncantonal":         df[(df["is_cantonal_holiday"]==1) & (df["is_weekend"]==0)]["notfall_admissions"].mean(),
            "Férié\nfédéral":          df[(df["is_federal_holiday"]==1) & (df["is_weekend"]==0)]["notfall_admissions"].mean(),
        }
        colors_ht = [TEAL, GRAY, ACC, "#5BA4CF", RED]
        bars = ax6.bar(list(day_types.keys()), list(day_types.values()),
                       color=colors_ht, edgecolor="white", linewidth=0.5, alpha=0.9)
        normal_mean = day_types["Jour normal\n(semaine)"]
        ax6.axhline(normal_mean, color=NAVY, lw=1.5, linestyle="--",
                    label=f"Référence = {normal_mean:.1f}")
        for bar, v in zip(bars, day_types.values()):
            ax6.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.2,
                     f"{v:.1f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=NAVY)
        ax6.set_title("Impact type de jour\nsur admissions urgences")
        ax6.set_ylabel("Moy. admissions/jour")
        ax6.legend(fontsize=8)
        ax6.set_ylim(0, max(day_types.values()) * 1.25)

    # ── 7. Jours jusqu'au prochain férié vs admissions ────────────
    ax7 = fig.add_subplot(gs[2, 1])
    if "days_to_next_holiday" in df.columns:
        bins_data = df[df["days_to_next_holiday"] <= 14].groupby("days_to_next_holiday")["notfall_admissions"].mean()
        ax7.plot(bins_data.index, bins_data.values, color=TEAL, lw=2, marker="o", markersize=4)
        ax7.axhline(df["notfall_admissions"].mean(), color=GRAY, lw=1,
                    linestyle="--", label="Moyenne globale")
        ax7.fill_between(bins_data.index, bins_data.values,
                         df["notfall_admissions"].mean(),
                         alpha=0.15, color=TEAL)
        ax7.set_xlabel("Jours jusqu'au prochain férié")
        ax7.set_ylabel("Moy. admissions urgences")
        ax7.set_title("Effet d'anticipation\ndes jours fériés")
        ax7.legend(fontsize=8)
        ax7.invert_xaxis()

    # ── 8. Distribution admissions par holiday_type ───────────────
    ax8 = fig.add_subplot(gs[2, 2])
    if "holiday_type" in df.columns:
        type_labels = {0: "Normal", 1: "Fér.Fédéral", 2: "Fér.Cantonal", 3: "Vacances"}
        type_colors = {0: TEAL, 1: RED, 2: ACC, 3: MINT}
        for ht in sorted(df["holiday_type"].unique()):
            subset = df[df["holiday_type"] == ht]["notfall_admissions"]
            label  = type_labels.get(int(ht), str(ht))
            color  = type_colors.get(int(ht), GRAY)
            ax8.hist(subset, bins=20, alpha=0.5, color=color,
                     label=f"{label} (n={len(subset)}, μ={subset.mean():.1f})",
                     edgecolor="white", linewidth=0.3)
        ax8.set_xlabel("Admissions urgences")
        ax8.set_ylabel("Fréquence")
        ax8.set_title("Distribution admissions\npar type de jour")
        ax8.legend(fontsize=7.5)

    out = PLOTS_DIR / f"05_seasonal_analysis_{horizon}h.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    logger.info(f"  Plot saved: {out.name}")
    return out


def plot_metrics_summary(all_results: dict) -> Path:
    """
    Plot 6 — Résumé des performances des 3 modèles
    Comparaison visuelle 24h / 48h / 72h.
    """
    horizons = sorted(all_results.keys())
    maes     = [all_results[h]["mae"]            for h in horizons]
    rmses    = [all_results[h]["rmse"]           for h in horizons]
    r2s      = [all_results[h]["r2"]             for h in horizons]
    acc2s    = [all_results[h]["accuracy_pm2"]   for h in horizons]
    acc5s    = [all_results[h]["accuracy_pm5"]   for h in horizons]
    cv_maes  = [all_results[h]["cv_mae_mean"]    for h in horizons]

    labels   = [f"{h}h" for h in horizons]
    x        = np.arange(len(horizons))
    colors   = [TEAL, MINT, ACC]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Rapport de performance — Swiss ED Predictor\n"
        "3 modèles XGBoost · Test set 20% chronologique",
        fontsize=14, fontweight="bold", color=NAVY, y=1.02
    )

    def _bar(ax, vals, title, ylabel, fmt=".2f", ref=None, ref_label=None):
        bars = ax.bar(x, vals, color=colors, edgecolor="white",
                      linewidth=0.5, alpha=0.9, width=0.5)
        if ref is not None:
            ax.axhline(ref, color=RED, lw=1.5, linestyle="--",
                       label=ref_label or f"Référence = {ref}")
            ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f"{v:{fmt}}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=NAVY)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.3)

    _bar(axes[0,0], maes,  "MAE — Test set",  "admissions/jour",
         ref=4.0, ref_label="Référence King et al. (UK) MAE=4.0")
    _bar(axes[0,1], rmses, "RMSE — Test set", "admissions/jour")
    _bar(axes[0,2], r2s,   "R² — Test set",   "", fmt=".4f")

    _bar(axes[1,0], acc2s, "Précision ±2 admissions", "%", fmt=".1f")
    _bar(axes[1,1], acc5s, "Précision ±5 admissions", "%", fmt=".1f")

    # CV MAE vs Test MAE
    ax = axes[1,2]
    w  = 0.3
    b1 = ax.bar(x - w/2, cv_maes, w, color=TEAL, alpha=0.85,
                edgecolor="white", label="MAE cross-validation")
    b2 = ax.bar(x + w/2, maes,    w, color=MINT, alpha=0.85,
                edgecolor="white", label="MAE test set")
    for bar, v in list(zip(b1, cv_maes)) + list(zip(b2, maes)):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.03,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("CV MAE vs Test MAE\n(vérification overfitting)")
    ax.set_ylabel("MAE (admissions)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(cv_maes), max(maes)) * 1.35)

    plt.tight_layout()
    out = PLOTS_DIR / f"06_metrics_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    logger.info(f"  Plot saved: {out.name}")
    return out


def generate_all_plots(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizon: int,
    df_test: pd.DataFrame,
    df_full: pd.DataFrame,
    shap_importance: pd.DataFrame,
    cv_scores: list[dict],
    all_results: dict | None = None,
) -> list[Path]:
    """
    Génère tous les plots pour un horizon donné.
    Appelée depuis run_pipeline() après chaque entraînement.

    Returns:
        Liste des chemins des images générées.
    """
    logger.info(f"\n  Generating plots — horizon {horizon}h →  {PLOTS_DIR}")
    paths = []

    paths.append(plot_predictions_vs_actual(model, X_test, y_test, horizon, df_test))
    paths.append(plot_shap_importance(shap_importance, horizon))
    paths.append(plot_residuals(model, X_test, y_test, horizon))
    paths.append(plot_cv_scores(cv_scores, horizon))
    paths.append(plot_seasonal_analysis(df_full, horizon))

    if all_results is not None and len(all_results) >= len([24, 48, 72]):
        paths.append(plot_metrics_summary(all_results))

    logger.success(f"  {len(paths)} plots saved → {PLOTS_DIR}")
    return paths
def run_pipeline(horizons, data_path=DATA_PATH, n_cv_splits=5,
                 test_ratio=0.20, save_shap=True, save_plots=True):
    logger.info("=" * 60)
    logger.info("Swiss ED Predictor — XGBoost Training Pipeline v1")
    logger.info(f"Dataset : {data_path.name}")
    logger.info(f"Horizons: {horizons}h | CV folds: {n_cv_splits} | Test: {int(test_ratio*100)}%")
    logger.info("=" * 60)

    df = load_and_prepare(data_path)

    available = [f for f in ALL_FEATURES if f in df.columns]
    missing   = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        logger.warning(f"Features manquantes: {missing}")

    logger.info(f"\nFeatures actives ({len(available)}) :")
    for grp, feats in [("SpiGes", SPIGES_FEATURES), ("Lags", LAG_FEATURES),
                        ("Météo", METEO_FEATURES), ("Trafic", TRAFFIC_FEATURES),
                        ("Interactions", INTERACTION_FEATURES)]:
        active = [f for f in feats if f in available]
        logger.info(f"  {grp:<14}: {len(active):>2}  {active}")

    # Split temporel
    df_sorted  = df.sort_values(["date", "kanton_hospital"]).reset_index(drop=True)
    split_idx  = int(len(df_sorted) * (1 - test_ratio))
    df_train   = df_sorted.iloc[:split_idx]
    df_test    = df_sorted.iloc[split_idx:]

    logger.info(
        f"\nSplit temporel :"
        f"\n  Train : {len(df_train):,} lignes | "
        f"{df_train['date'].min().date()} → {df_train['date'].max().date()}"
        f"\n  Test  : {len(df_test):,}  lignes | "
        f"{df_test['date'].min().date()} → {df_test['date'].max().date()}"
    )

    all_results = {}

    for horizon in horizons:
        target_col = TARGET_COLS[horizon]
        logger.info(f"\n{'▶'*3} HORIZON {horizon}h — cible: {target_col}")

        df_tr = df_train.dropna(subset=[target_col] + available)
        df_te = df_test.dropna(subset=[target_col] + available)

        X_tr = df_tr[available];  y_tr = df_tr[target_col].astype(int)
        X_te = df_te[available];  y_te = df_te[target_col].astype(int)

        logger.info(f"  Après nettoyage NaN — train: {len(X_tr):,} | test: {len(X_te):,}")

        # CV + final training
        cv_scores, _ = cross_validate(X_tr, y_tr, n_cv_splits, f"{horizon}h")
        model, metrics = train_final(X_tr, y_tr, X_te, y_te, horizon)

        metrics["cv_mae_mean"] = round(np.mean([s["mae"] for s in cv_scores]), 3)
        metrics["cv_mae_std"]  = round(np.std( [s["mae"] for s in cv_scores]), 3)
        metrics["cv_r2_mean"]  = round(np.mean([s["r2"]  for s in cv_scores]), 4)

        # SHAP
        X_all = df_sorted.dropna(subset=available)[available]
        shap_imp = compute_shap(model, X_all, horizon, save=save_shap)
        metrics["top_features"] = shap_imp.head(5)["feature"].tolist()

        # Sauvegardes
        joblib.dump(model, MODEL_DIR / f"xgboost_ed_{horizon}h.pkl")
        (MODEL_DIR / f"features_{horizon}h.txt").write_text("\n".join(available))
        with open(MODEL_DIR / f"metrics_{horizon}h.json", "w") as f:
            json.dump(metrics, f, indent=2)

        all_results[horizon] = metrics

        # ── Plots ─────────────────────────────────────────────────
        if save_plots:
            try:
                generate_all_plots(
                    model=model,
                    X_test=X_te,
                    y_test=y_te,
                    horizon=horizon,
                    df_test=df_te,
                    df_full=df_sorted,
                    shap_importance=shap_imp,
                    cv_scores=cv_scores,
                    all_results=all_results if len(all_results) == len(horizons) else None,
                )
            except Exception as e:
                logger.warning(f"  Plot generation failed: {e}")

        logger.success(f"  ✅ Modèle {horizon}h → models/xgboost_ed_{horizon}h.pkl")

    _print_report(all_results)

    # ── Plot récapitulatif final (tous horizons) ──────────────────
    if save_plots and len(all_results) > 1:
        try:
            plot_metrics_summary(all_results)
            logger.success(f"Summary plot → {PLOTS_DIR}/06_metrics_summary.png")
        except Exception as e:
            logger.warning(f"Summary plot failed: {e}")
    with open(MODEL_DIR / "training_report.json", "w") as f:
        json.dump({str(k): {kk: (vv if isinstance(vv,(int,float,str,list)) else str(vv))
                             for kk,vv in v.items()}
                   for k,v in all_results.items()}, f, indent=2)

    return all_results


def _print_report(results):
    logger.info(f"\n{'═'*70}")
    logger.info("RAPPORT FINAL — Performances test set (20% final, ordre chronologique)")
    logger.info(f"{'═'*70}")
    logger.info(f"  {'Horizon':>8} {'MAE':>7} {'RMSE':>7} {'R²':>8} {'MAPE':>8} {'±2adm':>7} {'±5adm':>7}")
    logger.info(f"  {'─'*60}")
    for h, m in sorted(results.items()):
        logger.info(
            f"  {str(h)+'h':>8} {m['mae']:>7.2f} {m['rmse']:>7.2f} "
            f"{m['r2']:>8.4f} {m['mape_pct']:>7.1f}% "
            f"{m['accuracy_pm2']:>6.1f}% {m['accuracy_pm5']:>6.1f}%"
        )
    logger.info(f"  {'─'*60}")
    logger.info("  Top features (SHAP) :")
    for h, m in sorted(results.items()):
        logger.info(f"    {h}h : {' > '.join(m['top_features'])}")
    logger.info(f"{'═'*70}")


# ═══════════════════════════════════════════════════════════════════
# 7. INFÉRENCE
# ═══════════════════════════════════════════════════════════════════

def predict_single(canton, horizon, features, model_dir=MODEL_DIR):
    model_path    = model_dir / f"xgboost_ed_{horizon}h.pkl"
    features_path = model_dir / f"features_{horizon}h.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path} — lancer train.py d'abord")
    model         = joblib.load(model_path)
    feature_names = features_path.read_text().strip().split("\n")
    row           = pd.DataFrame([{f: features.get(f, 0) for f in feature_names}])
    pred          = float(max(0, model.predict(row)[0]))
    return {
        "canton": canton, "horizon_h": horizon,
        "predicted_admissions": int(round(pred)),
        "confidence_low":  max(0, int(round(pred)) - 3),
        "confidence_high": int(round(pred)) + 3,
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swiss ED Predictor — XGBoost Training")
    parser.add_argument("--horizon", default="all", help="24 | 48 | 72 | all")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--save-shap", action="store_true", default=True)
    parser.add_argument("--no-plots", action="store_true", default=False,
                        help="Désactiver la génération des plots")
    parser.add_argument("--test-ratio", type=float, default=0.20)
    args = parser.parse_args()

    horizons = [24, 48, 72] if args.horizon == "all" else [int(args.horizon)]

    run_pipeline(
        horizons=horizons,
        data_path=Path(args.data),
        n_cv_splits=args.cv_splits,
        test_ratio=args.test_ratio,
        save_shap=args.save_shap,
        save_plots=not args.no_plots,
    )


