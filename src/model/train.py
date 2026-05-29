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

INTERACTION_FEATURES = [
    "cold_x_weekend",
    "traffic_x_winter",
    "lag7_x_winter",
    "elderly_x_winter",
    "traffic_x_elderly",
]

ALL_FEATURES = (
    SPIGES_FEATURES + LAG_FEATURES
    + METEO_FEATURES + TRAFFIC_FEATURES
    + INTERACTION_FEATURES
)

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

    # Encodage canton
    canton_map = {c: i for i, c in enumerate(sorted(df["kanton_hospital"].unique()))}
    df["canton_enc"] = df["kanton_hospital"].map(canton_map)
    with open(MODEL_DIR / "canton_map.json", "w") as f:
        json.dump(canton_map, f)

    logger.info(f"  NaN après imputation — temp: {df['temperature_avg'].isna().sum()} | lag: {df['notfall_lag1'].isna().sum()}")
    return df


# ═══════════════════════════════════════════════════════════════════
# 3. CROSS-VALIDATION TEMPORELLE
# ═══════════════════════════════════════════════════════════════════

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

def run_pipeline(horizons, data_path=DATA_PATH, n_cv_splits=5,
                 test_ratio=0.20, save_shap=True):
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
        logger.success(f"  ✅ Modèle {horizon}h → models/xgboost_ed_{horizon}h.pkl")

    _print_report(all_results)
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
    parser.add_argument("--test-ratio", type=float, default=0.20)
    args = parser.parse_args()

    horizons = [24, 48, 72] if args.horizon == "all" else [int(args.horizon)]

    run_pipeline(
        horizons=horizons,
        data_path=Path(args.data),
        n_cv_splits=args.cv_splits,
        test_ratio=args.test_ratio,
        save_shap=args.save_shap,
    )