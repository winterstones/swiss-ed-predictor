"""
Feature Engineering Pipeline — v3
====================================
Point d'entrée central qui assemble TOUTES les sources de données
en une feature matrix unique prête pour XGBoost.

Flux de données :
─────────────────────────────────────────────────────────────────
  src/ingestion/spiges.py
    └─ load_daily()              → SpiGes daily (lags, démographie)
    └─ compute_seasonal_patterns()
    └─ compute_demographic_risk()

  src/ingestion/meteo_swiss.py
    └─ build_meteo_dataset()     → météo + pollen + anomalies thermiques
                                    27 features (SMN + pollen + NBCN)

  src/ingestion/transport.py     ← NOUVEAU v3
    └─ build_traffic_dataset()   → trafic routier ASTRA (DATEX II/SOAP)
                                    8 features (volume, anomalie, poids lourds)

         ▼  MERGE sur [date, canton]  ▼

  build_full_dataset()           → DataFrame complet 55+ features
  get_X_y(horizon=24)            → (X, y) prêts pour XGBoost
─────────────────────────────────────────────────────────────────

Sources intégrées :
  SpiGes/OFSP         → patterns historiques, lags, démographie
  MétéoSuisse SMN     → température, précipitations, vent, humidité
  MétéoSuisse pollen  → index pollinique (asthme, allergies)
  MétéoSuisse NBCN    → anomalies thermiques vs normales 1981-2010
  ASTRA traffic       → volume trafic routier, anomalie, poids lourds
                        Token: OPENTRANSPORT_TOKEN dans .env

Usage typique (dans train.py ou un notebook) :
    from src.features.engineering import get_X_y

    # Avec toutes les sources (connexion réseau)
    X, y = get_X_y(horizon=24)

    # Mode offline hackathon (données sample uniquement)
    X, y = get_X_y(horizon=24, include_meteo=False, include_traffic=False)

    model.fit(X, y)
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# ── Imports ingestion ─────────────────────────────────────────────
from src.ingestion.spiges import (
    load_daily,
    compute_seasonal_patterns,
    compute_demographic_risk,
    load_patients,
    DEFAULT_FEATURE_COLS,
)
from src.ingestion.meteo_swiss import (
    build_meteo_dataset,
    get_feature_names as meteo_feature_names,
)
from src.ingestion.transport import (
    build_traffic_dataset,
    get_traffic_feature_names,
)


# ═══════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE — build_full_dataset()
# ═══════════════════════════════════════════════════════════════════

def build_full_dataset(
    start_date: Optional[date] = None,
    end_date:   Optional[date] = None,
    spiges_daily_path:    str | Path = "data/sample/spiges_daily_aggregated.csv",
    spiges_patients_path: str | Path = "data/sample/spiges_synthetic_patients.csv",
    include_meteo:      bool = True,
    include_pollen:     bool = True,
    include_traffic:    bool = True,
    include_seasonal:   bool = True,
    include_demographic:bool = True,
) -> pd.DataFrame:
    """
    Assemble la feature matrix complète en mergeant toutes les sources.

    Étapes :
      1. SpiGes daily       — lags, démographie, temporel
      2. MétéoSuisse        — SMN + pollen + anomalies NBCN
      3. Merge SpiGes × Météo sur [date, canton]
      4. Trafic ASTRA       — compteurs routiers DATEX II
      5. Merge × Trafic sur [date, canton]
      6. Indice saisonnier  — SpiGes canton × mois
      7. Facteurs démo      — SpiGes patients annualisés
      8. Interactions       — features croisées multi-sources
      9. Jours fériés       — calendrier fédéral suisse

    Args:
        start_date:            Début période (défaut: min date SpiGes)
        end_date:              Fin période   (défaut: max date SpiGes)
        spiges_daily_path:     Chemin spiges_daily_aggregated.csv
        spiges_patients_path:  Chemin spiges_synthetic_patients.csv
        include_meteo:         Inclure MétéoSuisse (réseau requis)
        include_pollen:        Inclure données pollen dans météo
        include_traffic:       Inclure trafic ASTRA (OPENTRANSPORT_TOKEN requis)
        include_seasonal:      Ajouter indice saisonnier SpiGes
        include_demographic:   Ajouter facteurs démographiques SpiGes

    Returns:
        DataFrame complet — colonnes garanties :
          date, canton, burnr_gesv, notfall_admissions,
          target_notfall_next24h / next48h / next72h,
          + toutes les features météo + trafic + SpiGes + interactions
    """
    logger.info("=" * 60)
    logger.info("Building full feature dataset — v3 (SpiGes+Météo+Trafic)")
    logger.info("=" * 60)

    # ── ÉTAPE 1 : SpiGes daily ────────────────────────────────────
    logger.info("Step 1/7 — Loading SpiGes daily")
    spiges = load_daily(
        path=spiges_daily_path,
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date)     if end_date   else None,
        drop_na_targets=False,
    )

    if spiges.empty:
        raise RuntimeError("SpiGes daily est vide — vérifier le fichier CSV")

    _start = start_date or spiges["date"].min().date()
    _end   = end_date   or spiges["date"].max().date()
    logger.info(f"  SpiGes: {len(spiges):,} rows | {_start} → {_end}")

    df = spiges.copy()

    # ── ÉTAPE 2+3 : MétéoSuisse ───────────────────────────────────
    if include_meteo:
        logger.info("Step 2/7 — Fetching MétéoSuisse (SMN + pollen + NBCN)")
        try:
            meteo = build_meteo_dataset(
                start_date=_start,
                end_date=_end,
                include_pollen=include_pollen,
                include_normals=True,
            )
            if not meteo.empty:
                logger.info("Step 3/7 — Merging SpiGes × Météo on [date, canton]")
                df = df.merge(
                    meteo,
                    on=["date", "canton"],
                    how="left",
                    suffixes=("", "_meteo"),
                )
                n_meteo = len([c for c in df.columns if c in meteo_feature_names()])
                logger.success(f"  Météo merged — {n_meteo} features ajoutées")
            else:
                logger.warning("  Météo vide — skipping")
                logger.info("Step 3/7 — Skipped (no météo data)")
        except Exception as e:
            logger.warning(f"  Météo unavailable: {e} — continuing without")
            logger.info("Step 3/7 — Skipped (météo error)")
    else:
        logger.info("Step 2/7 — Météo skipped (include_meteo=False)")
        logger.info("Step 3/7 — Merge météo skipped")

    # ── ÉTAPE 4+5 : Trafic ASTRA ──────────────────────────────────
    if include_traffic:
        logger.info("Step 4/7 — Fetching ASTRA traffic counters (DATEX II)")
        try:
            traffic = build_traffic_dataset(target_date=_end)
            if not traffic.empty:
                logger.info("Step 5/7 — Merging × Trafic on [date, canton]")
                df = df.merge(
                    traffic,
                    on=["date", "canton"],
                    how="left",
                    suffixes=("", "_traffic"),
                )
                n_traffic = len([c for c in df.columns if c in get_traffic_feature_names()])
                logger.success(f"  Trafic merged — {n_traffic} features ajoutées")
            else:
                logger.warning("  Trafic vide — skipping")
                logger.info("Step 5/7 — Skipped (no traffic data)")
        except EnvironmentError:
            logger.warning(
                "  OPENTRANSPORT_TOKEN manquant — trafic skipped\n"
                "  → Ajouter dans .env : OPENTRANSPORT_TOKEN=votre_token"
            )
            logger.info("Step 5/7 — Skipped (no token)")
        except Exception as e:
            logger.warning(f"  Trafic unavailable: {e} — continuing without")
            logger.info("Step 5/7 — Skipped (traffic error)")
    else:
        logger.info("Step 4/7 — Trafic skipped (include_traffic=False)")
        logger.info("Step 5/7 — Merge trafic skipped")

    # ── ÉTAPE 6 : Indice saisonnier ───────────────────────────────
    if include_seasonal:
        logger.info("Step 6/7 — Adding seasonal index (SpiGes canton × month)")
        try:
            seasonal = compute_seasonal_patterns(spiges)
            if not seasonal.empty:
                df = df.merge(
                    seasonal[["canton", "month", "seasonal_index"]],
                    on=["canton", "month"],
                    how="left",
                )
                logger.success("  Seasonal index merged")
        except Exception as e:
            logger.warning(f"  Seasonal index failed: {e}")
    else:
        logger.info("Step 6/7 — Seasonal index skipped")

    # ── ÉTAPE 7 : Facteurs démographiques ─────────────────────────
    if include_demographic:
        logger.info("Step 7/7 — Adding demographic risk factors")
        try:
            patients = load_patients(path=spiges_patients_path)
            demo = compute_demographic_risk(patients)
            if not demo.empty:
                df = df.merge(
                    demo[["canton", "pct_elderly_annual",
                           "pct_severe_annual", "pct_ips_annual"]],
                    on="canton",
                    how="left",
                )
                logger.success("  Demographic risk merged")
        except Exception as e:
            logger.warning(f"  Demographic risk failed: {e}")
    else:
        logger.info("Step 7/7 — Demographic risk skipped")

    # ── ÉTAPE 8 : Interactions croisées ───────────────────────────
    df = _add_interaction_features(df)

    # ── ÉTAPE 9 : Jours fériés ────────────────────────────────────
    df = _add_holiday_features(df)

    # ── Tri final ─────────────────────────────────────────────────
    df = df.sort_values(["canton", "date"]).reset_index(drop=True)

    # ── Résumé des sources disponibles ────────────────────────────
    meteo_ok   = "temp_mean_c" in df.columns
    traffic_ok = "daily_traffic_volume" in df.columns
    seasonal_ok= "seasonal_index" in df.columns

    logger.success(
        f"\n{'='*60}\n"
        f"✅ Full dataset ready — v3\n"
        f"   Rows     : {len(df):,}\n"
        f"   Columns  : {df.shape[1]}\n"
        f"   Cantons  : {df['canton'].nunique()}\n"
        f"   Period   : {df['date'].min().date()} → {df['date'].max().date()}\n"
        f"   Sources  : SpiGes ✅ | Météo {'✅' if meteo_ok else '❌'} "
        f"| Trafic {'✅' if traffic_ok else '❌'} "
        f"| Saisonnier {'✅' if seasonal_ok else '❌'}\n"
        f"{'='*60}"
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# RACCOURCI PRINCIPAL — get_X_y()
# ═══════════════════════════════════════════════════════════════════

def get_X_y(
    horizon: int = 24,
    feature_cols: Optional[list[str]] = None,
    **build_kwargs,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Raccourci principal : construit le dataset complet et retourne (X, y).

    C'est la fonction appelée directement dans train.py.

    Args:
        horizon:        Horizon de prédiction : 24, 48 ou 72 (heures)
        feature_cols:   Liste features X — défaut: FULL_FEATURE_COLS
        **build_kwargs: Passés à build_full_dataset()
                        Ex: include_traffic=False pour le mode offline

    Returns:
        X: DataFrame features (sans NaN sur target)
        y: Series cible (admissions urgences à horizon heures)

    Exemples :
        >>> # Mode complet
        >>> X, y = get_X_y(horizon=24)

        >>> # Mode offline hackathon
        >>> X, y = get_X_y(horizon=48, include_meteo=False, include_traffic=False)

        >>> # Prédiction 72h sur canton BE uniquement
        >>> daily = load_daily(canton="BE")
        >>> X, y = get_X_y(horizon=72, include_traffic=False)
    """
    target_col = f"target_notfall_next{horizon}h"
    if horizon not in [24, 48, 72]:
        raise ValueError(f"horizon doit être 24, 48 ou 72 — reçu: {horizon}")

    df = build_full_dataset(**build_kwargs)

    cols      = feature_cols or FULL_FEATURE_COLS
    available = [c for c in cols if c in df.columns]
    missing   = [c for c in cols if c not in df.columns]

    if missing:
        logger.warning(
            f"{len(missing)} features manquantes (ignorées) :\n"
            + "\n".join(f"  ✗ {c}" for c in missing)
        )

    df_clean = df.dropna(subset=[target_col] + available)
    X = df_clean[available].copy()
    y = df_clean[target_col].astype(int)

    logger.success(
        f"(X, y) prêts | horizon={horizon}h | "
        f"{len(X):,} samples × {len(available)} features | "
        f"y̅={y.mean():.1f} ± {y.std():.1f}"
    )
    return X, y


# ═══════════════════════════════════════════════════════════════════
# INTERACTIONS CROISÉES — multi-sources
# ═══════════════════════════════════════════════════════════════════

def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features d'interactions croisées entre toutes les sources.

    Ces features capturent des effets combinés non linéaires :
      Météo × Temporel    : froid le lundi, pluie le weekend
      Météo × Démographie : froid + personnes âgées → urgences
      Météo × Trafic      : neige + trafic = accidents graves
      Trafic × Temporel   : pic trafic + vendredi = risque accidents
      Lag × Saisonnier    : tendance corrigée par la saison
      Pollen × Saison     : allergies printanières
    """
    # ── Météo × Temporel ─────────────────────────────────────────
    if "temp_cold" in df.columns:
        df["cold_x_winter"]  = df["temp_cold"] * df.get("is_winter", 0)
        df["cold_x_monday"]  = df["temp_cold"] * (df["day_of_week"] == 0).astype(int)

    if "is_precipitation" in df.columns:
        df["rain_x_weekend"] = df["is_precipitation"] * df.get("is_weekend", 0)

    if "temp_cold" in df.columns and "humidity_pct" in df.columns:
        df["cold_x_humid"]   = df["temp_cold"] * (df["humidity_pct"] > 80).astype(int)

    # ── Météo × Démographie ───────────────────────────────────────
    if "temp_cold" in df.columns and "pct_elderly" in df.columns:
        df["cold_x_elderly"] = df["temp_cold"] * (df["pct_elderly"] > 0.30).astype(int)

    if "is_snow" in df.columns and "pct_elderly" in df.columns:
        df["snow_x_elderly"] = df.get("is_snow", 0) * (df["pct_elderly"] > 0.30).astype(int)

    # ── Météo × Trafic — NOUVEAU v3 ───────────────────────────────
    if "is_snow" in df.columns and "daily_traffic_volume" in df.columns:
        # Neige + fort trafic → risque accidents élevé → urgences traumato
        df["snow_x_traffic"] = (
            df.get("is_snow", 0) *
            (df["daily_traffic_volume"] > df["daily_traffic_volume"].quantile(0.6)).astype(int)
        )

    if "heavy_rain" in df.columns and "heavy_vehicle_pct" in df.columns:
        # Pluie forte + beaucoup de poids lourds → accidents graves
        df["rain_x_heavy_vehicles"] = (
            df.get("heavy_rain", 0) *
            (df["heavy_vehicle_pct"] > 0.15).astype(int)
        )

    # ── Trafic × Temporel — NOUVEAU v3 ───────────────────────────
    if "high_traffic_day" in df.columns:
        # Vendredi soir + trafic élevé = pic typique accidents
        df["traffic_x_friday"] = (
            df["high_traffic_day"] *
            (df["day_of_week"] == 4).astype(int)
        )
        # Trafic anormal + jour ouvré = événement exceptionnel
        df["traffic_x_weekday"] = (
            df["high_traffic_day"] *
            (df.get("is_weekend", 0) == 0).astype(int)
        )

    if "low_traffic_day" in df.columns:
        # Trafic très bas + pas férié → grève → urgences différées
        df["low_traffic_x_no_holiday"] = (
            df["low_traffic_day"] *
            (df.get("is_holiday", 0) == 0).astype(int)
        )

    if "traffic_vs_normal" in df.columns and "is_winter" in df.columns:
        # Trafic anomalie en hiver → conditions météo exceptionnelles
        df["traffic_anomaly_x_winter"] = (
            df["traffic_vs_normal"].fillna(1.0) *
            df["is_winter"]
        )

    # ── Lag × Saisonnier ─────────────────────────────────────────
    if "notfall_lag7" in df.columns and "seasonal_index" in df.columns:
        df["lag7_x_seasonal"] = df["notfall_lag7"] * df["seasonal_index"]

    # ── Pollen × Saison ───────────────────────────────────────────
    if "pollen_index" in df.columns:
        df["pollen_x_spring"] = (
            df["pollen_index"] *
            df["month"].isin([3, 4, 5]).astype(int)
        )

    logger.debug(f"Interaction features added — total columns: {df.shape[1]}")
    return df


# ═══════════════════════════════════════════════════════════════════
# JOURS FÉRIÉS SUISSES
# ═══════════════════════════════════════════════════════════════════

def _add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features jours fériés fédéraux suisses."""
    if "date" not in df.columns:
        return df

    all_holidays = set()
    for year in df["date"].dt.year.unique():
        all_holidays.update(_get_swiss_holidays(int(year)))

    df["is_holiday"] = df["date"].dt.date.isin(all_holidays).astype(int)

    holiday_dates = pd.Series(sorted(all_holidays))
    df["days_to_next_holiday"] = df["date"].apply(
        lambda d: _days_to_next(d.date(), holiday_dates)
    )
    return df


def _get_swiss_holidays(year: int) -> set:
    from datetime import date as d
    return {
        d(year, 1, 1),    # Nouvel An
        d(year, 8, 1),    # Fête nationale
        d(year, 12, 25),  # Noël
        d(year, 12, 26),  # Saint-Étienne
    }


def _days_to_next(current: date, holidays: pd.Series) -> int:
    future = holidays[holidays > current]
    if future.empty:
        return 30
    return min(30, (future.iloc[0] - current).days)


# ═══════════════════════════════════════════════════════════════════
# FULL_FEATURE_COLS — liste complète pour XGBoost
# ═══════════════════════════════════════════════════════════════════

FULL_FEATURE_COLS = [
    # ── Temporelles (SpiGes) ──────────────────────────────────────
    "month",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "is_winter",
    "is_summer",
    "is_holiday",
    "days_to_next_holiday",

    # ── Lags SpiGes ───────────────────────────────────────────────
    "notfall_lag1",
    "notfall_lag7",
    "notfall_roll7",

    # ── Démographie daily (SpiGes) ────────────────────────────────
    "pct_elderly",
    "mean_severity",
    "mean_nems",
    "ips_cases",

    # ── Facteurs démographiques stables (SpiGes patients) ─────────
    "pct_elderly_annual",
    "pct_severe_annual",
    "pct_ips_annual",

    # ── Saisonnier (SpiGes) ───────────────────────────────────────
    "seasonal_index",

    # ── Température (MétéoSuisse SMN) ─────────────────────────────
    "temp_mean_c",
    "temp_min_c",
    "temp_max_c",
    "temp_cold",
    "temp_freezing",
    "temp_hot",
    "temp_anomaly_c",
    "cold_streak",
    "temp_range_c",

    # ── Précipitations (MétéoSuisse SMN) ──────────────────────────
    "precipitation_mm",
    "is_precipitation",
    "heavy_rain",
    "is_snow",

    # ── Vent, humidité, soleil (MétéoSuisse SMN) ──────────────────
    "wind_gust_ms",
    "strong_wind",
    "humidity_pct",
    "high_humidity",
    "sunshine_min",
    "low_sunshine",

    # ── Pollen (MétéoSuisse pollen) ───────────────────────────────
    "pollen_index",
    "high_pollen",

    # ── Trafic routier (ASTRA via opentransportdata.swiss) ─────────
    "daily_traffic_volume",
    "heavy_vehicle_pct",
    "avg_speed_kmh",
    "low_speed_flag",
    "traffic_per_hour",
    "traffic_vs_normal",
    "high_traffic_day",
    "low_traffic_day",

    # ── Interactions croisées ─────────────────────────────────────
    # Météo × Temporel
    "cold_x_winter",
    "cold_x_monday",
    "cold_x_humid",
    "rain_x_weekend",
    # Météo × Démographie
    "cold_x_elderly",
    "snow_x_elderly",
    # Météo × Trafic (NOUVEAU v3)
    "snow_x_traffic",
    "rain_x_heavy_vehicles",
    # Trafic × Temporel (NOUVEAU v3)
    "traffic_x_friday",
    "traffic_x_weekday",
    "low_traffic_x_no_holiday",
    "traffic_anomaly_x_winter",
    # Lag × Saisonnier
    "lag7_x_seasonal",
    # Pollen × Saison
    "pollen_x_spring",
]