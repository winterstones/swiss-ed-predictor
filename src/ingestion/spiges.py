"""
SpiGes Data Loader — v2
========================
Charge et prépare les données SpiGes pour le modèle de prédiction
des pics d'affluence aux urgences hospitalières suisses.

Deux modes de fonctionnement :
  1. SAMPLE (développement/hackathon)
     → Charge les fichiers synthétiques depuis data/sample/
     → spiges_daily_aggregated.csv    — agrégé par jour × hôpital
     → spiges_synthetic_patients.csv  — données patient-niveau

  2. PRODUCTION (post-hackathon)
     → Charge les vrais fichiers SpiGes XML/CSV depuis OFSP/OFS
     → Structure conforme au schéma TTL SpiGes v1.4/1.5
     → https://register.ld.admin.ch/i14y/dataset/SpiGes_Administratives

Colonnes clés produites (toutes utilisées dans features/engineering.py) :
  date, canton, burnr_gesv, notfall_admissions, total_admissions,
  pct_elderly, mean_severity, mean_nems, ips_cases, mean_los_hours,
  notfall_lag1, notfall_lag7, notfall_roll7,
  target_notfall_next24h, target_notfall_next48h, target_notfall_next72h

Source: OFSP — SpiGes Statistique des hôpitaux
Licence: Open Government Data (OGD) — réutilisation libre avec citation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# ── Chemins par défaut ────────────────────────────────────────────
SAMPLE_DIR = Path("data/sample")

SAMPLE_DAILY_CSV    = SAMPLE_DIR / "spiges_daily_aggregated.csv"
SAMPLE_PATIENTS_CSV = SAMPLE_DIR / "spiges_synthetic_patients.csv"

# ── Colonnes SpiGes officielles (schéma TTL v1.4/1.5) ────────────
# Correspondance champ SpiGes → nom ML utilisé dans le pipeline
SPIGES_FIELD_MAP = {
    # Identifiants
    "burnr_gesv":           "burnr_gesv",
    "kanton_hospital":      "canton",
    # Temporel
    "eintrittsdatum":       "eintrittsdatum",
    "year":                 "year",
    "month":                "month",
    "day_of_week":          "day_of_week",
    "week_of_year":         "week_of_year",
    "is_weekend":           "is_weekend",
    "is_winter":            "is_winter",
    "is_summer":            "is_summer",
    # Admission
    "eintrittsart":         "eintrittsart",
    "is_notfall":           "is_notfall",
    "hauptleistungsstelle": "hauptleistungsstelle",
    "einw_instanz":         "einw_instanz",
    # Patient
    "alter":                "alter",
    "alter_group":          "alter_group",
    "geschlecht":           "geschlecht",
    "wohnkanton":           "wohnkanton",
    "nationalitaet":        "nationalitaet",
    "versicherungsklasse":  "versicherungsklasse",
    # Scores cliniques
    "schwere_score":        "schwere_score",
    "nems":                 "nems",
    "aufenthalt_ips":       "aufenthalt_ips",
    "aufenthalt_imc":       "aufenthalt_imc",
    "beatmung":             "beatmung",
    # Sortie
    "los_hours":            "los_hours",
    "los_days":             "los_days",
    "austrittsentscheid":   "austrittsentscheid",
    "sekundaertransport":   "sekundaertransport",
}

# Colonnes obligatoires dans daily pour le ML
DAILY_REQUIRED_COLS = [
    "date", "canton", "burnr_gesv",
    "notfall_admissions", "total_admissions",
    "pct_elderly", "mean_severity", "mean_nems",
    "notfall_lag1", "notfall_lag7", "notfall_roll7",
    "target_notfall_next24h",
]

# Cantons hospitaliers couverts
HOSPITAL_CANTONS = ["BE", "ZH", "GE", "VD", "BS", "AG", "SG", "VS", "NE", "TI"]


# ═══════════════════════════════════════════════════════════════════
# 1. CHARGEMENT — données daily agrégées (fichier principal ML)
# ═══════════════════════════════════════════════════════════════════

def load_daily(
    path: str | Path = SAMPLE_DAILY_CSV,
    canton: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    drop_na_targets: bool = True,
) -> pd.DataFrame:
    """
    Charge spiges_daily_aggregated.csv — fichier principal pour l'entraînement ML.

    C'est le fichier d'entrée direct de XGBoost :
      X = features temporelles + lag + démographiques + météo (à merger)
      y = target_notfall_next24h / next48h / next72h

    Args:
        path:             Chemin vers le CSV (défaut: data/sample/)
        canton:           Filtrer sur un canton (ex: "BE") — None = tous
        start_date:       Filtre début "YYYY-MM-DD"
        end_date:         Filtre fin   "YYYY-MM-DD"
        drop_na_targets:  Supprimer les lignes sans target (derniers 3 jours)

    Returns:
        DataFrame prêt pour l'entraînement, trié par [canton, date]

    Exemple:
        >>> daily = load_daily()
        >>> X = daily[FEATURE_COLS]
        >>> y = daily["target_notfall_next24h"]
    """
    path = Path(path)
    logger.info(f"Loading SpiGes daily from {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"SpiGes daily file not found: {path}\n"
            f"→ Placer spiges_daily_aggregated.csv dans {SAMPLE_DIR}/"
        )

    df = pd.read_csv(path, low_memory=False)

    # ── Parse date ───────────────────────────────────────────────
    date_col = _find_date_col(df)
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])

    # ── Normaliser le nom du canton ──────────────────────────────
    if "kanton_hospital" in df.columns and "canton" not in df.columns:
        df = df.rename(columns={"kanton_hospital": "canton"})

    # ── Filtres ──────────────────────────────────────────────────
    if canton:
        df = df[df["canton"] == canton.upper()]
        logger.info(f"  Filtered to canton {canton}: {len(df):,} rows")

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    # ── Supprimer les lignes sans target ─────────────────────────
    if drop_na_targets and "target_notfall_next24h" in df.columns:
        before = len(df)
        df = df.dropna(subset=["target_notfall_next24h"])
        dropped = before - len(df)
        if dropped:
            logger.debug(f"  Dropped {dropped} rows with no target (last 3 days per hospital)")

    # ── Typage ───────────────────────────────────────────────────
    df = _cast_daily_types(df)

    # ── Tri ──────────────────────────────────────────────────────
    df = df.sort_values(["canton", "date"]).reset_index(drop=True)

    # ── Validation ───────────────────────────────────────────────
    _validate_daily(df)

    logger.success(
        f"SpiGes daily loaded: {len(df):,} rows × {df.shape[1]} cols "
        f"| {df['canton'].nunique()} cantons "
        f"| {df['date'].min().date()} → {df['date'].max().date()}"
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# 2. CHARGEMENT — données patient-niveau (exploration & features avancées)
# ═══════════════════════════════════════════════════════════════════

def load_patients(
    path: str | Path = SAMPLE_PATIENTS_CSV,
    notfall_only: bool = True,
    canton: Optional[str] = None,
    min_year: int = 2021,
) -> pd.DataFrame:
    """
    Charge spiges_synthetic_patients.csv — données patient-niveau SpiGes.

    Utile pour :
      - Explorer les distributions (âge, sévérité, diagnostic)
      - Construire des features avancées (ratio IPS, profil démographique)
      - Valider les patterns saisonniers

    Champs SpiGes officiels inclus : abc_fall, eintrittsart, hauptleistungsstelle,
    schwere_score, nems, beatmung, aufenthalt_ips, austrittsentscheid, etc.

    Args:
        path:         Chemin vers le CSV (défaut: data/sample/)
        notfall_only: Si True, garde uniquement les urgences (eintrittsart == 1)
        canton:       Filtrer sur un canton
        min_year:     Année minimale (défaut: 2021)

    Returns:
        DataFrame patient-niveau nettoyé
    """
    path = Path(path)
    logger.info(f"Loading SpiGes patients from {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"SpiGes patients file not found: {path}\n"
            f"→ Placer spiges_synthetic_patients.csv dans {SAMPLE_DIR}/"
        )

    df = pd.read_csv(path, low_memory=False)

    # ── Parse dates ──────────────────────────────────────────────
    if "eintrittsdatum" in df.columns:
        df["eintrittsdatum_dt"] = pd.to_datetime(
            df["eintrittsdatum"].astype(str), format="%Y%m%d", errors="coerce"
        )

    if "austrittsdatum" in df.columns:
        df["austrittsdatum_dt"] = pd.to_datetime(
            df["austrittsdatum"].astype(str), format="%Y%m%d", errors="coerce"
        )

    # ── Normaliser canton ────────────────────────────────────────
    if "kanton_hospital" in df.columns and "canton" not in df.columns:
        df = df.rename(columns={"kanton_hospital": "canton"})

    # ── Filtres ──────────────────────────────────────────────────
    if notfall_only:
        if "is_notfall" in df.columns:
            df = df[df["is_notfall"] == 1]
        elif "eintrittsart" in df.columns:
            df = df[df["eintrittsart"] == 1]
        logger.info(f"  Urgences only: {len(df):,} patients")

    if canton:
        df = df[df["canton"] == canton.upper()]

    if min_year and "year" in df.columns:
        df = df[df["year"] >= min_year]

    df = df.reset_index(drop=True)
    logger.success(f"SpiGes patients loaded: {len(df):,} rows × {df.shape[1]} cols")
    return df


# ═══════════════════════════════════════════════════════════════════
# 3. FEATURES — calculs à partir des données SpiGes
# ═══════════════════════════════════════════════════════════════════

def compute_seasonal_patterns(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'indice saisonnier par canton × mois.

    L'indice saisonnier = ratio admissions du mois / moyenne annuelle du canton.
    > 1.0 = mois de pic, < 1.0 = mois creux.

    Utilisé comme feature dans features/engineering.py pour capturer
    les patterns de long terme indépendamment des lags courts.

    Args:
        daily: Output de load_daily()

    Returns:
        DataFrame avec colonnes: canton, month, avg_daily_admissions, seasonal_index
    """
    if daily.empty or "notfall_admissions" not in daily.columns:
        logger.warning("Cannot compute seasonal patterns — empty or missing column")
        return pd.DataFrame()

    patterns = (
        daily.groupby(["canton", "month"])["notfall_admissions"]
        .mean()
        .reset_index()
        .rename(columns={"notfall_admissions": "avg_daily_admissions"})
    )

    canton_annual = patterns.groupby("canton")["avg_daily_admissions"].transform("mean")
    patterns["seasonal_index"] = (patterns["avg_daily_admissions"] / canton_annual).round(3)

    logger.info(
        f"Seasonal patterns: {patterns['canton'].nunique()} cantons × 12 months\n"
        + patterns.groupby("canton")["seasonal_index"].agg(["min", "max"])
          .round(2).to_string()
    )
    return patterns.sort_values(["canton", "month"]).reset_index(drop=True)


def compute_demographic_risk(patients: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les facteurs de risque démographiques par canton.

    Features produites :
      - pct_elderly_annual    : part annuelle de patients 65+ (risque hivernal)
      - pct_severe_annual     : part de cas sévérité >= 3
      - pct_ips_annual        : taux de transfert en soins intensifs
      - mean_los_by_severity  : durée de séjour par score de sévérité

    Ces features sont des constantes par canton — utiles comme prior
    dans le modèle, pas comme features temporelles.

    Args:
        patients: Output de load_patients()

    Returns:
        DataFrame avec une ligne par canton
    """
    if patients.empty:
        return pd.DataFrame()

    agg = {}

    # Part de patients ≥ 65 ans
    if "alter" in patients.columns:
        patients["is_elderly"] = (patients["alter"] >= 65).astype(int)
        agg["pct_elderly_annual"] = ("is_elderly", "mean")

    # Part de cas sévères (schwere_score >= 3)
    if "schwere_score" in patients.columns:
        patients["is_severe"] = (patients["schwere_score"] >= 3).astype(int)
        agg["pct_severe_annual"] = ("is_severe", "mean")

    # Taux de passage en soins intensifs
    if "aufenthalt_ips" in patients.columns:
        patients["went_to_ips"] = (patients["aufenthalt_ips"] > 0).astype(int)
        agg["pct_ips_annual"] = ("went_to_ips", "mean")

    # Durée de séjour moyenne
    if "los_hours" in patients.columns:
        agg["mean_los_hours"] = ("los_hours", "mean")

    if not agg:
        return pd.DataFrame()

    result = (
        patients.groupby("canton")
        .agg(**{k: v for k, v in agg.items()})
        .round(3)
        .reset_index()
    )

    logger.info(f"Demographic risk computed for {len(result)} cantons")
    return result


def compute_weekly_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le profil hebdomadaire des urgences par canton.

    Produit le facteur multiplicatif attendu pour chaque jour de la semaine.
    Ex: lundi = 1.18 (18% au-dessus de la moyenne), dimanche = 0.72.

    Utilisé dans features/engineering.py pour pondérer les prédictions.

    Args:
        daily: Output de load_daily()

    Returns:
        DataFrame avec colonnes: canton, day_of_week, weekly_factor
    """
    if daily.empty:
        return pd.DataFrame()

    profile = (
        daily.groupby(["canton", "day_of_week"])["notfall_admissions"]
        .mean()
        .reset_index()
        .rename(columns={"notfall_admissions": "avg_by_dow"})
    )

    canton_mean = profile.groupby("canton")["avg_by_dow"].transform("mean")
    profile["weekly_factor"] = (profile["avg_by_dow"] / canton_mean).round(3)

    return profile.sort_values(["canton", "day_of_week"]).reset_index(drop=True)


def get_ml_ready(
    daily: Optional[pd.DataFrame] = None,
    horizon: int = 24,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retourne (X, y) prêts pour XGBoost.

    Raccourci pour éviter de répéter la sélection de colonnes
    dans les notebooks et dans train.py.

    Args:
        daily:        Output de load_daily() — si None, charge automatiquement
        horizon:      24, 48 ou 72 heures
        feature_cols: Liste de colonnes X — si None, utilise DEFAULT_FEATURE_COLS

    Returns:
        X: DataFrame features
        y: Series target

    Exemple:
        >>> X, y = get_ml_ready(horizon=24)
        >>> model.fit(X, y)
    """
    if daily is None:
        daily = load_daily()

    target_col = f"target_notfall_next{horizon}h"
    if target_col not in daily.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"horizon doit être 24, 48 ou 72."
        )

    cols = feature_cols or DEFAULT_FEATURE_COLS
    available = [c for c in cols if c in daily.columns]
    missing   = [c for c in cols if c not in daily.columns]

    if missing:
        logger.warning(f"Missing feature columns (will be ignored): {missing}")

    df_clean = daily.dropna(subset=[target_col] + available)
    X = df_clean[available]
    y = df_clean[target_col].astype(int)

    logger.info(
        f"ML-ready dataset: {len(X):,} samples × {len(available)} features "
        f"| target: {target_col} | mean_y={y.mean():.1f}"
    )
    return X, y


# ── Features par défaut pour XGBoost ─────────────────────────────
DEFAULT_FEATURE_COLS = [
    # Temporelles
    "month",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "is_winter",
    "is_summer",
    # Lags SpiGes (mémoire historique)
    "notfall_lag1",
    "notfall_lag7",
    "notfall_roll7",
    # Démographiques (agrégés journaliers)
    "pct_elderly",
    "mean_severity",
    "mean_nems",
    "ips_cases",
    "mean_los_hours",
    # Météo (à merger depuis meteo_swiss.py)
    "temp_mean_c",
    "temp_min_c",
    "is_precipitation",
    "is_snow",
    "temp_cold",
    "temp_freezing",
    "cold_streak",
    "humidity_pct",
    "pollen_index",
    "temp_anomaly_c",
    # Interactions météo × temporel
    "cold_x_humid",
]


# ═══════════════════════════════════════════════════════════════════
# 4. HELPERS INTERNES
# ═══════════════════════════════════════════════════════════════════

def _find_date_col(df: pd.DataFrame) -> str:
    """Trouve la colonne date dans le DataFrame."""
    candidates = ["date", "eintrittsdatum_dt", "eintrittsdatum", "datum"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Aucune colonne date trouvée parmi: {df.columns.tolist()}")


def _cast_daily_types(df: pd.DataFrame) -> pd.DataFrame:
    """Force les types corrects sur le DataFrame daily."""
    int_cols = [
        "year", "month", "day_of_week", "week_of_year",
        "is_weekend", "is_winter", "is_summer",
        "total_admissions", "notfall_admissions", "ips_cases",
    ]
    float_cols = [
        "pct_elderly", "mean_severity", "mean_nems", "mean_los_hours",
        "notfall_lag1", "notfall_lag7", "notfall_roll7",
        "target_notfall_next24h", "target_notfall_next48h", "target_notfall_next72h",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _validate_daily(df: pd.DataFrame) -> None:
    """Valide les colonnes obligatoires et loggue un warning si manquantes."""
    missing = [c for c in DAILY_REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.warning(
            f"Colonnes obligatoires manquantes dans daily: {missing}\n"
            f"  → Certaines features ML ne seront pas disponibles"
        )
    else:
        logger.debug("Daily validation OK — toutes les colonnes obligatoires présentes")


# ═══════════════════════════════════════════════════════════════════
# 5. PRODUCTION — chargement vrais fichiers SpiGes XML/CSV (post-hackathon)
# ═══════════════════════════════════════════════════════════════════

def load_spiges_xml(path: str | Path) -> pd.DataFrame:
    """
    Charge un vrai fichier SpiGes XML (format OFSP officiel).

    Structure attendue conforme au schéma TTL SpiGes v1.4/1.5 :
    https://register.ld.admin.ch/i14y/dataset/SpiGes_Administratives

    NOTE : Cette fonction est un STUB pour le hackathon.
    Elle sera implémentée en Phase 2 avec les vrais fichiers OFS.

    Args:
        path: Chemin vers le fichier XML SpiGes

    Returns:
        DataFrame avec les mêmes colonnes que load_patients()
    """
    logger.warning(
        "load_spiges_xml() est un stub — Phase 2 post-hackathon.\n"
        "Utiliser load_patients() avec les données synthétiques pour l'instant."
    )
    # TODO Phase 2:
    #   import xml.etree.ElementTree as ET
    #   tree = ET.parse(path)
    #   root = tree.getroot()
    #   → parser les éléments AdminstrativesType
    #   → mapper les champs selon SPIGES_FIELD_MAP
    raise NotImplementedError(
        "Implémentation prévue en Phase 2. "
        "Utiliser load_patients() pour le hackathon."
    )


# ═══════════════════════════════════════════════════════════════════
# CLI — test rapide
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("SpiGes Data Loader — test")
    print("=" * 60)

    # Test 1 : daily
    try:
        daily = load_daily()
        print(f"\n✅ Daily: {len(daily):,} rows × {daily.shape[1]} cols")
        print(daily[["date", "canton", "notfall_admissions", "pct_elderly"]].head(5).to_string(index=False))

        # Patterns saisonniers
        patterns = compute_seasonal_patterns(daily)
        print(f"\n📊 Seasonal patterns (extrait BE):")
        print(patterns[patterns["canton"] == "BE"].to_string(index=False))

        # Profil hebdomadaire
        profile = compute_weekly_profile(daily)
        print(f"\n📅 Weekly profile (BE):")
        days = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        be = profile[profile["canton"] == "BE"].copy()
        be["jour"] = be["day_of_week"].map(lambda x: days[x])
        print(be[["jour", "weekly_factor"]].to_string(index=False))

    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("→ Lance d'abord: python generate_spiges.py")

    # Test 2 : get_ml_ready
    try:
        X, y = get_ml_ready(horizon=24)
        print(f"\n✅ ML-ready X: {X.shape}, y: {y.shape}")
        print(f"   Features: {X.columns.tolist()}")
        print(f"   y mean: {y.mean():.1f}, std: {y.std():.1f}")
    except Exception as e:
        print(f"\n⚠️  get_ml_ready: {e}")

    # Test 3 : patients (optionnel)
    if len(sys.argv) > 1 and sys.argv[1] == "--patients":
        try:
            pts = load_patients()
            risk = compute_demographic_risk(pts)
            print(f"\n✅ Patients: {len(pts):,} rows")
            print(f"\n👥 Demographic risk:")
            print(risk.to_string(index=False))
        except FileNotFoundError as e:
            print(f"\n⚠️  {e}")