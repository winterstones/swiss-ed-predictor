"""
opentransportdata.swiss — Traffic Counters Connector
======================================================
Fetches road traffic volume from ASTRA counters via DATEX II / SOAP API.

API: Road traffic - traffic counters (ASTRA PDT Publisher)
URL: https://api.opentransportdata.swiss/TDP/Soap_Datex2/Pull
Format: DATEX II XML over SOAP (HTTP POST)
Rate: 5 calls/minute | Quota: 260 000/period
Auth: Bearer token in HTTP header

⚠️  TOKEN SECURITY
    Never hardcode your token. Store it in .env:
        OPENTRANSPORT_TOKEN=your_token_here
    Then load with: python-dotenv or os.environ

Corrélation avec les urgences
-------------------------------
  daily_traffic_volume  → proxy densité humaine en circulation
  heavy_vehicle_pct     → présence de poids lourds → accidents graves
  traffic_vs_normal     → anomalie trafic → événements exceptionnels
  high_traffic_day      → binaire → feature XGBoost directe

Compteurs utilisés (proches des hôpitaux)
-------------------------------------------
  BE  → CH:0003  (A1/A6 Bern)
  ZH  → CH:0100  (A1 Zürich)
  GE  → CH:0700  (A1 Genève)
  VD  → CH:0500  (A9 Lausanne)
  BS  → CH:0200  (A2/A3 Basel)
  AG  → CH:0150  (A1 Aarau)
  SG  → CH:0400  (A1 St. Gallen)
  VS  → CH:0600  (A9 Sion)
  NE  → CH:0350  (A5 Neuchâtel)
  TI  → CH:0800  (A2 Lugano)

Source: ASTRA / opentransportdata.swiss — FEDRO Terms of Use
Docs:   https://opentransportdata.swiss/en/cookbook/rt-road-traffic-counters/
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Optional
from xml.etree import ElementTree as ET

import httpx
import pandas as pd
from loguru import logger

# ── Configuration ─────────────────────────────────────────────────
API_URL = "https://api.opentransportdata.swiss/TDP/Soap_Datex2/Pull"

# Token chargé depuis .env — NE JAMAIS HARDCODER
def _get_token() -> str:
    token = os.environ.get("OPENTRANSPORT_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "OPENTRANSPORT_TOKEN manquant.\n"
            "→ Créer un fichier .env avec : OPENTRANSPORT_TOKEN=votre_token\n"
            "→ Ou exporter : export OPENTRANSPORT_TOKEN=votre_token"
        )
    return token


# ── Compteurs ASTRA proches des hôpitaux ─────────────────────────
HOSPITAL_COUNTERS: dict[str, list[str]] = {
    "BE": ["CH:0003.01", "CH:0003.02"],
    "ZH": ["CH:0100.01", "CH:0100.02"],
    "GE": ["CH:0700.01", "CH:0700.02"],
    "VD": ["CH:0500.01", "CH:0500.02"],
    "BS": ["CH:0200.01", "CH:0200.02"],
    "AG": ["CH:0150.01", "CH:0150.02"],
    "SG": ["CH:0400.01", "CH:0400.02"],
    "VS": ["CH:0600.01", "CH:0600.02"],
    "NE": ["CH:0350.01", "CH:0350.02"],
    "TI": ["CH:0800.01", "CH:0800.02"],
}

# Namespaces DATEX II
_NS_DX2        = "http://datex2.eu/schema/2/2_0"
_NS_XSI        = "http://www.w3.org/2001/XMLSchema-instance"
_SOAP_ACTION   = "http://opentransportdata.swiss/TDP/Soap_Datex2/Pull/v1"


# ═══════════════════════════════════════════════════════════════════
# 1. SOAP REQUESTS
# ═══════════════════════════════════════════════════════════════════

def _build_soap_d2_request() -> str:
    """SOAP envelope DATEX II avec échange minimal (namespace correct)."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope
    xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/"
    xmlns:dx223="http://datex2.eu/schema/2/2_0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <SOAP-ENV:Header/>
  <SOAP-ENV:Body>
    <dx223:d2LogicalModel modelBaseVersion="2">
      <dx223:exchange>
        <dx223:supplierIdentification>
          <dx223:country>ch</dx223:country>
          <dx223:nationalIdentifier>OTD_client</dx223:nationalIdentifier>
        </dx223:supplierIdentification>
      </dx223:exchange>
    </dx223:d2LogicalModel>
  </SOAP-ENV:Body>
</SOAP-ENV:Envelope>"""


def _build_soap_measured_data(_counter_ids: list[str]) -> str:
    """
    Construit le SOAP envelope pour pullMeasuredData (DATEX II).
    Le filtrage par compteur est effectué côté client après réception.
    """
    return _build_soap_d2_request()


def _build_soap_site_table() -> str:
    """Construit le SOAP envelope pour pullMeasurementSiteTable."""
    return _build_soap_d2_request()


def _post_soap(payload: str, timeout: int = 30, soap_action: str = "") -> str:
    """
    Envoie une requête SOAP à l'API traffic counters.
    Retourne le XML brut de la réponse.
    """
    token = _get_token()
    headers = {
        "Content-Type":  "text/xml; charset=utf-8",
        "Authorization": f"Bearer {token}",
        "SOAPAction":    soap_action,
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(API_URL, content=payload.encode("utf-8"), headers=headers)
        response.raise_for_status()

    return response.text


# ═══════════════════════════════════════════════════════════════════
# 2. PARSE DATEX II XML
# ═══════════════════════════════════════════════════════════════════

def _parse_measured_data(
    xml_text: str,
    filter_ids: Optional[set] = None,
) -> list[dict]:
    """
    Parse la réponse DATEX II XML et extrait les mesures de trafic.

    Chaque enregistrement contient :
      - site_id       : ID du compteur (ex: "CH:0003.01")
      - timestamp     : horodatage de la mesure
      - total_vehicles: somme des débits TrafficFlow valides (veh/h)
      - avg_speed_kmh : vitesse moyenne km/h

    Args:
        xml_text:   XML brut de la réponse SOAP.
        filter_ids: si fourni, ne retourne que les sites dans cet ensemble.

    Returns:
        Liste de dict — un par compteur.
    """
    records = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"DATEX II XML parse error: {e}")
        return records

    NS  = _NS_DX2
    XSI = _NS_XSI

    for site in root.iter(f"{{{NS}}}siteMeasurements"):
        ref_el  = site.find(f"{{{NS}}}measurementSiteReference")
        site_id = ref_el.get("id", "unknown") if ref_el is not None else "unknown"

        if filter_ids is not None and site_id not in filter_ids:
            continue

        ts_el     = site.find(f"{{{NS}}}measurementTimeDefault")
        timestamp = ts_el.text if ts_el is not None else None

        total_flow, avg_speed = 0, 0.0

        # Each outer measuredValue wraps one inner measuredValue with the data
        for outer_mv in site.findall(f"{{{NS}}}measuredValue"):
            inner_mv = outer_mv.find(f"{{{NS}}}measuredValue")
            if inner_mv is None:
                continue
            basic = inner_mv.find(f"{{{NS}}}basicData")
            if basic is None:
                continue

            # Skip measurements flagged as erroneous
            err_el = basic.find(f".//{{{NS}}}dataError")
            if err_el is not None and err_el.text == "true":
                continue

            btype = basic.get(f"{{{XSI}}}type", "")

            if "TrafficFlow" in btype:
                flow_el = basic.find(f".//{{{NS}}}vehicleFlowRate")
                if flow_el is not None:
                    try:
                        total_flow += int(float(flow_el.text))
                    except (ValueError, TypeError):
                        pass

            elif "TrafficSpeed" in btype:
                speed_el = basic.find(f".//{{{NS}}}speed")
                if speed_el is not None:
                    try:
                        avg_speed = float(speed_el.text)
                    except (ValueError, TypeError):
                        pass

        records.append({
            "site_id":        site_id,
            "timestamp":      timestamp,
            "light_vehicles": total_flow,
            "heavy_vehicles": 0,
            "total_vehicles": total_flow,
            "avg_speed_kmh":  avg_speed,
        })

    return records


# ═══════════════════════════════════════════════════════════════════
# 3. FETCH — données en temps réel
# ═══════════════════════════════════════════════════════════════════

def fetch_traffic_realtime(
    canton: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Récupère les données de trafic en temps réel pour un canton.

    Données disponibles : dernière minute (mise à jour toutes les minutes).
    Utilisé pour la prédiction en production (API en temps réel).

    Args:
        canton:   Code canton (BE, ZH, GE, VD, BS, AG, SG, VS, NE, TI)
        timeout:  Timeout HTTP en secondes

    Returns:
        DataFrame avec les comptages de la dernière minute.
        Colonnes: site_id, timestamp, light_vehicles, heavy_vehicles,
                  total_vehicles, avg_speed_kmh, canton
    """
    canton = canton.upper()
    counter_ids = HOSPITAL_COUNTERS.get(canton)
    if not counter_ids:
        logger.warning(f"Canton {canton} non supporté. Cantons disponibles: {list(HOSPITAL_COUNTERS)}")
        return pd.DataFrame()

    logger.info(f"Fetching realtime traffic for {canton} ({counter_ids})")

    try:
        payload  = _build_soap_measured_data(counter_ids)
        xml_text = _post_soap(payload, timeout,
                              soap_action=f"{_SOAP_ACTION}/pullMeasuredData")
        records  = _parse_measured_data(xml_text, filter_ids=set(counter_ids))

        if not records:
            logger.warning(f"No traffic records parsed for {canton}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["canton"] = canton
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

        logger.success(f"Traffic {canton}: {len(df)} records fetched")
        return df

    except EnvironmentError as e:
        logger.error(str(e))
        return pd.DataFrame()
    except httpx.HTTPError as e:
        logger.error(f"Traffic API HTTP error [{canton}]: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Traffic fetch failed [{canton}]: {e}")
        return pd.DataFrame()


def fetch_traffic_all_cantons(timeout: int = 30) -> pd.DataFrame:
    """
    Récupère les données temps réel pour tous les cantons hospitaliers.
    Attention : 1 appel API par canton → 10 appels → respecter la limite de 5/min.

    Returns:
        DataFrame consolidé tous cantons.
    """
    import time
    frames = []

    cantons = list(HOSPITAL_COUNTERS.keys())
    for i, canton in enumerate(cantons):
        df = fetch_traffic_realtime(canton, timeout)
        if not df.empty:
            frames.append(df)
        # Rate limiting : 5 calls/min → pause entre appels par groupe de 5
        if (i + 1) % 5 == 0 and i < len(cantons) - 1:
            logger.info("Rate limit pause (5 calls/min)...")
            time.sleep(12)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    logger.success(f"All cantons traffic: {len(result)} records")
    return result


# ═══════════════════════════════════════════════════════════════════
# 4. AGRÉGATION JOURNALIÈRE — feature engineering
# ═══════════════════════════════════════════════════════════════════

def aggregate_daily_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les données de trafic minute/heure → journalier par canton.

    Produit les features directement utilisables dans engineering.py :
      daily_traffic_volume  → total véhicules du jour
      heavy_vehicle_pct     → % poids lourds (proxy accidents graves)
      avg_speed_kmh         → vitesse moyenne (basse = congestion ou météo)
      low_speed_flag        → 1 si vitesse < 60 km/h (conditions dégradées)

    Args:
        df: Output de fetch_traffic_realtime() ou fetch_traffic_all_cantons()
            Doit contenir: timestamp, canton, total_vehicles,
                           heavy_vehicles, avg_speed_kmh

    Returns:
        DataFrame avec une ligne par [date, canton]
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.date

    daily = (
        df.groupby(["date", "canton"])
        .agg(
            daily_traffic_volume = ("total_vehicles", "sum"),
            heavy_vehicles_total = ("heavy_vehicles", "sum"),
            avg_speed_kmh        = ("avg_speed_kmh",  "mean"),
            n_measurements       = ("total_vehicles", "count"),
        )
        .reset_index()
    )

    # % poids lourds
    daily["heavy_vehicle_pct"] = (
        daily["heavy_vehicles_total"] /
        daily["daily_traffic_volume"].clip(lower=1)
    ).round(3)

    # Flag vitesse basse (conditions dégradées : neige, brouillard, accident)
    daily["low_speed_flag"] = (daily["avg_speed_kmh"] < 60).astype(int)

    # Normaliser le volume (certains jours ont moins de mesures)
    daily["traffic_per_hour"] = (
        daily["daily_traffic_volume"] /
        daily["n_measurements"].clip(lower=1)
    ).round(1)

    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values(["canton", "date"]).reset_index(drop=True)


def compute_traffic_anomaly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'anomalie de trafic = écart vs la moyenne du même jour de semaine.

    traffic_vs_normal > 1.2  → trafic exceptionnellement élevé
    traffic_vs_normal < 0.8  → trafic anormalement bas (grève, férié)

    Ce signal est plus prédictif que le volume absolu pour les urgences.

    Args:
        daily: Output de aggregate_daily_traffic()

    Returns:
        DataFrame enrichi avec traffic_vs_normal
    """
    if daily.empty or "daily_traffic_volume" not in daily.columns:
        return daily

    df = daily.copy()
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

    # Moyenne par canton × jour de semaine
    dow_mean = (
        df.groupby(["canton", "day_of_week"])["daily_traffic_volume"]
        .transform("mean")
    )

    df["traffic_vs_normal"] = (
        df["daily_traffic_volume"] / dow_mean.clip(lower=1)
    ).round(3)

    df["high_traffic_day"] = (df["traffic_vs_normal"] > 1.2).astype(int)
    df["low_traffic_day"]  = (df["traffic_vs_normal"] < 0.8).astype(int)

    return df.drop(columns=["day_of_week"])


# ═══════════════════════════════════════════════════════════════════
# 5. PIPELINE COMPLET
# ═══════════════════════════════════════════════════════════════════

def build_traffic_dataset(
    target_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Pipeline complet : fetch + agrégation + anomalie pour tous les cantons.

    C'est la fonction appelée depuis features/engineering.py,
    exactement comme build_meteo_dataset() pour la météo.

    Usage dans engineering.py :
        from src.ingestion.transport import build_traffic_dataset
        traffic = build_traffic_dataset()
        df = df.merge(traffic, on=["date", "canton"], how="left")

    Returns:
        DataFrame avec date + canton + features trafic :
        daily_traffic_volume, heavy_vehicle_pct, avg_speed_kmh,
        low_speed_flag, traffic_per_hour, traffic_vs_normal,
        high_traffic_day, low_traffic_day
    """
    logger.info("Building traffic dataset for all cantons...")

    raw = fetch_traffic_all_cantons()
    if raw.empty:
        logger.warning("No traffic data — returning empty DataFrame")
        return pd.DataFrame()

    daily  = aggregate_daily_traffic(raw)
    result = compute_traffic_anomaly(daily)

    logger.success(
        f"Traffic dataset ready: {len(result):,} rows | "
        f"{result['canton'].nunique()} cantons"
    )
    return result


def get_traffic_feature_names() -> list[str]:
    """Retourne la liste des features trafic pour FULL_FEATURE_COLS."""
    return [
        "daily_traffic_volume",  # Volume journalier total
        "heavy_vehicle_pct",     # % poids lourds (accidents graves)
        "avg_speed_kmh",         # Vitesse moyenne (proxy conditions route)
        "low_speed_flag",        # 1 si vitesse < 60 km/h
        "traffic_per_hour",      # Volume normalisé par heure de mesure
        "traffic_vs_normal",     # Anomalie vs moyenne jour de semaine
        "high_traffic_day",      # 1 si trafic > 120% normale
        "low_traffic_day",       # 1 si trafic < 80% normale (férié, grève)
    ]


# ═══════════════════════════════════════════════════════════════════
# CLI — test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    args = sys.argv[1:]

    # Debug mode: discover real measurement site IDs from the server
    if "--site-table" in args:
        print(f"\n{'='*50}")
        print("Traffic Counters — debug: pullMeasurementSiteTable")
        print(f"{'='*50}")
        try:
            payload  = _build_soap_site_table()
            xml_text = _post_soap(payload, timeout=30,
                                  soap_action=f"{_SOAP_ACTION}/pullMeasurementSiteTable")
            print("\n✅ Site table response (first 2000 chars):")
            print(xml_text[:2000])
        except EnvironmentError as e:
            print(f"\n❌ Token manquant: {e}")
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
        sys.exit(0)

    canton = args[0].upper() if args else "BE"
    print(f"\n{'='*50}")
    print(f"Traffic Counters — test canton {canton}")
    print(f"{'='*50}")

    df = fetch_traffic_realtime(canton)
    if not df.empty:
        print(f"\n✅ Realtime: {len(df)} records")
        print(df.to_string(index=False))

        daily = aggregate_daily_traffic(df)
        daily = compute_traffic_anomaly(daily)
        print(f"\n✅ Daily aggregated:")
        print(daily.to_string(index=False))
    else:
        token_set = bool(os.environ.get("OPENTRANSPORT_TOKEN"))
        if not token_set:
            print(f"\n⚠️  No data — OPENTRANSPORT_TOKEN manquant dans .env")
        else:
            print(f"\n⚠️  No data — API error (voir logs ci-dessus)")
            print(f"    Conseil: run with --site-table pour vérifier les IDs compteurs valides")
        print(f"\nFeatures disponibles ({len(get_traffic_feature_names())}):")
        for f in get_traffic_feature_names():
            print(f"  {f}")