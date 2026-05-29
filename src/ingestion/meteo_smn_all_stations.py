# MeteoSwiss open data: https://www.meteoswiss.admin.ch/services-and-publications/service/open-data.html
# https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/<station>/ogd-smn_<station>_<g>_<f>.csv
# g : granularity, t, h, d, m, or y
# f : update frequency; historical, recent, or now

from pystac_client import Client
import pandas as pd
from io import BytesIO
import requests
from pathlib import Path


STAC_ROOT  = "https://data.geo.admin.ch/api/stac/v1"

COLLECTION_OGD_SMN = "ch.meteoschweiz.ogd-smn"          # automatic weather stations --> temperature, precipitation
GRANULARITY = "d"                                       # daily
FREQUENCY   = "historical"                              # "historical" | "recent" | "now"

# COLLECTION_OGD_SMN_PRECIP = "ch.meteoschweiz.ogd-smn-precip"
COLLECTION_OGD_POLLEN = "ch.meteoschweiz.ogd-pollen"


def read_smn_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(BytesIO(r.content), sep=";", encoding="cp1252")
    df.columns = df.columns.str.lower()           # normalize across hist/recent
    return df

def fetch_all_stations(granularity=GRANULARITY, frequency=FREQUENCY):
    cat = Client.open(STAC_ROOT)
    search = cat.search(collections=[COLLECTION_OGD_SMN])

    frames = []
    for item in search.items():
        station = item.id.lower()
        asset_key = f"ogd-smn_{station}_{granularity}_{frequency}.csv"
        asset = item.assets.get(asset_key)
        if asset is None:
            continue   # not every station provides every granularity/frequency
        df = read_smn_csv(asset.href)
        df["station_abbr"] = station.upper()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fetch_and_save_all(out_dir="data/historical", granularity=GRANULARITY, frequency=FREQUENCY):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cat = Client.open(STAC_ROOT)

    for item in cat.search(collections=[COLLECTION_OGD_SMN]).items():
        station = item.id.lower()
        asset_key = f"ogd-smn_{station}_{granularity}_{frequency}.csv"
        asset = item.assets.get(asset_key)
        if asset is None:
            continue
        path = out / f"{station}_{granularity}_{frequency}.csv"
        if path.exists():
            # skip e.g. if we ran before and failed at some point, can skip already existing --> only works like that if we download data once, if we continuously update it, that's wrong
            continue               
        df = read_smn_csv(asset.href)
        df.to_csv(path, index=False)
        print(f"  {station} → {path} ({len(df)} rows)")


def fetch_one_station(station_abbr, granularity=GRANULARITY, frequency=FREQUENCY):
    cat = Client.open(STAC_ROOT)
    search = cat.search(collections=[COLLECTION_OGD_SMN])
    target = station_abbr.lower()

    for item in search.items():
        if item.id.lower() != target:
            continue
        asset_key = f"ogd-smn_{target}_{granularity}_{frequency}.csv"
        asset = item.assets.get(asset_key)
        if asset is None:
            raise KeyError(
                f"No asset {asset_key}; available: {list(item.assets)[:5]}…"
            )
        df = read_smn_csv(asset.href)
        df["station_abbr"] = station_abbr.upper()
        return df
    raise LookupError(f"Station {station_abbr!r} not found in collection")

if __name__ == "__main__":
    TEST = False
    if TEST:
        df = fetch_one_station("SMA")    # Zürich / Fluntern
        print(df.shape)
        print(df.tail())
    else:
        fetch_and_save_all()
#        latest = fetch_all_stations()
 #       print(latest.shape, latest["station_abbr"].nunique(), "stations")

