from pathlib import Path
import csv
import re
from collections import defaultdict

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
HISTORICAL_DIR = DATA_DIR / "historical"
SPIGES_FILE = DATA_DIR / "spiges_daily_aggregated.csv"

OUTPUT_FILE = DATA_DIR / "meteo_daily_by_canton.csv"

TEMP_AVG_CODE = "tre200d0"
TEMP_MAX_CODE = "tre200dx"
TEMP_MIN_CODE = "tre200dn"


def find_file(filename: str) -> Path:
    candidates = [
        BASE_DIR / filename,
        DATA_DIR / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(BASE_DIR.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {filename}. Put it either in project root or in data/."
    )


def clean_rtf_wrapped_csv_text(raw: str) -> str:
    if not raw.lstrip().startswith("{\\rtf"):
        return raw

    def decode_hex_escape(match):
        value = int(match.group(1), 16)
        return bytes([value]).decode("latin-1")

    raw = re.sub(r"\\'([0-9a-fA-F]{2})", decode_hex_escape, raw)

    possible_headers = [
        "station_abbr;",
        "parameter_shortname;",
    ]

    start_positions = [raw.find(h) for h in possible_headers if raw.find(h) != -1]
    if start_positions:
        raw = raw[min(start_positions):]

    raw = raw.replace("\\\n", "\n")
    raw = re.sub(r"\\[a-zA-Z]+\d* ?", "", raw)
    raw = raw.replace("{", "").replace("}", "")
    raw = raw.replace("\\", "")

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if ";" in line:
            lines.append(line)

    return "\n".join(lines) + "\n"


def read_semicolon_csv_maybe_rtf(path: Path):
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    cleaned = clean_rtf_wrapped_csv_text(raw)

    reader = csv.DictReader(cleaned.splitlines(), delimiter=";")
    rows = list(reader)

    if not rows:
        raise ValueError(f"No rows could be read from {path}")

    return rows


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def load_station_to_canton(stations_meta_file: Path):
    rows = read_semicolon_csv_maybe_rtf(stations_meta_file)

    mapping = {}

    for row in rows:
        station = row.get("station_abbr")
        canton = row.get("station_canton")

        if station and canton:
            mapping[station.strip().upper()] = canton.strip().upper()

    if not mapping:
        raise ValueError("Could not build station -> canton mapping.")

    return mapping


def load_parameter_descriptions(parameters_meta_file: Path):
    rows = read_semicolon_csv_maybe_rtf(parameters_meta_file)

    descriptions = {}

    for row in rows:
        code = row.get("parameter_shortname")
        if not code:
            continue

        descriptions[code.strip()] = {
            "de": row.get("parameter_description_de", ""),
            "en": row.get("parameter_description_en", ""),
            "unit": row.get("parameter_unit", ""),
        }

    return descriptions


def load_spiges_cantons():
    if not SPIGES_FILE.exists():
        print(f"Warning: {SPIGES_FILE} not found. Will export all cantons found in Meteo data.")
        return None

    rows = read_csv(SPIGES_FILE)
    cantons = sorted(
        {
            row.get("kanton_hospital", "").strip().upper()
            for row in rows
            if row.get("kanton_hospital", "").strip()
        }
    )

    if not cantons:
        print("Warning: Could not detect cantons from SpiGes file. Will export all cantons.")
        return None

    return set(cantons)


def main():
    stations_meta_file = find_file("ogd-smn_meta_stations.csv")
    parameters_meta_file = find_file("ogd-smn_meta_parameters.csv")

    print(f"Using station meta: {stations_meta_file}")
    print(f"Using parameter meta: {parameters_meta_file}")

    station_to_canton = load_station_to_canton(stations_meta_file)
    parameter_descriptions = load_parameter_descriptions(parameters_meta_file)
    spiges_cantons = load_spiges_cantons()

    if spiges_cantons:
        print()
        print("Restricting output to SpiGes cantons:")
        print(", ".join(sorted(spiges_cantons)))

    print()
    print("Temperature parameter mapping:")
    for code in [TEMP_AVG_CODE, TEMP_MAX_CODE, TEMP_MIN_CODE]:
        info = parameter_descriptions.get(code, {})
        print(
            f"  {code}: "
            f"{info.get('de', 'unknown')} / "
            f"{info.get('en', 'unknown')} "
            f"[{info.get('unit', '')}]"
        )

    if not HISTORICAL_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {HISTORICAL_DIR}")

    historical_files = sorted(HISTORICAL_DIR.rglob("*.csv"))

    if not historical_files:
        raise FileNotFoundError(f"No historical CSV files found under {HISTORICAL_DIR}")

    print()
    print(f"Found historical CSV files: {len(historical_files)}")

    # Key: (date, canton)
    # Value: list of station measurements for this canton-day.
    grouped = defaultdict(list)

    used_rows = 0
    skipped_rows = 0
    skipped_wrong_canton = 0
    unknown_station_rows = 0

    for path in historical_files:
        rows = read_csv(path)

        for row in rows:
            station = row.get("station_abbr", "").strip().upper()
            date = row.get("reference_timestamp", "").strip()

            if not station or not date:
                skipped_rows += 1
                continue

            canton = station_to_canton.get(station)

            if not canton:
                unknown_station_rows += 1
                continue

            if spiges_cantons and canton not in spiges_cantons:
                skipped_wrong_canton += 1
                continue

            temp_avg = to_float(row.get(TEMP_AVG_CODE))
            temp_max = to_float(row.get(TEMP_MAX_CODE))
            temp_min = to_float(row.get(TEMP_MIN_CODE))

            if temp_avg is None and temp_max is None and temp_min is None:
                skipped_rows += 1
                continue

            grouped[(date, canton)].append(
                {
                    "station_abbr": station,
                    "temperature_avg": temp_avg,
                    "temperature_max": temp_max,
                    "temperature_min": temp_min,
                }
            )

            used_rows += 1

    print()
    print(f"Used station-day rows: {used_rows}")
    print(f"Skipped empty/invalid rows: {skipped_rows}")
    print(f"Skipped rows outside SpiGes cantons: {skipped_wrong_canton}")
    print(f"Rows with unknown station code: {unknown_station_rows}")

    output_rows = []

    for (date, canton), values in sorted(grouped.items()):
        avg_values = [v["temperature_avg"] for v in values if v["temperature_avg"] is not None]
        max_values = [v["temperature_max"] for v in values if v["temperature_max"] is not None]
        min_values = [v["temperature_min"] for v in values if v["temperature_min"] is not None]

        station_abbrs = sorted({v["station_abbr"] for v in values})

        output_rows.append(
            {
                "date": date,
                "canton_abbr": canton,
                "temperature_avg": round(sum(avg_values) / len(avg_values), 3) if avg_values else "",
                "temperature_max": round(max(max_values), 3) if max_values else "",
                "temperature_min": round(min(min_values), 3) if min_values else "",
                "station_count": len(station_abbrs),
                "station_abbrs": "|".join(station_abbrs),
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "date",
            "canton_abbr",
            "temperature_avg",
            "temperature_max",
            "temperature_min",
            "station_count",
            "station_abbrs",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print(f"Wrote: {OUTPUT_FILE}")
    print(f"Output rows: {len(output_rows)}")

    print()
    print("Preview:")
    for row in output_rows[:20]:
        print(row)


if __name__ == "__main__":
    main()
