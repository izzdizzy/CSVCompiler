#!/usr/bin/env python3
"""
combine_sensor_data.py

Combine sensor data from multiple ZIP files (each containing CSV files,
possibly in nested folders) into one clean CSV: combined.csv

Output columns:
    timestamp, kind, area, value, source_zip, source_csv

Also produces:
    processing_log.csv  - per-file processing status
    duplicates.csv      - conflicting rows (same timestamp/kind/area, different value)

Usage:
    python combine_sensor_data.py <input_folder_with_zips> [output_dir]

If the combined result is expected to exceed ~1,000,000 rows, the script
writes one CSV per kind instead (e.g. BTU.csv), each with columns:
    timestamp, area, value, source_zip, source_csv
"""

import argparse
import csv
import io
import os
import re
import sys
import zipfile

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Mapping of ZIP filename -> kind. Keys may be exact filenames ("sensors_btu.zip")
# or substrings matched case-insensitively against the ZIP name ("btu").
KIND_MAP = {
    "btu": "BTU",
    "temp": "Temperature",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "pressure": "Pressure",
    "flow": "Flow",
    "kwh": "kWh",
    "power": "Power",
    "voltage": "Voltage",
    "current": "Current",
}

# Columns that carry no useful data and should always be dropped.
IGNORE_COLUMNS = {"11", "alm disabled"}

# Header-detection keywords for the title/tag, value, date and time columns.
TITLE_KEYWORDS = ("title", "tag", "name", "descrip", "label", "point")
VALUE_KEYWORDS = ("value", "val", "reading", "measure")
DATE_KEYWORDS = ("date", "dia")
TIME_KEYWORDS = ("time", "hr", "hour")

# Columns used positionally when headers are missing/unclear.
POSITIONAL_COLUMNS = ["title", "value", "date", "time"]  # cols 0,1,2,3

# Threshold above which we split output per kind instead of one combined.csv.
MAX_COMBINED_ROWS = 1_000_000

OUTPUT_COLUMNS = ["timestamp", "kind", "area", "value", "source_zip", "source_csv"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def infer_kind(zip_name: str, log_fn) -> str:
    """Infer the 'kind' from a ZIP filename using KIND_MAP."""
    base = os.path.basename(zip_name)
    stem = re.sub(r"\.zip$", "", base, flags=re.IGNORECASE)

    # Exact match on full filename or stem (case-insensitive)
    lowered = {k.lower(): v for k, v in KIND_MAP.items()}
    for candidate in (base.lower(), stem.lower()):
        if candidate in lowered:
            return lowered[candidate]

    # Substring match on map keys
    for key, kind in lowered.items():
        if key in stem.lower():
            return kind

    # Fallback: use the uppercase stem itself only if it looks like a real label
    # (short alphanumeric token); otherwise unknown.
    log_fn(f"Could not infer kind for ZIP '{base}' -> kind='unknown'")
    return "unknown"


def looks_like_header(row):
    """Heuristic: a header row contains text keywords rather than data values."""
    if row is None or len(row) == 0:
        return False
    joined = " ".join(str(c).strip().lower() for c in row if c is not None)
    keywords = TITLE_KEYWORDS + VALUE_KEYWORDS + DATE_KEYWORDS + TIME_KEYWORDS
    hits = sum(1 for kw in keywords if kw in joined)
    # A genuine header typically matches at least two keyword families
    # (e.g. 'tag', 'value', 'date', 'time') and contains no digits-as-data
    return hits >= 2


def read_csv_bytes(raw: bytes):
    """Decode CSV bytes trying several encodings. Returns list of rows (list of lists)."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("latin-1", errors="replace")

    text = text.replace("\x00", "")
    rows = []
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel  # default comma
    for r in csv.reader(io.StringIO(text), dialect):
        rows.append(r)
    return rows


def find_column_indices(header_row):
    """Map header names to (title, value, date, time) column indices, or None."""
    norm = [str(c).strip().lower() for c in header_row]

    def search(keywords):
        for i, name in enumerate(norm):
            if name in IGNORE_COLUMNS:
                continue
            for kw in keywords:
                if kw in name:
                    return i
        return None

    idx = {
        "title": search(TITLE_KEYWORDS),
        "value": search(VALUE_KEYWORDS),
        "date": search(DATE_KEYWORDS),
        "time": search(TIME_KEYWORDS),
    }
    # Headers are only "clear" if value/date/time were all found.
    if idx["value"] is None or idx["date"] is None or idx["time"] is None:
        return None
    return idx


def parse_timestamp(dates, times):
    """Combine date and time Series into a datetime Series (NaT on failure)."""
    s = dates.astype(str).str.strip() + " " + times.astype(str).str.strip()
    ts = pd.to_datetime(s, errors="coerce")
    if ts.isna().any():
        # Retry with explicit formats commonly seen in sensor exports
        alt = pd.to_datetime(s, format="%m/%d/%y %I:%M:%S %p", errors="coerce")
        ts = ts.fillna(alt)
    return ts


def process_csv_rows(rows, kind, area_from_name, source_zip, source_csv, log_fn):
    """Turn raw CSV rows into a tidy DataFrame of output records."""
    if not rows:
        return None, 0, "empty file"

    # Strip whitespace in every cell
    rows = [[c.strip() if isinstance(c, str) else c for c in r] for r in rows]

    # Skip leading blank rows to find the first content row
    content_idx = next((i for i, r in enumerate(rows) if any(str(c).strip() for c in r)), None)
    if content_idx is None:
        return None, 0, "empty file"

    rows = rows[content_idx:]
    header_row = rows[0]
    has_header = looks_like_header(header_row)

    col_title = col_value = col_date = col_time = None
    if has_header:
        idx = find_column_indices(header_row)
        if idx is not None:
            col_title = idx["title"]
            col_value = idx["value"]
            col_date = idx["date"]
            col_time = idx["time"]
        else:
            # Headers exist but are unclear -> fall back to positions
            log_fn(f"{source_zip}/{source_csv}: headers unclear, using positional columns")
            has_header = False
    if not has_header:
        col_title, col_value, col_date, col_time = 0, 1, 2, 3

    data_rows = rows[1:] if has_header else rows
    width = max((len(r) for r in data_rows), default=0)
    needed = max(i for i in (col_title, col_value, col_date, col_time) if i is not None) + 1
    if width < needed:
        return None, 0, f"not enough columns (found {width}, need {needed})"

    def cell(r, i):
        return r[i] if i is not None and i < len(r) else ""

    recs = []
    for r in data_rows:
        if not any(str(c).strip() for c in r):
            continue  # blank row
        val_raw = str(cell(r, col_value)).strip()
        date_raw = str(cell(r, col_date)).strip()
        time_raw = str(cell(r, col_time)).strip()
        title_raw = str(cell(r, col_title)).strip() if col_title is not None else ""

        if val_raw == "" or date_raw == "" or time_raw == "":
            continue  # missing value/date/time, separator rows etc.
        try:
            value = float(val_raw)
        except ValueError:
            continue  # non-numeric ('???', 'N/A', ...)
        if value != value:  # NaN
            continue

        recs.append((title_raw, value, date_raw, time_raw))

    rows_read = len(data_rows)
    if not recs:
        return None, rows_read, "no valid data rows"

    df = pd.DataFrame(recs, columns=["title", "value", "date", "time"])
    df["timestamp"] = parse_timestamp(df["date"], df["time"])
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return None, rows_read, "all timestamps failed to parse"

    # Area determination
    if area_from_name:
        df["area"] = area_from_name
    else:
        titles = df["title"].astype(str).str.strip()
        unique_titles = titles[titles != ""].unique()
        if len(unique_titles) == 1:
            df["area"] = unique_titles[0]
        else:
            df["area"] = titles.replace("", source_csv)

    df["kind"] = kind
    df["source_zip"] = source_zip
    df["source_csv"] = source_csv
    df = df[OUTPUT_COLUMNS]
    return df, rows_read, None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", nargs="?", default=".",
                        help="Folder containing the ZIP files")
    parser.add_argument("output_dir", nargs="?", default=".",
                        help="Folder for combined.csv / logs (default: input folder)")
    args = parser.parse_args(argv)

    in_dir = os.path.abspath(args.input_folder)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir != "." else in_dir
    os.makedirs(out_dir, exist_ok=True)

    log_messages = []

    def log_fn(msg):
        log_messages.append(msg)
        print(f"[log] {msg}", file=sys.stderr)

    zip_names = sorted(
        f for f in os.listdir(in_dir) if f.lower().endswith(".zip")
    )
    if not zip_names:
        log_fn(f"No ZIP files found in {in_dir}")

    frames = []
    log_rows = []

    for zip_name in zip_names:  # deterministic order: sorted by filename
        zip_path = os.path.join(in_dir, zip_name)

        # ---- Corrupt ZIP handling: skip and continue (before inferring kind) ----
        try:
            zf = zipfile.ZipFile(zip_path)
        except (zipfile.BadZipFile, OSError) as e:
            log_fn(f"Skipping corrupt/unreadable ZIP {zip_name}: {e}")
            log_rows.append(dict(source_zip=zip_name, source_csv="", status="skipped",
                                 rows_read=0, rows_kept=0, rows_dropped=0,
                                 error_message=f"bad zip: {e}"))
            continue

        kind = infer_kind(zip_name, log_fn)
        with zf:
            csv_entries = sorted(
                e for e in zf.namelist()
                if e.lower().endswith(".csv") and not e.endswith("/")
            )
            if not csv_entries:
                log_rows.append(dict(source_zip=zip_name, source_csv="", status="no_csv",
                                     rows_read=0, rows_kept=0, rows_dropped=0,
                                     error_message="no CSV files inside zip"))

            for entry in csv_entries:
                inner_name = os.path.basename(entry)
                area_from_name = re.sub(r"\.csv$", "", inner_name, flags=re.IGNORECASE)
                if not area_from_name or not re.search(r"[A-Za-z0-9]", area_from_name):
                    area_from_name = None  # filename not useful

                status, err = "ok", ""
                rows_read = rows_kept = 0
                try:
                    raw = zf.read(entry)
                    rows = read_csv_bytes(raw)
                    df, rows_read, err = process_csv_rows(
                        rows, kind, area_from_name, zip_name, inner_name, log_fn
                    )
                    if df is None:
                        status = "skipped"
                        err = err or "no usable data"
                    else:
                        rows_kept = len(df)
                        frames.append(df)
                except Exception as e:  # never crash on a single bad file
                    status = "error"
                    err = f"{type(e).__name__}: {e}"
                    log_fn(f"Error reading {zip_name}/{inner_name}: {err}")

                log_rows.append(dict(
                    source_zip=zip_name,
                    source_csv=inner_name,
                    status=status,
                    rows_read=rows_read,
                    rows_kept=rows_kept,
                    rows_dropped=max(rows_read - rows_kept, 0),
                    error_message=err,
                ))

    # ---- Concatenate efficiently ----
    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=OUTPUT_COLUMNS)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")

    dup_log = pd.DataFrame(
        columns=["timestamp", "kind", "area", "value", "source_zip", "source_csv", "reason"]
    )

    if not combined.empty:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
        combined = combined.dropna(subset=["timestamp", "value"])

        key_cols = ["timestamp", "kind", "area"]

        # 1) Drop exact duplicates on (timestamp, kind, area, value)
        exact_dupes = combined.duplicated(subset=key_cols + ["value"], keep="first")
        combined = combined[~exact_dupes]

        # 2) Conflicting values for same (timestamp, kind, area): log them, keep last
        conflict_mask = combined.duplicated(subset=key_cols, keep=False)
        if conflict_mask.any():
            conflicts = combined[conflict_mask]
            # Within each conflicting group, everything except the last row is dropped
            grp = conflicts.groupby(key_cols, sort=False)
            last_idx = grp.cumcount(ascending=False)  # 0 marks the last row of each group
            drop_idx = conflicts.index[last_idx > 0]
            dup_log = conflicts.loc[drop_idx].copy()
            dup_log["reason"] = "conflicting value; kept last processed row"
            combined = combined.drop(index=drop_idx)

        # 3) Final sort
        combined = combined.sort_values(["timestamp", "kind", "area"],
                                        kind="mergesort").reset_index(drop=True)

    # ---- Write outputs ----
    combined_path = os.path.join(out_dir, "combined.csv")
    dup_path = os.path.join(out_dir, "duplicates.csv")
    log_path = os.path.join(out_dir, "processing_log.csv")

    if len(combined) > MAX_COMBINED_ROWS:
        # Too big: write one CSV per kind instead of a single combined.csv
        per_kind_cols = ["timestamp", "area", "value", "source_zip", "source_csv"]
        kinds_written = []
        for kind, grp in combined.groupby("kind", sort=True):
            fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(kind)) + ".csv"
            fpath = os.path.join(out_dir, fname)
            grp[per_kind_cols].to_csv(fpath, index=False)
            kinds_written.append(fname)
        # combined.csv becomes an index pointing at the per-kind files
        pd.DataFrame({"kind": [k for k in combined['kind'].unique()],
                      "file": [re.sub(r'[^A-Za-z0-9_.-]+', '_', str(k)) + '.csv'
                               for k in combined['kind'].unique()]}
                     ).to_csv(combined_path, index=False)
        log_fn(f"Combined size {len(combined)} > {MAX_COMBINED_ROWS}; "
               f"wrote per-kind files instead: {', '.join(kinds_written)}")
    else:
        combined[OUTPUT_COLUMNS].to_csv(combined_path, index=False)

    dup_log.to_csv(dup_path, index=False)
    pd.DataFrame(log_rows, columns=["source_zip", "source_csv", "status",
                                    "rows_read", "rows_kept", "rows_dropped",
                                    "error_message"]).to_csv(log_path, index=False)

    print(f"Done. {len(combined)} rows kept -> {combined_path}")
    print(f"Duplicates logged -> {dup_path} ({len(dup_log)} rows)")
    print(f"Processing log    -> {log_path} ({len(log_rows)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
