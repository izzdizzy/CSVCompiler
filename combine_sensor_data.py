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
import os
import re
import sys
import zipfile  # still used by main() to open ZIPs from the input folder

import pandas as pd

# The shared cleaning/parsing logic now lives in processor.py so the Streamlit
# UI (app.py) and this CLI script stay in sync.  We re-export the constants and
# helpers below for backwards compatibility with anything importing them here.
from processor import (  # noqa: F401  (re-exported on purpose)
    KIND_MAP,
    IGNORE_COLUMNS,
    TITLE_KEYWORDS,
    VALUE_KEYWORDS,
    DATE_KEYWORDS,
    TIME_KEYWORDS,
    POSITIONAL_COLUMNS,
    OUTPUT_COLUMNS,
    infer_kind,
    looks_like_header,
    read_csv_bytes,
    find_column_indices,
    parse_timestamp,
    process_csv_rows,
)

# Threshold above which we split output per kind instead of one combined.csv.
MAX_COMBINED_ROWS = 1_000_000

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
                    # NOTE: the UI-facing processor.process_csv_rows now returns a
                    # 4-tuple (adds artifact accounting); this CLI keeps the old
                    # 3-tuple contract via the compatibility wrapper below.
                    from processor import process_csv_rows_legacy as _parse_rows
                    df, rows_read, err = _parse_rows(
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
