#!/usr/bin/env python3
"""
validation.py — Manual Validation engine for the standalone "Validation" tab.

This module validates a *manually uploaded* combined CSV file (or several of
them) against *manually uploaded* ZIP files containing the original per-area
CSV sources.  It deliberately shares NO state with the Process tab: every
input comes from the uploads handed to :func:`run_manual_validation`, so the
result is reproducible and independent of any previous processing run.

Design notes
------------
* Reuses the exact same parsing helpers as the main pipeline
  (``processor.read_csv_bytes``, ``processor.process_csv_rows``,
  ``processor.infer_kind``) so cleaning rules stay identical everywhere.
* The combined CSV is read in CHUNKS (``pd.read_csv(chunksize=...)`` over an
  in-memory BytesIO stream) so very large files never get materialised more
  than once; only the four comparison columns are kept.
* Corrupt ZIPs, empty CSVs, unreadable entries and invalid rows NEVER crash
  the run — they are counted and reported in dedicated tables.
* Row matching uses a composite key string built from
  ``timestamp | kind | area`` (+ ``source_zip | source_csv`` when the combined
  CSV carries those columns).  A small hash digest keeps the key compact.
* Numeric comparisons use a configurable tolerance (default 1e-6).

Public API:
    DEFAULT_VALIDATION_TOLERANCE   default numeric comparison tolerance
    VALIDATION_PREVIEW_ROWS        UI preview cap (imported by app.py)
    ManualValidationReport         result container (dataclass)
    clean_raw_dataframe            shared row-cleaning rules (requirement 6)
    read_combined_csv_chunks       chunked combined-CSV reader
    load_raw_from_zips             robust ZIP -> raw DataFrame loader
    run_manual_validation          full comparison pipeline (requirements 7/8)
    df_to_csv_bytes                helper to serialize report tables
"""

import hashlib
import io
import os
import re
import time
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Shared helpers from the main pipeline — identical cleaning rules everywhere.
from processor import (
    OUTPUT_COLUMNS,
    VALUE_KEYWORDS,
    _SEPARATOR_RE,
    infer_kind,
    parse_timestamp,
    process_csv_rows,
    read_csv_bytes,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default numeric tolerance for value comparisons (requirement 8: 0.000001).
DEFAULT_VALIDATION_TOLERANCE = 1e-6

#: Preview tables in the UI are capped at this many rows so the page never
#: freezes when a validation produces thousands of issues (requirement 10).
VALIDATION_PREVIEW_ROWS = 100

#: Rows per chunk when streaming a large combined CSV through pandas.
COMBINED_CHUNKSIZE = 100_000


@dataclass
class ManualValidationReport:
    """Outcome of comparing an uploaded combined CSV against uploaded ZIPs.

    Every ``*_df`` frame holds the FULL result set: the UI shows only the
    first VALIDATION_PREVIEW_ROWS rows but downloads the complete tables
    (requirement 11).
    """

    passed: bool = False
    total_combined_rows: int = 0     # valid data rows parsed from combined CSV
    total_raw_rows: int = 0          # cleaned valid rows parsed from ZIP CSVs
    missing_count: int = 0           # raw rows absent from the combined CSV
    unexpected_count: int = 0        # combined rows absent from the ZIP data
    mismatch_count: int = 0          # same key, values differ beyond tolerance
    conflict_count: int = 0          # duplicate-conflict groups (both sides)
    skipped_invalid_count: int = 0   # invalid rows skipped during cleaning
    skipped_files_count: int = 0     # files that produced no valid rows
    has_source_cols: bool = False    # combined CSV carried source_zip/source_csv?
    tolerance: float = DEFAULT_VALIDATION_TOLERANCE
    elapsed_seconds: float = 0.0
    missing_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    unexpected_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    mismatches_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    conflicts_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped_files_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped_rows_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    gaps_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: list = field(default_factory=list)   # human-readable problem lines
    notes: list = field(default_factory=list)    # informational lines


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _norm_key_str(v) -> str:
    """Normalise a kind/area cell to a comparable string ('' for missing)."""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "nat") else s


def _key_digest(series: pd.Series) -> pd.Series:
    """Compact fixed-width hash of a string Series (memory-friendly join key)."""
    return series.map(lambda s: hashlib.blake2b(
        s.encode("utf-8", "replace"), digest_size=16).hexdigest())


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a report table to CSV bytes for st.download_button."""
    buf = io.BytesIO()
    if df is None or df.empty:
        # Emit at least a header line so downloads are never zero-byte oddities.
        pd.DataFrame(columns=list(df.columns) if df is not None else []).to_csv(
            buf, index=False)
    else:
        df.to_csv(buf, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Cleaning rules shared by both sides (requirement 6)
# ---------------------------------------------------------------------------


def clean_raw_dataframe(df: pd.DataFrame):
    """Clean a raw parsed DataFrame into validated records + skipped rows.

    Rules applied (mirrors requirement 6):
      * drop separator/artifact rows (only dashes/pipes/broken table markers);
      * drop rows with a missing/unparseable timestamp (date+time combined
        first when a ``timestamp`` column is absent but date/time exist);
      * drop rows whose value is missing/blank/non-numeric ('???', 'N/A');
      * convert ``value`` to numeric;
      * remove EXACT duplicates (same timestamp, kind, area AND value);
      * flag keys (timestamp, kind, area) carrying DIFFERENT values as
        duplicate conflicts (first occurrence kept, variants reported).

    Returns ``(clean_df, skipped_df, conflicts_df)`` where ``clean_df`` has
    columns [timestamp, kind, area, value] (+ source_zip/source_csv
    passthrough when present), ``skipped_df`` lists every dropped row with a
    ``reason`` column, and ``conflicts_df`` lists all rows involved in a
    conflict group with a ``conflict_group`` id.
    """
    empty = pd.DataFrame(columns=["timestamp", "kind", "area", "value"])
    if df is None or df.empty:
        return empty, pd.DataFrame(), pd.DataFrame(
            columns=list(empty.columns) + ["conflict_group"])

    work = df.copy()
    for col in ("timestamp", "kind", "area", "value"):
        if col not in work.columns:
            work[col] = np.nan if col in ("timestamp", "value") else ""

    # ---- Combine date + time into timestamp where it is missing ----------
    ts_missing = work["timestamp"].isna() | (
        work["timestamp"].astype(str).str.strip().isin(["", "NaT", "None", "nan"]))
    if "date" in work.columns and "time" in work.columns and ts_missing.any():
        try:
            fixed = parse_timestamp(work.loc[ts_missing, "date"],
                                    work.loc[ts_missing, "time"])
            work.loc[ts_missing, "timestamp"] = fixed.values
        except Exception:
            pass  # leave NaT -> classified as invalid below
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")

    # ---- Normalise text key columns --------------------------------------
    work["kind"] = work["kind"].map(_norm_key_str)
    work["area"] = work["area"].map(_norm_key_str)

    # ---- Numeric value conversion ('???', blanks, NaN become invalid) ----
    work["_value_num"] = pd.to_numeric(work["value"], errors="coerce")

    # ---- Separator/artifact rows (dashes, pipes, broken table markers) ---
    joined = (work["timestamp"].astype(str) + " " + work["kind"].astype(str)
              + " " + work["area"].astype(str) + " " + work["value"].astype(str))
    is_sep = joined.str.replace(r"[|]", " ", regex=True).str.match(_SEPARATOR_RE)

    # ---- Classification ---------------------------------------------------
    reasons = np.select(
        [is_sep.values,
         work["timestamp"].isna().values,
         work["_value_num"].isna().values],
        ["separator/artifact row",
         "missing or unparseable timestamp",
         "missing or non-numeric value"],
        default="")
    work["_reason"] = reasons

    # Skipped rows keep their ORIGINAL columns + a reason (for reporting).
    bad_mask = work["_reason"] != ""
    orig_cols = [c for c in df.columns if c in work.columns]
    skipped = work.loc[bad_mask, orig_cols].copy()
    if not skipped.empty:
        skipped["reason"] = work.loc[bad_mask, "_reason"].values

    valid = work.loc[~bad_mask, ["timestamp", "kind", "area", "_value_num"] +
                    [c for c in ("source_zip", "source_csv") if c in work.columns]].copy()
    valid["value"] = valid["_value_num"]
    valid = valid.drop(columns=["_value_num"])

    key_cols = ["timestamp", "kind", "area"]

    # ---- Exact duplicates (same key AND same value) -> collapse safely ----
    exact_dupes = valid.duplicated(subset=key_cols + ["value"], keep="first")
    n_exact = int(exact_dupes.sum())
    deduped = valid[~exact_dupes]
    if n_exact:
        ex = valid[exact_dupes].copy()
        ex["reason"] = "exact duplicate row (same timestamp, kind, area, value)"
        skipped = pd.concat([skipped, ex], ignore_index=True)

    # ---- Duplicate conflicts: same key, DIFFERENT values ------------------
    extra_cols = [c for c in deduped.columns if c not in key_cols + ["value"]]
    conflicts_df = pd.DataFrame(
        columns=key_cols + ["value"] + extra_cols + ["conflict_group"])
    clean = deduped
    if not deduped.empty:
        conflict_mask = deduped.duplicated(subset=key_cols, keep=False)
        if conflict_mask.any():
            conflicts = deduped[conflict_mask].copy()
            grp_ids = conflicts.groupby(key_cols, sort=False,
                                        observed=True).ngroup() + 1
            conflicts_df = pd.concat([conflicts,
                                      grp_ids.rename("conflict_group")], axis=1)
            # Keep the FIRST occurrence per key in the cleaned set; the
            # remaining variants are reported as conflicts, never hidden.
            drop_idx = conflicts.index[grp_ids > 0]
            clean = deduped.drop(index=drop_idx)

    clean = clean.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
    return (clean[key_cols + ["value"]],
            skipped.reset_index(drop=True),
            conflicts_df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Combined CSV reader (chunked; requirements 4 & 12)
# ---------------------------------------------------------------------------


def read_combined_csv_chunks(upload, chunksize: int = COMBINED_CHUNKSIZE):
    """Yield cleaned chunks of an uploaded combined CSV (chunked reading).

    Expected columns: timestamp, kind, area, value, source_zip, source_csv.
    * ``source_zip`` / ``source_csv`` are OPTIONAL — when missing, callers
      validate on (timestamp, kind, area, value) only.
    * When ``timestamp`` is absent but ``date`` and ``time`` exist, they are
      combined into a single timestamp column.

    Raises ValueError for structural problems (empty file / unknown layout)
    so the caller can show a friendly message instead of crashing.
    """
    name = getattr(upload, "name", "combined csv")
    raw = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    if not raw or not raw.strip():
        raise ValueError(f"'{name}': file is empty")

    # ---- Probe the header to locate the expected columns ------------------
    f = io.BytesIO(raw)
    try:
        probe = pd.read_csv(f, nrows=5, dtype=str)
    except Exception as e:
        raise ValueError(f"'{name}': could not parse CSV header "
                         f"({type(e).__name__}: {e})")
    if probe.empty:
        raise ValueError(f"'{name}': no data rows found")

    cols = {str(c).strip().lower(): c for c in probe.columns}
    ts_col = cols.get("timestamp")
    date_col = cols.get("date")
    time_col = cols.get("time")
    if ts_col is None and not (date_col and time_col):
        raise ValueError(f"'{name}': expected a 'timestamp' column (or both "
                         "'date' and 'time'), plus 'kind', 'area' and 'value' "
                         "columns")
    kind_col = cols.get("kind")
    area_col = cols.get("area")
    val_col = next((cols[k] for k in ("value", "val", "reading") if k in cols), None)
    if val_col is None:
        raise ValueError(f"'{name}': missing 'value' column")
    src_zip_col = cols.get("source_zip")
    src_csv_col = cols.get("source_csv")

    usecols = [c for c in (ts_col, date_col, time_col, kind_col, area_col,
                           val_col, src_zip_col, src_csv_col) if c]
    rename = {}
    for orig, target in ((ts_col, "timestamp"), (date_col, "date"),
                         (time_col, "time"), (kind_col, "kind"),
                         (area_col, "area"), (val_col, "value"),
                         (src_zip_col, "source_zip"), (src_csv_col, "source_csv")):
        if orig and target != orig:
            rename[orig] = target

    # ---- Stream the actual data in chunks (single pass over the bytes) ----
    f = io.BytesIO(raw)
    reader = pd.read_csv(f, usecols=usecols, dtype=str, chunksize=chunksize)
    for chunk in reader:
        if rename:
            chunk = chunk.rename(columns=rename)
        # Optional columns may be entirely absent -> fill with ''.
        if "kind" not in chunk.columns:
            chunk["kind"] = ""
        if "area" not in chunk.columns:
            chunk["area"] = ""
        # Combine date+time into timestamp when there is no timestamp column.
        if "timestamp" not in chunk.columns:
            chunk["timestamp"] = parse_timestamp(chunk["date"], chunk["time"])
        yield chunk


# ---------------------------------------------------------------------------
# ZIP loader (robust; requirement 5)
# ---------------------------------------------------------------------------


def load_raw_from_zips(zip_items, progress=None):
    """Parse every CSV inside the uploaded ZIPs into one raw DataFrame.

    `zip_items` is an iterable of ``(zip_name, zip_bytes)`` pairs.
    Robustness guarantees:
      * corrupt ZIPs / unreadable entries never crash — they are recorded in
        ``skipped_df`` and parsing continues with the remaining files;
      * nested folders are supported; non-CSV entries are ignored;
      * empty CSVs and files with no usable rows are recorded as skipped;
      * each CSV goes through the SAME parser as the Process tab
        (``process_csv_rows``), extended with a single-column 'timestamp'
        fallback for raw exports that carry one combined timestamp.

    Live-progress contract (when `progress` is a ProgressTracker):
      * ``set_current_file()`` / ``finish_file()`` are called once per CSV so
        the UI's file-completion bar ticks after EVERY file — including ones
        that end up skipped or corrupted;
      * ``add_rows()`` reports scanned/kept rows per file (secondary info);
      * ``stage()`` lines are deduplicated by the tracker itself, so warnings
        like "Could not infer kind ..." appear exactly once per ZIP.

    Returns ``(raw_df, skipped_df, artifact_totals)``.
    """
    frames = []
    skipped_rows = []
    totals = dict(expected_rows=0, parsed_rows=0, blank_rows=0, separator_rows=0,
                  repeated_header_rows=0, bad_value_rows=0, bad_timestamp_rows=0,
                  malformed_rows=0)

    def log_fn(msg):
        # Parser chatter surfaces through the live stage log when provided.
        # ``ProgressTracker.stage`` deduplicates identical lines internally, so
        # a warning such as "Could not infer kind for ZIP ..." is shown ONCE
        # per ZIP (never repeated for every CSV inside that ZIP).  Without a
        # tracker nothing is printed — but never crash either.
        if progress is not None:
            try:
                progress.stage(msg)
            except Exception:
                pass

    def _file_done(status):
        """Advance the file-based progress bar (no-op without a tracker)."""
        if progress is not None:
            try:
                progress.finish_file(status=status)
            except Exception:
                pass

    items = sorted(zip_items, key=lambda t: str(t[0]))

    # ---- Pre-scan: count ALL CSV entries up front ------------------------
    # The exact total drives the primary progress bar (completed files /
    # total files) so it can be rendered before any parsing starts.  Corrupt
    # archives simply contribute 0 here and are reported during the main pass.
    if progress is not None:
        total_csvs = 0
        for _name, payload in items:
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as zf_pre:
                    total_csvs += sum(1 for finfo in zf_pre.infolist()
                                      if finfo.filename.lower().endswith(".csv")
                                      and not finfo.filename.endswith("/"))
            except Exception:
                continue
        if total_csvs:
            progress._total_files = total_csvs

    file_index = 0  # 1-based counter across all CSVs in all ZIPs ("file i of N")
    for zi, (zip_name, payload) in enumerate(items):
        zip_name = str(zip_name)
        if progress is not None:
            progress.stage(f"📦 Reading ZIP '{zip_name}' ({zi + 1}/{len(items)})…")
        try:
            zf = zipfile.ZipFile(io.BytesIO(payload))
        except Exception as e:
            # Corrupt/unreadable archive: report it, keep going (no crash).
            skipped_rows.append(dict(file=zip_name, entry="",
                                     reason=f"corrupt or unreadable ZIP: "
                                            f"{type(e).__name__}: {e}"))
            continue

        with zf:
            try:
                entries = sorted(e for e in zf.namelist()
                                 if e.lower().endswith(".csv") and not e.endswith("/"))
            except Exception as e:
                skipped_rows.append(dict(file=zip_name, entry="",
                                         reason=f"cannot list ZIP contents: {e}"))
                continue

            if not entries:
                skipped_rows.append(dict(file=zip_name, entry="",
                                         reason="no CSV files inside ZIP"))
                continue

            # Kind inferred ONCE per ZIP; the tracker's dedupe guarantees the
            # "unknown kind" warning is logged once per ZIP, not per CSV.
            kind = infer_kind(zip_name, log_fn)
            for entry in entries:
                inner_name = os.path.basename(entry)
                file_index += 1
                if progress is not None:
                    try:
                        progress.set_current_file(zip_name, inner_name, file_index)
                    except Exception:
                        pass
                status = "error"  # flipped to "ok"/"skipped" below
                area_from_name = re.sub(r"\.csv$", "", inner_name, flags=re.IGNORECASE)
                if not area_from_name or not re.search(r"[A-Za-z0-9]", area_from_name):
                    area_from_name = None  # filename not useful as area
                try:
                    raw_bytes = zf.read(entry)
                    rows = read_csv_bytes(raw_bytes)
                    # Detect a single combined 'timestamp' column (if any) in
                    # the first few rows — enables ts_col parsing mode.
                    ts_col = None
                    for r0 in rows[:5]:
                        norm0 = [str(c).strip().lower() for c in r0]
                        if "timestamp" in norm0:
                            ts_col = norm0.index("timestamp")
                            break
                    df, rows_read, err, art = process_csv_rows(
                        rows, kind, area_from_name, zip_name, inner_name,
                        log_fn, ts_col=ts_col, progress=progress)
                    for k in totals:
                        totals[k] += getattr(art, k)
                    if df is None or df.empty:
                        status = "skipped"
                        skipped_rows.append(dict(
                            file=zip_name, entry=entry,
                            reason=err or "empty file or no valid rows"))
                        continue
                    status = "ok"
                    frames.append(df)
                except Exception as e:
                    # A single unreadable entry must never abort the run.
                    status = "error"
                    skipped_rows.append(dict(
                        file=zip_name, entry=entry,
                        reason=f"error reading entry: {type(e).__name__}: {e}"))
                finally:
                    # One unit of work finished -> tick the file-based bar.
                    _file_done(status)

    if frames:
        raw = pd.concat(frames, ignore_index=True)
        del frames  # free the per-file frames immediately (flat memory)
    else:
        raw = pd.DataFrame(columns=OUTPUT_COLUMNS)
    return raw, pd.DataFrame(skipped_rows), totals


# ---------------------------------------------------------------------------
# Main entry point for the Validation tab
# ---------------------------------------------------------------------------


def run_manual_validation(combined_uploads, zip_items,
                          tolerance: float = DEFAULT_VALIDATION_TOLERANCE,
                          progress=None) -> ManualValidationReport:
    """Validate uploaded combined CSV file(s) against uploaded ZIP file(s).

    Stateless: uses ONLY the files handed to this call — never session state
    or results from the Process tab (requirement 3).

    Checks performed (requirement 7):
      * missing rows     — cleaned ZIP rows absent from the combined CSV;
      * unexpected rows  — combined rows with no counterpart in the ZIP data;
      * value mismatches — same (timestamp, kind, area), values differ beyond
                           the numeric tolerance (requirement 8);
      * duplicate conflicts — same key with different values (both sides);
      * skipped files / skipped invalid rows — cleaning accounting;
      * minute-level coverage gaps per (kind, area) — informational.

    Row matching uses the composite key (timestamp, kind, area):
      * when the combined CSV provides source_zip/source_csv columns, keys are
        scoped per source file (precise 1:1 comparison);
      * otherwise keys are compared globally, tolerating duplicates on either
        side (a combined row "matches" a raw row with the same key; value
        equality within tolerance decides match vs. mismatch).
    """
    t0 = time.monotonic()
    rep = ManualValidationReport(tolerance=float(tolerance))

    def stage(msg):
        if progress is not None:
            progress.stage(msg)

    # ==================================================================
    # 1) Parse the uploaded combined CSV(s) — chunked, memory-bounded.
    # ==================================================================
    comb_chunks = []
    comb_skipped = []          # invalid combined rows (per-row reasons)
    comb_total_valid = 0       # valid data rows found in combined files
    comb_total_invalid = 0     # rows dropped while cleaning combined files
    has_src_cols_seen = None   # do source_zip/source_csv columns exist?
    for up in combined_uploads:
        name = getattr(up, "name", str(up))
        try:
            for chunk in read_combined_csv_chunks(up):
                chunk["timestamp"] = pd.to_datetime(chunk["timestamp"],
                                                   errors="coerce")
                chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
                chunk["kind"] = chunk["kind"].map(_norm_key_str)
                chunk["area"] = chunk["area"].map(_norm_key_str)
                bad = chunk["timestamp"].isna() | chunk["value"].isna()
                if bad.any():
                    bad_rows = chunk[bad].copy()
                    bad_rows["reason"] = np.where(
                        bad_rows["timestamp"].isna(),
                        "missing/unparseable timestamp",
                        "missing/non-numeric value")
                    bad_rows.insert(0, "file", name)
                    comb_skipped.append(bad_rows)
                    comb_total_invalid += int(bad.sum())
                    chunk = chunk[~bad]
                if not chunk.empty:
                    cols = ["timestamp", "kind", "area", "value"]
                    extra = [c for c in ("source_zip", "source_csv")
                             if c in chunk.columns]
                    comb_chunks.append(chunk[cols + extra])
                    comb_total_valid += len(chunk)
                src_ok = {"source_zip", "source_csv"} <= set(chunk.columns)
                if has_src_cols_seen is None:
                    has_src_cols_seen = src_ok
                elif has_src_cols_seen and not src_ok:
                    # Mixed layouts across files: fall back to global matching.
                    has_src_cols_seen = False
                    stage("⚠️ Combined files have inconsistent source columns — "
                          "falling back to global key matching")
        except ValueError as e:
            # Structural problem (empty file / wrong columns): report clearly.
            rep.issues.append(str(e))
            comb_skipped.append(pd.DataFrame(
                [{"file": name, "reason": f"unreadable combined CSV: {e}"}]))
            stage(f"❌ {e}")
        except Exception as e:  # never crash the app on one bad upload
            rep.issues.append(f"Unexpected error reading '{name}': "
                              f"{type(e).__name__}: {e}")
            stage(f"❌ Unexpected error reading '{name}': {e}")

    rep.has_source_cols = bool(has_src_cols_seen)
    if comb_chunks:
        combined = pd.concat(comb_chunks, ignore_index=True)
        del comb_chunks
    else:
        combined = pd.DataFrame(columns=["timestamp", "kind", "area", "value"])
    if comb_skipped:
        skipped_comb = pd.concat(comb_skipped, ignore_index=True)
        del comb_skipped
    else:
        skipped_comb = pd.DataFrame()
    stage(f"Combined CSV parsed: {comb_total_valid:,} valid row(s), "
          f"{comb_total_invalid:,} invalid/skipped row(s)")

    # ==================================================================
    # 2) Parse + clean the raw data from the uploaded ZIPs.
    # ==================================================================
    stage("Reading uploaded ZIP archives…")
    raw_all, skipped_zip_files, zip_artifacts = load_raw_from_zips(
        zip_items, progress=progress)
    stage(f"ZIP scan finished: {len(raw_all):,} pre-clean row(s) from "
          f"{len(zip_items)} ZIP(s)")

    # Apply the shared cleaning rules (requirement 6) to the raw ZIP data.
    raw_clean, skipped_raw_rows, raw_conflicts = clean_raw_dataframe(raw_all)
    del raw_all  # release the pre-clean frame as soon as possible
    rep.total_raw_rows = int(len(raw_clean))

    # ==================================================================
    # 3) Compare combined vs cleaned raw on the composite key.
    # ==================================================================
    tol = abs(float(tolerance))
    scope_cols = ["source_zip", "source_csv"] if rep.has_source_cols else []
    key_cols = ["timestamp", "kind", "area"] + scope_cols

    def make_keys(df: pd.DataFrame) -> pd.Series:
        """Composite key string -> hashed digest Series for cheap joins."""
        ts = pd.to_datetime(df["timestamp"], errors="coerce").astype(str)
        parts = [ts.fillna(""), df["kind"].astype(str), df["area"].astype(str)]
        for c in scope_cols:
            parts.append(df[c].fillna("").astype(str))
        return _key_digest(pd.concat(parts, axis=1).agg("|".join, axis=1))

    raw_view = raw_clean.copy()
    raw_view["_key"] = make_keys(raw_view)
    comb_view = combined.copy()
    comb_view["_key"] = make_keys(comb_view)

    # ---- Raw-side duplicate conflicts (already grouped by cleaner) -------
    conf_frames = []
    if not raw_conflicts.empty:
        rc = raw_conflicts.copy()
        rc["side"] = "zip raw data"
        conf_frames.append(rc)

    # ---- Group by key and reconcile values within tolerance --------------
    group_cols = ["_key"] + key_cols
    raw_g = (raw_view.groupby(group_cols, sort=False, observed=True)["value"]
             .agg(list).reset_index())
    comb_g = (comb_view.groupby(group_cols, sort=False, observed=True)["value"]
              .agg(list).reset_index())

    merged = raw_g.merge(comb_g, on="_key", how="outer", suffixes=("_raw", "_comb"))

    missing_rows, unexpected_rows, mismatch_rows = [], [], []
    for _, mrow in merged.iterrows():
        rv = mrow["value_raw"] if isinstance(mrow["value_raw"], list) else []
        cv = mrow["value_comb"] if isinstance(mrow["value_comb"], list) else []
        base = {c: mrow.get(c) for c in key_cols}
        if not cv:
            # Key exists only in the raw ZIP data -> MISSING from combined.
            for v in rv:
                missing_rows.append({**base, "value": v})
        elif not rv:
            # Key exists only in the combined CSV -> UNEXPECTED row.
            for v in cv:
                unexpected_rows.append({**base, "value": v})
        else:
            # Greedy value reconciliation within tolerance: pair every raw
            # value with an equal-enough combined value; leftovers become
            # mismatches (both partners listed for context).
            pool = list(cv)
            for v in rv:
                hit = next((i for i, w in enumerate(pool)
                            if abs(float(v) - float(w)) <= tol), None)
                if hit is None:
                    mismatch_rows.append({**base, "raw_value": v,
                                          "combined_value": None})
                else:
                    pool.pop(hit)
            for w in pool:
                mismatch_rows.append({**base, "raw_value": None,
                                      "combined_value": w})

    rep.missing_df = pd.DataFrame(missing_rows,
                                  columns=key_cols + ["value"])
    rep.unexpected_df = pd.DataFrame(unexpected_rows,
                                     columns=key_cols + ["value"])
    rep.mismatches_df = pd.DataFrame(mismatch_rows,
                                     columns=key_cols + ["raw_value",
                                                         "combined_value"])

    # ---- Combined-side duplicate conflicts (same key, different values) ---
    if not combined.empty:
        cc_dup_mask = combined.duplicated(subset=["timestamp", "kind", "area"],
                                          keep=False)
        if cc_dup_mask.any():
            cc = combined[cc_dup_mask].copy()
            grp = cc.groupby(["timestamp", "kind", "area"], sort=False,
                             observed=True).ngroup() + 1
            # Only genuine VALUE conflicts count (identical values are just
            # exact duplicates, tolerated in global matching mode).
            vary = (cc.groupby(["timestamp", "kind", "area"], observed=True)["value"]
                    .transform(lambda s: s.nunique() > 1))
            cc = cc[vary.astype(bool)].copy()
            if not cc.empty:
                cc["conflict_group"] = grp[cc.index]
                cc["side"] = "combined csv"
                conf_frames.append(cc[["timestamp", "kind", "area", "value",
                                       "conflict_group", "side"]])

    if conf_frames:
        cd = pd.concat(conf_frames, ignore_index=True)
        # Re-number conflict groups sequentially (raw side first, then the
        # combined side) so group ids stay unique across both sources.
        offset = 0
        new_ids = []
        for _, gdf in cd.groupby("side", sort=False):
            rel = gdf["conflict_group"].rank(method="dense").astype(int) + offset
            new_ids.append(rel.rename(gdf.index.name))
            offset += int(gdf["conflict_group"].max())
        cd["conflict_group"] = pd.concat(new_ids).sort_index().values
        rep.conflicts_df = cd
    rep.conflict_count = int(rep.conflicts_df["conflict_group"].nunique()) \
        if not rep.conflicts_df.empty else 0

    # ---- Missing source files (uploaded but produced no valid rows) ------
    skipped_files = skipped_zip_files
    if not skipped_comb.empty and "reason" in skipped_comb.columns:
        file_level = skipped_comb[skipped_comb["reason"].str.startswith(
            "unreadable combined CSV", na=False)]
        if not file_level.empty:
            skipped_files = pd.concat(
                [skipped_files,
                 file_level.rename(columns={"file": "file"})[
                     [c for c in ("file", "reason") if c in file_level.columns]]],
                ignore_index=True)
    rep.skipped_files_df = skipped_files
    rep.skipped_files_count = int(len(skipped_files))

    # ---- Skipped invalid rows (both sides, one table) --------------------
    skip_parts = []
    if not skipped_raw_rows.empty:
        sr = skipped_raw_rows.copy()
        sr.insert(0, "side", "zip raw data")
        skip_parts.append(sr)
    if not skipped_comb.empty:
        sc = skipped_comb.copy()
        sc.insert(0, "side", "combined csv")
        skip_parts.append(sc)
    rep.skipped_rows_df = (pd.concat(skip_parts, ignore_index=True)
                           if skip_parts else pd.DataFrame())
    rep.skipped_invalid_count = int(len(rep.skipped_rows_df))

    # ---- Minute-level continuity gaps per (kind, area) -------------------
    # Informational check on the CLEANED RAW data (the reference dataset).
    gap_rows = []
    if not raw_clean.empty:
        notes_gap = ("Minute-gap analysis assumes one row per minute per "
                     "(kind, area); gaps are informational, not failures.")
        rep.notes.append(notes_gap)
        one_min = 60 * 1_000_000_000  # ns in one minute
        ts_ns = (raw_clean["timestamp"].values.astype("datetime64[ns]")
                 .astype("int64"))
        kinds = raw_clean["kind"].astype(str).values
        areas = raw_clean["area"].astype(str).values
        groups = pd.Series(range(len(raw_clean))).groupby([kinds, areas], sort=True)
        for (k, a), idx in groups:
            t = np.sort(ts_ns[idx.values])
            if len(t) < 2:
                continue
            expected_minutes = (int(t[-1]) - int(t[0])) // one_min + 1
            actual_minutes = len(np.unique(t // one_min))
            missing_minutes = expected_minutes - actual_minutes
            if missing_minutes > 0:
                gap_rows.append(dict(kind=k, area=a,
                                     min_timestamp=pd.Timestamp(int(t[0])),
                                     max_timestamp=pd.Timestamp(int(t[-1])),
                                     minutes_present=int(actual_minutes),
                                     minutes_expected=int(expected_minutes),
                                     missing_minutes=int(missing_minutes)))
        rep.gaps_df = pd.DataFrame(gap_rows)
        if gap_rows:
            tot = sum(g["missing_minutes"] for g in gap_rows)
            rep.notes.append(f"Found {tot:,} missing minute(s) across "
                             f"{len(gap_rows)} (kind, area) series — likely "
                             "sensor dropout in the source data, not a "
                             "combination bug.")

    # ==================================================================
    # 4) Counters, issues and summary table.
    # ==================================================================
    rep.total_combined_rows = int(comb_total_valid)
    rep.missing_count = int(len(rep.missing_df))
    rep.unexpected_count = int(len(rep.unexpected_df))
    rep.mismatch_count = int(len(rep.mismatches_df))

    if not combined_uploads:
        rep.issues.append("No combined CSV files were uploaded.")
    if not zip_items:
        rep.issues.append("No ZIP files were uploaded.")
    if rep.missing_count:
        rep.issues.append(f"{rep.missing_count:,} row(s) present in the cleaned "
                          "ZIP data are MISSING from the combined CSV.")
    if rep.unexpected_count:
        rep.issues.append(f"{rep.unexpected_count:,} row(s) in the combined CSV "
                          "were NOT found in the uploaded ZIP data.")
    if rep.mismatch_count:
        rep.issues.append(f"{rep.mismatch_count:,} value mismatch(es) — same "
                          f"(timestamp, kind, area) but different values beyond "
                          f"tolerance {tol:g}.")
    if rep.conflict_count:
        rep.issues.append(f"{rep.conflict_count} duplicate-conflict group(s): "
                          "same (timestamp, kind, area) with different values.")
    if rep.skipped_files_count:
        rep.issues.append(f"{rep.skipped_files_count} file(s) were uploaded but "
                          "produced no valid rows (see Skipped files).")
    if rep.skipped_invalid_count:
        rep.notes.append(f"{rep.skipped_invalid_count:,} invalid row(s) were "
                         "skipped during cleaning (see Skipped invalid rows).")
    if rep.has_source_cols:
        rep.notes.append("Combined CSV carries source_zip/source_csv columns — "
                         "rows matched per source file (strict 1:1 comparison).")
    else:
        rep.notes.append("Combined CSV lacks source_zip/source_csv columns — "
                         "rows matched globally on (timestamp, kind, area, value).")
    if zip_artifacts:
        rep.notes.append(
            "ZIP parse artifacts: separators {}, repeated headers {}, bad "
            "values {}, bad timestamps {}, malformed/blank {}.".format(
                zip_artifacts["separator_rows"],
                zip_artifacts["repeated_header_rows"],
                zip_artifacts["bad_value_rows"],
                zip_artifacts["bad_timestamp_rows"],
                zip_artifacts["blank_rows"] + zip_artifacts["malformed_rows"]))

    rep.passed = not rep.issues

    rep.summary_df = pd.DataFrame([
        dict(metric="Pass/Fail status", value="PASS" if rep.passed else "FAIL"),
        dict(metric="Combined CSV rows uploaded (valid)", value=rep.total_combined_rows),
        dict(metric="Cleaned raw rows from ZIP files", value=rep.total_raw_rows),
        dict(metric="Missing rows (in ZIP, not in combined)", value=rep.missing_count),
        dict(metric="Unexpected rows (in combined, not in ZIP)", value=rep.unexpected_count),
        dict(metric="Value mismatches", value=rep.mismatch_count),
        dict(metric="Duplicate conflict groups", value=rep.conflict_count),
        dict(metric="Skipped invalid rows", value=rep.skipped_invalid_count),
        dict(metric="Skipped files", value=rep.skipped_files_count),
        dict(metric="Numeric tolerance", value=rep.tolerance),
        dict(metric="Source columns present in combined CSV",
             value="yes" if rep.has_source_cols else "no"),
        dict(metric="Validation duration (s)",
             value=round(time.monotonic() - t0, 3)),
    ])
    rep.elapsed_seconds = time.monotonic() - t0
    return rep
