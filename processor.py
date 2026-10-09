#!/usr/bin/env python3
"""
processor.py

Data cleaning / combining logic for the ZIP -> combined CSV tool.

This module is deliberately UI-free: it only exposes ``process_zips()``, which
takes an iterable of ``(zip_name, zip_bytes)`` pairs and returns a tidy result
object (see ``ProcessResult``).  The Streamlit UI lives in ``app.py`` and the
original command-line pipeline lives in ``combine_sensor_data.py``; both reuse
the helpers defined here so the parsing rules stay identical everywhere.

Output columns:
    timestamp, kind, area, value, source_zip, source_csv

Processing rules (kept in sync with combine_sensor_data.py):
 1. Every ZIP is processed in sorted-filename order (deterministic).
 2. Only ``.csv`` entries are read (nested folders supported); other file
    types are never loaded/parsed.
 3. Header detection: if the first row looks like a header
    (``tag/title/name``, ``value/reading``, ``date``, ``time`` keywords),
    columns are mapped by name — extra columns such as ``11`` and
    ``Alm Disabled`` are ignored.  If headers are missing/unclear, positional
    columns are used: ``0=title/tag, 1=value, 2=date, 3=time``.
 4. Blank rows, repeated header-like rows and page-break/separator rows are
    skipped.
 5. Date + time are parsed into one ``timestamp``
    (``pd.to_datetime(..., errors="coerce")``; formats like ``9/19/26`` and
    ``1:23:00 PM``); unparseable rows are dropped.
 6. Values are converted to numeric; non-numeric values (``???``, ``N/A``,
    blanks, ...) are dropped.
 7. ``kind`` is inferred from the ZIP filename via ``KIND_MAP``; otherwise
    ``kind = "unknown"``.
 8. ``area`` = CSV filename without extension; if that isn't useful, the
    title/tag column when it uniquely identifies the area; otherwise
    ``source_csv``.
 9. Duplicate handling on ``(timestamp, kind, area)`` is configurable:
    ``keep last`` (default), ``keep first``, or ``keep all``.
10. Error handling: corrupt ZIPs, missing/empty/unreadable CSVs never crash
    the run — they are logged as warnings/errors in the returned summary.
"""

import csv
import io
import os
import re
import threading
import time
import zipfile
from collections import deque
from dataclasses import dataclass, field

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

OUTPUT_COLUMNS = ["timestamp", "kind", "area", "value", "source_zip", "source_csv"]

# Allowed duplicate-handling modes (exposed in the UI dropdown).
DUPLICATE_MODES = ("keep last", "keep first", "keep all")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ProcessResult:
    """Everything the UI needs to show after a processing run."""

    combined: pd.DataFrame = field(default_factory=pd.DataFrame)   # final tidy rows
    per_file_log: list = field(default_factory=list)               # dict per CSV/ZIP
    messages: list = field(default_factory=list)                   # human-readable notes
    errors: list = field(default_factory=list)                     # hard failures
    zips_processed: int = 0                                        # ZIPs opened OK
    zips_failed: int = 0                                           # corrupt ZIPs
    csv_files_found: int = 0                                       # .csv entries seen
    csv_files_processed: int = 0                                   # CSVs that produced rows
    csv_files_skipped: int = 0                                     # CSVs skipped/errored
    rows_read: int = 0                                             # raw data rows seen
    rows_kept: int = 0                                             # rows in `combined`
    rows_dropped: int = 0                                          # invalid/duplicate rows
    duplicates_removed: int = 0                                    # rows removed by dedupe policy
    elapsed_seconds: float = 0.0                                   # total wall-clock processing time
    cancelled: bool = False                                        # True if the user pressed cancel


class CancelledError(Exception):
    """Raised internally when the user requests cancellation mid-run."""


# ---------------------------------------------------------------------------
# Live progress tracking (thread-safe, UI-agnostic)
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Thread-safe snapshot of live processing progress + ETA estimation.

    Design notes
    ------------
    * The processor may run on a *background worker thread* while the
      Streamlit script thread polls this object — hence every mutation is
      guarded by a lock and readers always get an immutable ``dict`` copy.
    * ETA uses a *moving average* over the last few speed samples:
        - If row counts are available for completed files -> estimate from
          rows/second against the projected total number of rows.
        - Otherwise -> fall back to files/second against the total file count.
        - If neither basis is reliable (too little data, unknown totals),
          ``eta_seconds`` stays ``None`` so the UI can show an *indeterminate*
          progress state instead of failing or lying about the ETA.
    """

    #: Number of recent speed samples kept for the moving average.
    MOVING_WINDOW = 8

    def __init__(self, total_files: int = 0, total_rows_hint: int = 0):
        self._lock = threading.Lock()
        self._cancel = threading.Event()

        # ---- static-ish totals (may be refined once known) ----
        self._total_files = max(int(total_files or 0), 0)   # CSV files expected
        self._total_rows_hint = max(int(total_rows_hint or 0), 0)  # 0 => unknown

        # ---- live counters ----
        self.files_done = 0            # CSV files finished (processed/skipped/failed)
        self.rows_seen = 0             # raw data rows scanned so far
        self.rows_kept = 0             # valid rows collected so far
        self.rows_dropped = 0          # invalid rows dropped so far
        self.current_zip = ""          # ZIP being processed right now
        self.current_csv = ""          # CSV being processed right now
        self.current_rows = 0          # rows scanned in the current CSV
        self.started_at = time.monotonic()

        # ---- moving-average speed samples: (rows_delta, seconds_delta) ----
        self._samples = deque(maxlen=self.MOVING_WINDOW)
        self._last_rows = 0            # rows_seen at previous sample point
        self._last_time = self.started_at

        # Most recent non-zero ETA estimate + its basis — kept so the UI can
        # report "estimated time accuracy" after the run finishes.
        self.last_eta_seconds = None
        self.last_eta_basis = None

    # -- cooperative cancellation -------------------------------------
    def cancel(self):
        """Request cancellation; the worker checks this between units of work."""
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    # -- updates (worker thread) ---------------------------------------
    def set_current_file(self, zip_name: str, csv_name: str, file_index: int):
        """Mark which file is being processed (1-based `file_index`)."""
        with self._lock:
            self.current_zip = zip_name
            self.current_csv = csv_name
            self.files_done = max(self.files_done, file_index - 1)
            self.current_rows = 0

    def add_rows(self, read: int, kept: int, dropped: int):
        """Record per-row progress for the current file (chunked updates)."""
        with self._lock:
            self.rows_seen += read
            self.rows_kept += kept
            # Clamp running totals to rows actually seen so an early
            # size-based estimate can never make "dropped" go negative or
            # "kept" exceed what was scanned.
            self.rows_kept = min(self.rows_kept, self.rows_seen)
            self.rows_dropped = max(self.rows_seen - self.rows_kept, 0)
            self.current_rows += read

    def finish_file(self):
        """Mark the current file complete and take one speed sample."""
        now = time.monotonic()
        with self._lock:
            self.files_done += 1
            dt = now - self._last_time
            drows = self.rows_seen - self._last_rows
            if dt > 0:
                # Sample even when drows == 0 (empty/bad files still cost time);
                # rows component simply contributes zero throughput.
                self._samples.append((drows, self.files_done, dt))
            self._last_time = now
            self._last_rows = self.rows_seen
            self.current_rows = 0
            # Once every file is done there is nothing left to estimate —
            # drop any stale ETA so the UI shows a clean finishing state.
            if self._total_files and self.files_done >= self._total_files:
                self._samples.clear()
                self._last_rows = self.rows_seen

    def note_total_rows(self, total_rows: int):
        """Refine the projected total row count once it becomes known (0 = unknown)."""
        with self._lock:
            if total_rows and total_rows >= self.rows_seen:
                self._total_rows_hint = int(total_rows)

    def _avg_rows_per_sec(self):
        """Moving average of rows/sec over recent samples; None if no samples."""
        if not self._samples:
            return None
        rows = sum(r for r, _, t in self._samples)
        secs = sum(t for _, _, t in self._samples)
        if secs <= 0:
            return None
        return rows / secs

    def snapshot(self) -> dict:
        """Return an immutable dict describing the current progress state."""
        with self._lock:
            elapsed = time.monotonic() - self.started_at
            avg_rps = self._avg_rows_per_sec()

            # --- fraction & ETA ---
            fraction = None
            eta = None
            basis = None  # what the ETA was computed from ("rows" / "files" / None)

            all_files_done = bool(self._total_files
                                  and self.files_done >= self._total_files)

            if self._total_rows_hint > 0 and avg_rps and avg_rps > 0:
                # Rows basis: preferred when both totals and speed are known.
                # The hint is only a rough size-based estimate, so never let
                # observed progress fall below it — use whichever is larger.
                total = max(self._total_rows_hint, self.rows_seen)
                remaining_rows = max(total - self.rows_seen, 0)
                fraction = min(self.rows_seen / total, 1.0)
                eta = remaining_rows / avg_rps
                basis = "rows"
            elif self._total_files > 0:
                # Files basis: works even without row counts.
                done = min(self.files_done, self._total_files)
                fraction = done / self._total_files
                files_rate = None
                secs = sum(t for _, _, t in self._samples)
                n_samples = len(self._samples)
                if secs > 0 and n_samples > 0:
                    files_rate = n_samples / secs  # each sample == one finished file
                if files_rate and files_rate > 0:
                    eta = max(self._total_files - done, 0) / files_rate
                    basis = "files"
                else:
                    eta = None  # indeterminate: not enough timing data yet
            # else: totals unknown -> indeterminate (fraction=None, eta=None)

            # Whole run finished -> deterministic terminal state.
            if all_files_done:
                fraction = 1.0
                eta = 0.0 if basis else None

            # Remember the last meaningful ETA for the post-run accuracy note.
            if eta is not None and eta > 0:
                self.last_eta_seconds = eta
                self.last_eta_basis = basis

            return {
                "elapsed": elapsed,
                "fraction": fraction,          # None => indeterminate progress
                "eta_seconds": eta,            # None => cannot estimate yet
                "eta_basis": basis,
                "files_done": self.files_done,
                "total_files": self._total_files,
                "rows_seen": self.rows_seen,
                "rows_kept": self.rows_kept,
                "rows_dropped": self.rows_dropped,
                "current_zip": self.current_zip,
                "current_csv": self.current_csv,
                "current_rows": self.current_rows,
                "rows_per_sec": avg_rps,       # None until first sample exists
                "total_rows_hint": self._total_rows_hint,  # 0 => unknown
                "cancelled": self._cancel.is_set(),
            }


def format_duration(seconds) -> str:
    """Human-friendly duration string ('12s', '3m 05s', '1h 02m'); '' for None."""
    if seconds is None:
        return ""
    seconds = max(int(round(seconds)), 0)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def infer_kind(zip_name: str, log_fn=lambda msg: None) -> str:
    """Infer the 'kind' from a ZIP filename using KIND_MAP ('unknown' fallback)."""
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

    # Cannot be inferred -> "unknown" (and leave a trace for the summary)
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
    # Fast path: the common sensor-export layout "9/19/26 1:23:00 PM".
    ts = pd.to_datetime(s, format="%m/%d/%y %I:%M:%S %p", errors="coerce")
    if ts.isna().any():
        # Retry only the failures with flexible parsing; unparseable stays NaT.
        alt = pd.to_datetime(s, errors="coerce", format=None)
        ts = ts.fillna(alt)
    return ts


def process_csv_rows(rows, kind, area_from_name, source_zip, source_csv, log_fn,
                     progress=None):
    """Turn raw CSV rows into a tidy DataFrame of output records.

    Returns ``(df_or_None, rows_read, error_message_or_None)``.
    Invalid rows (blank, repeated header/page-break, bad date/time,
    non-numeric value, unexpected layout) are skipped safely.

    `progress` (optional): a ``ProgressTracker``.  When given, the row loop
    runs in *chunks* so live counters (rows read/kept/dropped for the current
    file) update continuously and cancellation can stop mid-file — this keeps
    memory flat and the UI responsive even on very large CSVs.
    """
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

    # ---- Chunked row scan ----------------------------------------------
    # Large CSVs are walked in chunks so live progress (rows read/kept/
    # dropped for the current file) keeps updating and a cancel request can
    # stop the scan mid-file.  With no tracker, CHUNK stays huge and the
    # loop behaves exactly like the original single pass.
    CHUNK = 2000 if progress is not None else len(data_rows) + 1
    recs = []
    rows_read = 0
    rows_kept_chunk = 0
    for start in range(0, len(data_rows), CHUNK):
        if progress is not None and progress.is_cancelled():
            raise CancelledError("cancelled while reading "
                                 f"{source_zip}/{source_csv}")
        for r in data_rows[start:start + CHUNK]:
            if not any(str(c).strip() for c in r):
                continue  # blank row
            val_raw = str(cell(r, col_value)).strip()
            date_raw = str(cell(r, col_date)).strip()
            time_raw = str(cell(r, col_time)).strip()
            title_raw = str(cell(r, col_title)).strip() if col_title is not None else ""

            # Repeated header-like rows inside the data (page breaks) -> skip
            if looks_like_header([title_raw, val_raw, date_raw, time_raw]) and \
                    not _is_number(val_raw):
                continue

            if val_raw == "" or date_raw == "" or time_raw == "":
                continue  # missing value/date/time, separator rows etc.
            try:
                value = float(val_raw)
            except ValueError:
                continue  # non-numeric ('???', 'N/A', ...)
            if value != value:  # NaN
                continue

            recs.append((title_raw, value, date_raw, time_raw))
            rows_kept_chunk += 1

        rows_read = min(start + CHUNK, len(data_rows))
        if progress is not None:
            # Report this chunk's counts (kept here = valid-parsable rows;
            # final kept may still shrink after timestamp parsing / dedupe).
            progress.add_rows(len(data_rows[start:start + CHUNK]),
                              rows_kept_chunk,
                              rows_read - rows_kept_chunk)
            rows_kept_chunk = 0  # report deltas only, avoid double counting

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


def _is_number(s):
    """True if string parses as a finite float (used to spot repeated headers)."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------


def apply_duplicate_policy(df: pd.DataFrame, mode: str, log_fn=lambda msg: None):
    """De-duplicate on (timestamp, kind, area) according to `mode`.

    - "keep last":  exact dupes collapsed, then for conflicting values the
      last processed row wins (matches the CLI script's behaviour).
    - "keep first": same, but the first processed row wins.
    - "keep all":   nothing removed (raw combined view).

    Returns ``(df_after, n_removed)``.
    """
    if df.empty or mode == "keep all":
        return df, 0

    key_cols = ["timestamp", "kind", "area"]

    # 1) Drop *exact* duplicates (same key AND same value) — pure noise,
    #    regardless of keep-first/keep-last preference.
    before = len(df)
    df = df[~df.duplicated(subset=key_cols + ["value"], keep="first")]

    # 2) Conflicting values for the same (timestamp, kind, area):
    #    resolve according to the chosen policy.
    conflict_mask = df.duplicated(subset=key_cols, keep=False)
    if conflict_mask.any():
        conflicts = df[conflict_mask]
        grp = conflicts.groupby(key_cols, sort=False)
        # cumcount marks position within each group; keep the chosen side.
        seq = grp.cumcount(ascending=(mode != "keep first"))
        drop_idx = conflicts.index[seq > 0]
        df = df.drop(index=drop_idx)
        log_fn(f"Duplicates ({mode}): removed {len(drop_idx)} conflicting row(s) "
               f"on (timestamp, kind, area)")

    return df.reset_index(drop=True), before - len(df)


# ---------------------------------------------------------------------------
# Main entry point used by the UI (and reusable elsewhere)
# ---------------------------------------------------------------------------


def process_zips(zip_items, duplicate_mode: str = "keep last",
                 progress: ProgressTracker = None) -> ProcessResult:
    """Process an iterable of ``(zip_name, zip_bytes)`` pairs.

    `zip_items` may also be plain paths to ZIP files on disk; anything that
    cannot be opened is reported as an error instead of crashing the run.

    `progress` (optional): a ``ProgressTracker`` for live UI updates.  When
    given:
      * a cheap pre-scan counts every CSV entry and estimates the total row
        count from compressed sizes (so the progress bar/ETA can be based on
        rows/sec when possible, files/sec otherwise);
      * current file name + per-chunk row counters are reported continuously;
      * cancellation is honoured between files and within large files.
    """
    t0 = time.monotonic()
    result = ProcessResult()
    messages = result.messages

    def log_fn(msg):
        messages.append(msg)

    frames = []          # collected per-CSV DataFrames, concatenated once at the end
    pre_dedupe_rows = 0  # rows kept before duplicate resolution

    # Deterministic ordering makes runs reproducible.
    items = sorted(zip_items, key=lambda t: str(t[0]))

    # ---- Pre-scan: total CSV file count + rough total-row estimate --------
    # Uses only ZIP central-directory metadata (no decompression), so it is
    # fast even for big archives.  Estimates feed the ETA machinery; if the
    # estimate is unavailable the tracker falls back to files/sec.
    if progress is not None:
        total_csvs = 0
        est_rows = 0
        for _name, payload in items:
            try:
                if isinstance(payload, (str, os.PathLike)):
                    # Path on disk: open directly from the file (no full read).
                    with zipfile.ZipFile(os.fspath(payload)) as zf_pre:
                        infos = zf_pre.infolist()
                else:
                    with zipfile.ZipFile(io.BytesIO(payload)) as zf_pre:
                        infos = zf_pre.infolist()
            except Exception:
                continue  # corrupt ZIP: skipped now, reported properly later
            for info in infos:
                fn = info.filename
                if fn.lower().endswith(".csv") and not fn.endswith("/"):
                    total_csvs += 1
                    # ~25 bytes per typical sensor data row; header rows negligible.
                    est_rows += max(info.compress_size // 25, 1)
        if total_csvs:
            progress._total_files = total_csvs
        if est_rows:
            progress.note_total_rows(est_rows)

    file_index = 0  # 1-based counter of CSV files seen across all ZIPs (for "file i of N")
    try:
        for zip_name, payload in items:
            zip_name = str(zip_name)

            # ---- Cooperative cancellation between ZIP files ----
            if progress is not None and progress.is_cancelled():
                raise CancelledError("cancelled before " + zip_name)

            # ---- Accept either in-memory bytes or a filesystem path ----
            if isinstance(payload, (str, os.PathLike)):
                path = os.fspath(payload)
                zip_name = os.path.basename(path)
                try:
                    with open(path, "rb") as fh:
                        payload = fh.read()
                except OSError as e:
                    result.errors.append(f"Cannot read ZIP file {zip_name}: {e}")
                    result.zips_failed += 1
                    continue

            # ---- Corrupt ZIP handling: skip and continue ----
            try:
                zf = zipfile.ZipFile(io.BytesIO(payload))
            except (zipfile.BadZipFile, OSError) as e:
                result.errors.append(f"Skipping corrupt/unreadable ZIP {zip_name}: {e}")
                result.per_file_log.append(dict(source_zip=zip_name, source_csv="",
                                                status="skipped", rows_read=0, rows_kept=0,
                                                rows_dropped=0, error_message=f"bad zip: {e}"))
                result.zips_failed += 1
                continue

            result.zips_processed += 1
            kind = infer_kind(zip_name, log_fn)

            with zf:
                # Only load supported (.csv) entries; nested folders are fine,
                # directories themselves are filtered out.
                csv_entries = sorted(
                    e for e in zf.namelist()
                    if e.lower().endswith(".csv") and not e.endswith("/")
                )
                if not csv_entries:
                    result.errors.append(f"No CSV files found inside ZIP {zip_name}")
                    result.per_file_log.append(dict(source_zip=zip_name, source_csv="",
                                                    status="no_csv", rows_read=0, rows_kept=0,
                                                    rows_dropped=0,
                                                    error_message="no CSV files inside zip"))

                for entry in csv_entries:
                    result.csv_files_found += 1
                    file_index += 1
                    inner_name = os.path.basename(entry)
                    area_from_name = re.sub(r"\.csv$", "", inner_name, flags=re.IGNORECASE)
                    if not area_from_name or not re.search(r"[A-Za-z0-9]", area_from_name):
                        area_from_name = None  # filename not useful

                    # Tell the UI which file is being processed right now.
                    if progress is not None:
                        progress.set_current_file(zip_name, inner_name, file_index)

                    status, err = "ok", ""
                    rows_read = rows_kept = 0
                    try:
                        raw = zf.read(entry)
                        rows = read_csv_bytes(raw)
                        df, rows_read, err = process_csv_rows(
                            rows, kind, area_from_name, zip_name, inner_name, log_fn,
                            progress=progress,
                        )
                        if df is None:
                            status = "skipped"
                            err = err or "no usable data"
                            result.csv_files_skipped += 1
                            if rows_read == 0:
                                result.messages.append(
                                    f"{zip_name}/{inner_name}: empty or unusable CSV — skipped")
                        else:
                            rows_kept = len(df)
                            frames.append(df)
                            result.csv_files_processed += 1
                    except CancelledError:
                        # User pressed cancel — let it bubble to the handler below.
                        raise
                    except Exception as e:  # never crash on a single bad file
                        status = "error"
                        err = f"{type(e).__name__}: {e}"
                        result.csv_files_skipped += 1
                        result.errors.append(f"Error reading {zip_name}/{inner_name}: {err}")

                    # One unit of work finished -> speed sample for the ETA.
                    if progress is not None:
                        progress.finish_file()

                    result.per_file_log.append(dict(
                        source_zip=zip_name,
                        source_csv=inner_name,
                        status=status,
                        rows_read=rows_read,
                        rows_kept=rows_kept,
                        rows_dropped=max(rows_read - rows_kept, 0),
                        error_message=err,
                    ))
    except CancelledError as e:
        # ---- Cancellation: return whatever was collected so far ----
        result.cancelled = True
        result.errors.append(f"Processing cancelled by user ({e})")
        if progress is not None:
            snap = progress.snapshot()
            result.messages.append(
                f"Cancelled after {snap['files_done']}/{snap['total_files']} file(s), "
                f"{snap['rows_seen']:,} row(s) scanned.")
        frames = []  # partial output would be misleading — deliver an empty table
        combined = pd.DataFrame(columns=OUTPUT_COLUMNS)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        result.combined = combined
        result.rows_read = sum(r["rows_read"] for r in result.per_file_log)
        result.rows_kept = 0
        result.rows_dropped = result.rows_read
        result.elapsed_seconds = time.monotonic() - t0
        return result

    # ---- Concatenate efficiently: one pass over all collected frames ----
    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=OUTPUT_COLUMNS)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")

    if not combined.empty:
        # Final safety net: coerce types and drop any remaining invalid rows
        # (invalid dates/times or non-numeric values that slipped through).
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
        before_safety = len(combined)
        combined = combined.dropna(subset=["timestamp", "value"])
        n_safety = before_safety - len(combined)
        if n_safety:
            log_fn(f"Final safety filter dropped {n_safety} row(s) with unparseable "
                   "timestamps or values")
        combined = combined.sort_values(["timestamp", "kind", "area", "source_zip",
                                         "source_csv"], kind="mergesort").reset_index(drop=True)

    pre_dedupe_rows = len(combined)

    # ---- Duplicate resolution (UI-configurable) ----
    combined, removed = apply_duplicate_policy(combined, duplicate_mode, log_fn)
    combined = combined[OUTPUT_COLUMNS].reset_index(drop=True)

    result.combined = combined
    result.duplicates_removed = int(removed)
    result.rows_read = sum(r["rows_read"] for r in result.per_file_log)
    result.rows_kept = len(combined)
    result.rows_dropped = max(result.rows_read - result.rows_kept, 0)
    result.elapsed_seconds = time.monotonic() - t0
    # Sanity note when the per-file scan and final table disagree (dupes).
    if removed:
        log_fn(f"Removed {removed} duplicate row(s) after concatenation "
               f"(pre-dedupe rows: {pre_dedupe_rows})")
    return result
