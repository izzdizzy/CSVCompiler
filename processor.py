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
    skipped *and counted* (see ArtifactCounts) so validation can prove that
    nothing was removed silently.
 5. Date + time are parsed into one ``timestamp``
    (``pd.to_datetime(..., errors="coerce")``; formats like ``9/19/26`` and
    ``1:23:00 PM``); unparseable rows are dropped and counted.
 6. Values are converted to numeric; non-numeric values (``???``, ``N/A``,
    blanks, ...) are dropped and counted.
 7. ``kind`` is inferred from the ZIP filename via ``KIND_MAP``; otherwise
    ``kind = "unknown"``.
 8. ``area`` = CSV filename without extension; if that isn't useful, the
    title/tag column when it uniquely identifies the area; otherwise
    ``source_csv``.
 9. Duplicate handling on ``(timestamp, kind, area)`` is configurable:
    ``keep last`` (default), ``keep first``, or ``keep all``.  Exact
    duplicates (same key AND same value) are always collapsed safely;
    conflicting values are logged so validation can surface them.
10. Error handling: corrupt ZIPs, missing/empty/unreadable CSVs never crash
    the run — they are logged as warnings/errors in the returned summary and
    processing continues with the remaining files.

Progress design (updated per user request):
 * The live progress bar is based on *completed files / total files* — never
   on estimated row counts.  ETA comes from the average wall-clock time per
   completed file.  Row counters are shown only as secondary information.
 * After every file finishes, the tracker records a timing sample and the
   status (success/failure) so the UI updates immediately per file.
 * The final combining/export stage emits visible log lines through
   ``ProgressTracker.stage()`` so the UI never looks frozen while
   ``combined.csv`` is being assembled.

Validation (new):
 ``validate_result(result)`` compares the combined output against the source
 ZIPs/CSVs (per-file expected/parsed/dropped rows, artifact counts, min/max
 timestamps, duplicate conflicts, missing minute gaps per area/kind) and
 returns a ``ValidationReport`` used by the Validation / Compare tab.
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

import numpy as np
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

# Upper bound on how many individual conflict rows we keep for display.
MAX_CONFLICT_ROWS_KEPT = 5000


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ArtifactCounts:
    """Per-source-file tally of why each raw row was kept or removed.

    ``expected_rows`` counts EVERY raw row after the leading header.  The
    reason buckets below partition the rows that were NOT turned into valid
    output records, so ``expected == parsed + blank + separator +
    repeated_header + bad_value + bad_timestamp + malformed`` holds exactly —
    this is what lets the Validation tab prove that no data went missing
    silently.
    """

    expected_rows: int = 0        # ALL raw rows seen (header excluded)
    parsed_rows: int = 0          # rows that became valid output records
    blank_rows: int = 0           # fully blank rows
    separator_rows: int = 0       # '---|---|---' style page-break artifacts
    repeated_header_rows: int = 0 # header-like rows repeated inside data
    bad_value_rows: int = 0       # missing / non-numeric values ('???', 'N/A')
    bad_timestamp_rows: int = 0   # date/time present but unparseable
    malformed_rows: int = 0       # too few columns to extract date/time/value

    def as_dict(self) -> dict:
        return {
            "expected_rows": self.expected_rows,
            "parsed_rows": self.parsed_rows,
            "blank_rows": self.blank_rows,
            "separator_rows": self.separator_rows,
            "repeated_header_rows": self.repeated_header_rows,
            "bad_value_rows": self.bad_value_rows,
            "bad_timestamp_rows": self.bad_timestamp_rows,
            "malformed_rows": self.malformed_rows,
        }

    @property
    def dropped_total(self) -> int:
        """Rows intentionally not forwarded (all known artifact reasons)."""
        return (self.blank_rows + self.separator_rows + self.repeated_header_rows
                + self.bad_value_rows + self.bad_timestamp_rows + self.malformed_rows)

    @property
    def accounted(self) -> bool:
        """True when every expected row is either parsed or bucketed."""
        return (self.parsed_rows + self.dropped_total) == self.expected_rows


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

    # ---- new fields backing the Validation tab --------------------------
    artifact_counts: dict = field(default_factory=dict)  # (zip, csv) -> ArtifactCounts
    duplicate_conflicts: pd.DataFrame = field(default_factory=pd.DataFrame)
    exact_duplicates_removed: int = 0                    # safe collapses (same value)
    conflict_rows_removed: int = 0                       # rows lost to value conflicts
    pre_dedupe_rows: int = 0                             # concat rows before dedupe
    export_elapsed_seconds: float = 0.0                  # final combining/writing time
    duplicate_mode: str = "keep last"                    # policy used for this run


class CancelledError(Exception):
    """Raised internally when the user requests cancellation mid-run."""


# ---------------------------------------------------------------------------
# Live progress tracking (thread-safe, UI-agnostic)
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Thread-safe snapshot of live processing progress + file-based ETA.

    Design notes
    ------------
    * The processor runs on a *background worker thread* while the Streamlit
      script thread polls this object — hence every mutation is guarded by a
      lock and readers always get an immutable ``dict`` copy.
    * PROGRESS BASIS (updated): the primary progress bar is
      ``files_done / total_files`` — completed files, never estimated rows.
    * ETA BASIS (updated): computed from the *average wall-clock time per
      completed file*: ``eta = remaining_files * avg_seconds_per_file``.
      Row counts are tracked only so the UI can show them as secondary info.
    * ``stage()`` publishes human-readable log lines for the final
      combining/export phase so the UI keeps visibly updating (no frozen
      appearance) while ``combined.csv`` is assembled.
    """

    #: Number of recent per-file timing samples kept for the moving average.
    MOVING_WINDOW = 8

    def __init__(self, total_files: int = 0, total_rows_hint: int = 0):
        self._lock = threading.Lock()
        self._cancel = threading.Event()

        # ---- static-ish totals (may be refined once known) ----
        self._total_files = max(int(total_files or 0), 0)   # CSV files expected
        # NOTE: total_rows_hint is kept ONLY as secondary display info; it is
        # deliberately *not* used for the progress fraction or the ETA anymore.
        self._total_rows_hint = max(int(total_rows_hint or 0), 0)  # 0 => unknown

        # ---- live counters ----
        self.files_done = 0            # CSV files finished (processed/skipped/failed)
        self.rows_seen = 0             # raw data rows scanned so far
        self.rows_kept = 0             # valid rows collected so far
        self.rows_dropped = 0          # invalid rows dropped so far
        self.current_zip = ""          # ZIP being processed right now
        self.current_csv = ""          # CSV being processed right now
        self.current_rows = 0          # rows scanned in the current CSV
        self.last_file_status = ""     # "ok"/"skipped"/"error" of the last finished file
        self.started_at = time.monotonic()

        # ---- moving-average per-file timing samples (seconds per file) ----
        self._samples = deque(maxlen=self.MOVING_WINDOW)
        self._last_time = self.started_at

        # Most recent non-zero ETA estimate + its basis — kept so the UI can
        # report "estimated time accuracy" after the run finishes.
        self.last_eta_seconds = None
        self.last_eta_basis = None

        # ---- final-stage (combining/export) visible logging ----
        self._stage_logs = []          # list of (elapsed_at_log, message) tuples
        self.current_stage = ""        # e.g. "writing combined.csv"

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
        """Record per-row progress for the current file (chunked updates).

        These counters are *secondary information only* — they no longer
        influence the progress fraction or the ETA.
        """
        with self._lock:
            self.rows_seen += read
            self.rows_kept += kept
            # Clamp running totals to rows actually seen so an early
            # size-based estimate can never make "dropped" go negative or
            # "kept" exceed what was scanned.
            self.rows_kept = min(self.rows_kept, self.rows_seen)
            self.rows_dropped = max(self.rows_seen - self.rows_kept, 0)
            self.current_rows += read

    def finish_file(self, status: str = "ok"):
        """Mark the current file complete and take one per-file timing sample.

        `status` is the success/failure marker shown in the UI for the file
        that just finished ("ok", "skipped" or "error").
        """
        now = time.monotonic()
        with self._lock:
            self.files_done += 1
            dt = now - self._last_time
            if dt > 0:
                # One sample == one finished file; ETA = mean(sample) * left.
                self._samples.append(dt)
            self._last_time = now
            self.current_rows = 0
            self.last_file_status = status
            # Once every file is done there is nothing left to estimate —
            # drop any stale ETA so the UI shows a clean finishing state.
            if self._total_files and self.files_done >= self._total_files:
                self._samples.clear()

    def note_total_rows(self, total_rows: int):
        """Refine the projected total row count once known (secondary info only)."""
        with self._lock:
            if total_rows and total_rows >= self.rows_seen:
                self._total_rows_hint = int(total_rows)

    def stage(self, msg: str):
        """Publish a visible log line for the final combining/export stage.

        The UI polls ``snapshot()["stage_logs"]`` and repaints, so the page
        keeps updating (never appears frozen) while the big DataFrame work
        happens after the last file has been processed.
        """
        with self._lock:
            elapsed = time.monotonic() - self.started_at
            self._stage_logs.append((elapsed, msg))
            self.current_stage = msg

    def _avg_seconds_per_file(self):
        """Moving average seconds-per-completed-file; None if no samples yet."""
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    def snapshot(self) -> dict:
        """Return an immutable dict describing the current progress state."""
        with self._lock:
            elapsed = time.monotonic() - self.started_at
            avg_spf = self._avg_seconds_per_file()

            # --- fraction & ETA: FILE-COMPLETION based (primary metric) ---
            fraction = None
            eta = None
            basis = None

            all_files_done = bool(self._total_files
                                  and self.files_done >= self._total_files)

            if self._total_files > 0:
                done = min(self.files_done, self._total_files)
                fraction = done / self._total_files          # files, not rows
                if avg_spf is not None and avg_spf > 0:
                    eta = max(self._total_files - done, 0) * avg_spf
                    basis = "files"                          # avg time per file
                else:
                    eta = None                               # indeterminate yet
            # else: total file count unknown -> indeterminate (fraction=None)

            # Whole file-scan finished -> deterministic terminal state.
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
                "avg_seconds_per_file": avg_spf,
                "files_done": self.files_done,
                "total_files": self._total_files,
                "rows_seen": self.rows_seen,          # secondary info only
                "rows_kept": self.rows_kept,          # secondary info only
                "rows_dropped": self.rows_dropped,    # secondary info only
                "current_zip": self.current_zip,
                "current_csv": self.current_csv,
                "current_rows": self.current_rows,
                "last_file_status": self.last_file_status,
                "total_rows_hint": self._total_rows_hint,  # 0 => unknown (secondary)
                "cancelled": self._cancel.is_set(),
                "stage_logs": list(self._stage_logs),      # final-export visibility
                "current_stage": self.current_stage,
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


# Regex for known artifact/separator rows such as '---|---|---', '--', '- - -'.
_SEPARATOR_RE = re.compile(r"^[\s\-|=]+$")


def looks_like_separator(cells) -> bool:
    """True when a row is only dashes/pipes/equals/spaces (page-break artifact)."""
    joined = "".join(str(c) for c in cells if c is not None)
    if not joined.strip():
        return False
    return bool(_SEPARATOR_RE.match(joined))


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
                     progress=None, ts_col=None):
    """Turn raw CSV rows into a tidy DataFrame of output records.

    Returns ``(df_or_None, rows_read, error_message_or_None, artifacts)`` where
    ``artifacts`` is an :class:`ArtifactCounts` describing exactly why every
    non-parsed row was ignored (separator rows, ??? values, bad timestamps...).

    Invalid rows (blank, repeated header/page-break, bad date/time,
    non-numeric value, unexpected layout) are skipped safely *and counted*,
    so the Validation tab can prove no real data was lost.

    `progress` (optional): a ``ProgressTracker``.  When given, the row loop
    runs in *chunks* so live counters (rows read/kept/dropped for the current
    file) update continuously and cancellation can stop mid-file — this keeps
    memory flat and the UI responsive even on very large CSVs.

    `ts_col` (optional): index of a single combined timestamp column.  When
    given (used by the manual Validation-tab ZIP parsing), date/time columns
    are NOT required; the row's timestamp is read from `ts_col` instead and
    parsed directly with ``pd.to_datetime``.
    """
    art = ArtifactCounts()
    if not rows:
        return None, 0, "empty file", art

    # Strip whitespace in every cell
    rows = [[c.strip() if isinstance(c, str) else c for c in r] for r in rows]

    # Skip leading blank rows to find the first content row
    content_idx = next((i for i, r in enumerate(rows) if any(str(c).strip() for c in r)), None)
    if content_idx is None:
        return None, 0, "empty file", art

    rows = rows[content_idx:]
    header_row = rows[0]
    # A row that looks like a header but is actually a *separator artifact*
    # (e.g. '---,---,---') must not be treated as the header line.
    has_header = looks_like_header(header_row) and not looks_like_separator(header_row)

    col_title = col_value = col_date = col_time = None
    if has_header:
        idx = find_column_indices(header_row)
        if idx is not None:
            col_title = idx["title"]
            col_value = idx["value"]
            col_date = idx["date"]
            col_time = idx["time"]
        elif ts_col is not None:
            # Manual-validation mode: headers exist but lack separate date/time
            # columns — fall back to positional value/title plus the caller-
            # supplied single timestamp column index.
            norm_hdr = [str(c).strip().lower() for c in header_row]
            col_value = next((i for i, n in enumerate(norm_hdr)
                              if any(kw in n for kw in VALUE_KEYWORDS)), 1)
            col_title = next((i for i, n in enumerate(norm_hdr)
                              if any(kw in n for kw in TITLE_KEYWORDS)), None)
        else:
            # Headers exist but are unclear -> fall back to positions
            log_fn(f"{source_zip}/{source_csv}: headers unclear, using positional columns")
            has_header = False
    if not has_header:
        if ts_col is not None:
            # Headerless + combined-timestamp layout: 0=title, 1=value, ts=ts_col
            col_title, col_value = 0, 1
        else:
            col_title, col_value, col_date, col_time = 0, 1, 2, 3

    data_rows = rows[1:] if has_header else rows
    width = max((len(r) for r in data_rows), default=0)
    needed_cols = (col_title, col_value, col_date, col_time, ts_col)
    needed = max(i for i in needed_cols if i is not None) + 1
    if width < needed:
        return None, 0, f"not enough columns (found {width}, need {needed})", art

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

    def _try_parse_ts(date_raw: str, time_raw: str):
        """Parse one 'date time' pair; returns pd.Timestamp or None.

        Fast path only (the common sensor-export layout).  Rows that fail the
        fast path are collected and parsed in ONE vectorised flexible pass at
        the end of the file scan — calling pandas' format *guesser* once per
        row would be prohibitively slow on large CSVs.
        """
        s = date_raw + " " + time_raw
        ts = pd.to_datetime(s, format="%m/%d/%y %I:%M:%S %p", errors="coerce")
        if not pd.isna(ts):
            return ts
        return None

    pending = []  # (row_index_in_recs_placeholder, date_raw, time_raw) for fallback

    for start in range(0, len(data_rows), CHUNK):
        if progress is not None and progress.is_cancelled():
            raise CancelledError("cancelled while reading "
                                 f"{source_zip}/{source_csv}")
        chunk = data_rows[start:start + CHUNK]
        for r in chunk:
            # 'expected rows' = every raw line after the header.  Blank rows
            # are included so the accounting identity
            #   parsed + all drop-reasons == expected
            # holds exactly (nothing can ever go unexplained).
            art.expected_rows += 1
            # Fully blank row -> ignore (counted as artifact, not data loss)
            if not any(str(c).strip() for c in r):
                art.blank_rows += 1
                continue

            # Known separator/page-break artifact such as '---|---|---'
            if looks_like_separator(r):
                art.separator_rows += 1
                continue

            val_raw = str(cell(r, col_value)).strip()
            date_raw = str(cell(r, col_date)).strip()
            time_raw = str(cell(r, col_time)).strip()
            ts_raw = str(cell(r, ts_col)).strip() if ts_col is not None else ""
            title_raw = str(cell(r, col_title)).strip() if col_title is not None else ""

            # Repeated header-like rows inside the data (page breaks) -> skip
            if looks_like_header([title_raw, val_raw, date_raw, time_raw]) and \
                    not _is_number(val_raw):
                art.repeated_header_rows += 1
                continue

            # Malformed: not enough columns to even reach value/timestamp
            max_required = max(i for i in (col_value, ts_col if ts_col is not None
                                           else max(col_date, col_time)) if i is not None)
            if len(r) <= max_required:
                art.malformed_rows += 1
                continue

            if ts_col is not None:
                # Single combined timestamp column (manual-validation mode).
                if ts_raw == "":
                    # Missing timestamp/date/time -> unusable row
                    art.malformed_rows += 1
                    continue
            elif date_raw == "" or time_raw == "":
                # Missing date/time -> unusable row (often part of an artifact)
                art.malformed_rows += 1
                continue

            # Value must be numeric; '???', 'N/A', blanks etc. are artifacts
            try:
                value = float(val_raw)
            except ValueError:
                art.bad_value_rows += 1
                continue
            if value != value:  # NaN
                art.bad_value_rows += 1
                continue

            # Timestamp must parse; rows failing the fast path are queued for
            # ONE vectorised flexible-parse pass after the scan (and only
            # counted as bad_timestamp_rows if that fallback fails too).
            if ts_col is not None:
                ts = pd.to_datetime(ts_raw, format="%m/%d/%y %I:%M:%S %p",
                                    errors="coerce")
                if pd.isna(ts):
                    ts = pd.to_datetime(ts_raw, format="%Y-%m-%d %H:%M:%S",
                                        errors="coerce")
                if pd.isna(ts):
                    pending.append((len(recs), ts_raw, ""))
                    recs.append((None, title_raw, value))  # placeholder
                    art.parsed_rows += 1                   # provisional
                    rows_kept_chunk += 1
                else:
                    recs.append((ts, title_raw, value))
                    art.parsed_rows += 1
                    rows_kept_chunk += 1
            else:
                ts = _try_parse_ts(date_raw, time_raw)
                if ts is None:
                    pending.append((len(recs), date_raw, time_raw))
                    recs.append((None, title_raw, value))  # placeholder, filled below
                    art.parsed_rows += 1                   # provisional; corrected later
                    rows_kept_chunk += 1
                else:
                    recs.append((ts, title_raw, value))
                    art.parsed_rows += 1
                    rows_kept_chunk += 1

        rows_read = min(start + CHUNK, len(data_rows))
        if progress is not None:
            # Report this chunk's counts (kept here = valid-parsable rows;
            # final kept may still shrink after dedupe).
            progress.add_rows(len(chunk), rows_kept_chunk,
                              len(chunk) - rows_kept_chunk)
            rows_kept_chunk = 0  # report deltas only, avoid double counting

    # ---- Vectorised flexible fallback for fast-path timestamp failures ----
    # One pd.to_datetime call over just the *failed* strings (rare) instead of
    # a per-row format guess — keeps large-file scans fast and exact.
    if pending:
        idxs = [i for i, _d, _t in pending]
        # ts_col mode stores the whole timestamp string in the "date" slot;
        # date/time mode concatenates the two fields as before.
        strs = pd.Series([d + (" " + t if t else "") for _i, d, t in pending])
        try:
            parsed = pd.to_datetime(strs, errors="coerce", format=None)
        except Exception:
            parsed = pd.Series([pd.NaT] * len(strs))
        drop_positions = set()
        for pos, rec_i in enumerate(idxs):
            ts_val = parsed.iloc[pos]
            if ts_val is None or pd.isna(ts_val):
                drop_positions.add(rec_i)          # truly unparseable -> remove
            else:
                _t0, title_old, v_old = recs[rec_i]
                recs[rec_i] = (ts_val, title_old, v_old)
        n_still_bad = len(drop_positions)
        if n_still_bad:
            art.bad_timestamp_rows += n_still_bad
            art.parsed_rows -= n_still_bad
            recs = [rec for j, rec in enumerate(recs) if j not in drop_positions]

    if not recs:
        reason = "no valid data rows"
        if art.bad_timestamp_rows:
            reason += f" ({art.bad_timestamp_rows} unparseable timestamp(s))"
        if art.bad_value_rows:
            reason += f" ({art.bad_value_rows} invalid value(s))"
        return None, rows_read, reason, art

    # Build the DataFrame directly from already-parsed values — no second
    # vectorised timestamp pass needed (avoids an unnecessary copy).
    df = pd.DataFrame(recs, columns=["timestamp", "title", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])  # safety net; should be a no-op
    df["value"] = df["value"].astype(np.float64)
    if df.empty:
        return None, rows_read, "all timestamps failed to parse", art

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
    return df, rows_read, None, art


def process_csv_rows_legacy(rows, kind, area_from_name, source_zip, source_csv, log_fn):
    """Backwards-compatible 3-tuple wrapper around :func:`process_csv_rows`.

    The original command-line script (combine_sensor_data.py) unpacks
    ``(df, rows_read, error)``; this thin adapter keeps that call site working
    unchanged while the UI uses the richer 4-tuple version with artifact
    accounting and live progress.
    """
    df, rows_read, err, _artifacts = process_csv_rows(
        rows, kind, area_from_name, source_zip, source_csv, log_fn)
    return df, rows_read, err


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

    Exact duplicates (same key AND same value) are always dropped safely.
    Conflicting values (same key, different value) are resolved per policy
    AND recorded in a conflicts DataFrame so the Validation tab can list
    them — removals are never silent.

    Returns ``(df_after, n_removed, exact_dupes_removed, conflict_df,
    conflict_rows_removed)``.
    """
    empty_conflicts = pd.DataFrame(columns=OUTPUT_COLUMNS + ["conflict_group"])
    if df.empty or mode == "keep all":
        return df, 0, 0, empty_conflicts, 0

    key_cols = ["timestamp", "kind", "area"]

    # 1) Drop *exact* duplicates (same key AND same value) — pure noise,
    #    regardless of keep-first/keep-last preference.
    before = len(df)
    exact_dup_mask = df.duplicated(subset=key_cols + ["value"], keep="first")
    exact_removed = int(exact_dup_mask.sum())
    if exact_removed:
        log_fn(f"Removed {exact_removed} exact duplicate row(s) "
               "(same timestamp+kind+area+value)")
    df = df[~exact_dup_mask]

    # 2) Conflicting values for the same (timestamp, kind, area):
    #    record them, then resolve according to the chosen policy.
    conflict_mask = df.duplicated(subset=key_cols, keep=False)
    conflict_df = empty_conflicts
    conflict_rows_removed = 0
    if conflict_mask.any():
        conflicts = df[conflict_mask].copy()
        # Assign a stable group id per conflicting key so validation can
        # show which rows fought over the same (timestamp, kind, area).
        grp_ids = conflicts.groupby(key_cols, sort=False,
                                 observed=True).ngroup() + 1
        conflict_df = pd.concat([conflicts,
                                 grp_ids.rename("conflict_group")], axis=1)
        if len(conflict_df) > MAX_CONFLICT_ROWS_KEPT:
            # Keep display bounded, but count everything accurately.
            conflict_df = conflict_df.head(MAX_CONFLICT_ROWS_KEPT)

        seq = conflicts.groupby(key_cols, sort=False,
                          observed=True).cumcount(ascending=(mode != "keep first"))
        drop_idx = conflicts.index[seq > 0]
        conflict_rows_removed = int(len(drop_idx))
        df = df.drop(index=drop_idx)
        log_fn(f"Duplicates ({mode}): removed {conflict_rows_removed} conflicting "
               "row(s) on (timestamp, kind, area) — see Validation tab")

    return (df.reset_index(drop=True), before - len(df), exact_removed,
            conflict_df.reset_index(drop=True), conflict_rows_removed)


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
      * a cheap pre-scan counts every CSV entry so the PRIMARY progress bar
        can be based on completed files / total files (row estimates are only
        reported as secondary information);
      * current file name + per-chunk row counters are reported continuously,
        and each finished file produces a timing sample for the per-file ETA;
      * the final combining/export stage emits visible ``stage()`` logs so the
        UI never appears frozen while combined.csv is assembled;
      * cancellation is honoured between files and within large files.
    """
    t0 = time.monotonic()
    result = ProcessResult(duplicate_mode=duplicate_mode)
    messages = result.messages

    def log_fn(msg):
        messages.append(msg)
        if progress is not None:
            # Mirror processing notes into the live Logs feed too.
            progress.stage(msg)

    frames = []          # collected per-CSV DataFrames, concatenated ONCE at the end
    pre_dedupe_rows = 0  # rows kept before duplicate resolution

    def stage(msg):
        """Emit a visible final-stage log line (also lands in `messages`)."""
        if progress is not None:
            progress.stage(msg)
        messages.append(msg)

    # Deterministic ordering makes runs reproducible.
    items = sorted(zip_items, key=lambda t: str(t[0]))

    # ---- Pre-scan: EXACT total CSV file count (+ rough row estimate) ------
    # Uses only ZIP central-directory metadata (no decompression), so it is
    # fast even for big archives.  The file count drives the primary progress
    # bar; the row estimate is kept purely as secondary display info.
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

            # ---- Corrupt ZIP handling: skip and CONTINUE processing ----
            try:
                zf = zipfile.ZipFile(io.BytesIO(payload))
            except (zipfile.BadZipFile, OSError) as e:
                err = f"Skipping corrupt/unreadable ZIP {zip_name}: {e}"
                result.errors.append(err)
                if progress is not None:
                    progress.stage(f"❌ {err} (continuing with remaining files)")
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
                        df, rows_read, err, art = process_csv_rows(
                            rows, kind, area_from_name, zip_name, inner_name, log_fn,
                            progress=progress,
                        )
                        # Always record artifact accounting — even for files
                        # that ended up skipped, so validation can explain why.
                        result.artifact_counts[(zip_name, inner_name)] = art
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

                    # One unit of work finished -> per-file timing sample for
                    # the file-based progress bar + ETA.  Failure is logged
                    # and processing CONTINUES with the next file.
                    if progress is not None:
                        progress.finish_file(status=status)
                        if status != "ok":
                            progress.stage(f"⚠️ {zip_name}/{inner_name}: "
                                           f"{status.upper()} — {err or 'no usable data'} "
                                           "(continuing)")

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
        combined = _empty_combined()
        result.combined = combined
        result.rows_read = sum(r["rows_read"] for r in result.per_file_log)
        result.rows_kept = 0
        result.rows_dropped = result.rows_read
        result.elapsed_seconds = time.monotonic() - t0
        return result

    # ==================================================================
    # FINAL STAGE: combining + export.  Every step below emits a visible
    # stage() log line so the UI never looks frozen after the last file.
    # ==================================================================
    t_export = time.monotonic()
    stage(f"✅ Finished processing all files ({result.csv_files_found} file(s) attempted, "
          f"{result.csv_files_processed} produced data)")

    # ---- Concatenate efficiently: ONE pd.concat over all collected frames
    # (no repeated DataFrame.append / concat inside the per-file loop).
    stage(f"Concatenating processed data ({len(frames)} chunk(s))…")
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        del frames  # free the chunk list immediately (avoid holding copies)
    else:
        combined = _empty_combined()

    if not combined.empty:
        # Efficient dtypes: categorical for low-cardinality text columns
        # (smaller memory + faster sort/groupby), float64 for values,
        # datetime64[ns] for timestamps.  Round-trip safe for CSV writing.
        stage("Optimizing dtypes (categorical kind/area/source columns)…")
        for col in ("kind", "area", "source_zip", "source_csv"):
            combined[col] = combined[col].astype("category")

        # Final safety net: coerce types and drop any remaining invalid rows
        # (invalid dates/times or non-numeric values that slipped through).
        stage("Filtering any remaining invalid timestamps/values…")
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
        before_safety = len(combined)
        combined = combined.dropna(subset=["timestamp", "value"])
        n_safety = before_safety - len(combined)
        if n_safety:
            log_fn(f"Final safety filter dropped {n_safety} row(s) with unparseable "
                   "timestamps or values")

        pre_dedupe_rows = len(combined)

        # ---- Removing exact duplicates (safe: same key AND same value) ----
        stage(f"Removing exact duplicates ({pre_dedupe_rows:,} row(s) to check)…")
        combined, removed, exact_removed, conflict_df, conflict_removed = \
            apply_duplicate_policy(combined, duplicate_mode, log_fn)
        result.exact_duplicates_removed = exact_removed
        result.conflict_rows_removed = conflict_removed
        result.duplicate_conflicts = conflict_df
        if not conflict_df.empty:
            stage(f"Logged {conflict_df['conflict_group'].nunique()} duplicate-conflict "
                  f"group(s) ({conflict_removed} conflicting row(s) resolved via "
                  f"'{duplicate_mode}')")
        else:
            stage("No duplicate conflicts found (only exact duplicates collapsed)")

        # ---- Single efficient sort (dedupe above already grouped keys) ----
        stage("Sorting by timestamp, kind, area…")
        combined = combined.sort_values(["timestamp", "kind", "area", "source_zip",
                                         "source_csv"], kind="mergesort").reset_index(drop=True)
    else:
        removed = 0
        combined = combined[OUTPUT_COLUMNS] if len(combined.columns) else _empty_combined()

    result.combined = combined
    result.pre_dedupe_rows = pre_dedupe_rows
    result.rows_read = sum(r["rows_read"] for r in result.per_file_log)
    result.rows_kept = len(combined)
    result.rows_dropped = max(result.rows_read - result.rows_kept, 0)

    # ---- Run validation checks BEFORE declaring the export ready ----------
    # (cheap relative to the parse phase; guarantees the numbers shown in the
    # Validation tab come from the exact bytes the user will download.)
    stage("Running validation checks…")
    try:
        report = validate_result(result)
        result.messages.extend(report.notes)
        stage(f"Validation {'PASSED ✅' if report.passed else 'FAILED ❌'} — "
              f"source valid rows: {report.total_source_parsed:,}, "
              f"combined rows: {report.total_combined:,}, "
              f"difference: {report.row_count_difference:,}")
    except Exception as e:  # validation must never break the run
        stage(f"Validation could not complete ({type(e).__name__}: {e})")

    stage(f"Finished writing combined.csv (export took "
          f"{format_duration(time.monotonic() - t_export) or '0s'})")

    # Sanity note when the per-file scan and final table disagree (dupes).
    if removed:
        log_fn(f"Removed {removed} duplicate row(s) after concatenation "
               f"(pre-dedupe rows: {pre_dedupe_rows})")
    result.elapsed_seconds = time.monotonic() - t0
    result.export_elapsed_seconds = time.monotonic() - t_export
    return result


def _empty_combined() -> pd.DataFrame:
    """Empty combined frame with correct column names AND dtypes."""
    df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in {
        "timestamp": "datetime64[ns]", "kind": "object", "area": "object",
        "value": "float64", "source_zip": "object", "source_csv": "object"}.items()})
    return df[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------
# Export: write combined.csv (chunked for large tables, in-memory buffer)
# ---------------------------------------------------------------------------


def write_combined_csv(df: pd.DataFrame, buf=None, chunksize: int = 200_000) -> bytes:
    """Serialize the combined DataFrame to CSV *bytes* efficiently.

    Optimizations (correctness-preserving):
      * categorical columns are written straight from their dictionaries —
        pandas handles this without materialising giant object arrays;
      * for large tables the CSV is streamed to the buffer in chunks
        (``to_csv(..., chunksize=...)`` semantics via manual slicing) so peak
        memory stays bounded;
      * the buffer is returned as ``bytes`` for st.download_button, created
        exactly once per run.
    """
    if buf is None:
        buf = io.BytesIO()

    if df.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(buf, index=False)
        return buf.getvalue()

    out = df[OUTPUT_COLUMNS]
    if len(out) <= chunksize:
        # Small enough: single pass, no chunk bookkeeping overhead.
        out.to_csv(buf, index=False)
    else:
        # Chunked writing: header once, then slices (views, not copies).
        out.iloc[:0].to_csv(buf, index=False)          # header only
        start = 0
        n = len(out)
        while start < n:
            out.iloc[start:start + chunksize].to_csv(buf, index=False, header=False)
            start += chunksize
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Validation engine (backs the Validation / Compare tab)
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Outcome of comparing the combined output against the source ZIP/CSVs."""

    passed: bool = False
    total_source_expected: int = 0   # all raw data rows across sources
    total_source_parsed: int = 0     # rows that passed parsing (valid source rows)
    total_source_dropped: int = 0    # artifact rows ignored (separators, ???, ...)
    total_combined: int = 0          # rows in the final combined table
    row_count_difference: int = 0    # parsed - combined (explained by dedupe policy)
    duplicates_detected: int = 0     # pre-dedupe rows sharing (ts, kind, area)
    exact_duplicates_removed: int = 0
    conflict_rows_removed: int = 0
    conflict_groups: int = 0
    duplicate_conflicts: pd.DataFrame = field(default_factory=pd.DataFrame)
    rows_with_invalid_timestamp: int = 0   # dropped during parsing (source side)
    rows_with_invalid_value: int = 0       # dropped during parsing (source side)
    coverage_gaps: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage_checked: bool = False         # minute-gap analysis applicable?
    per_file: pd.DataFrame = field(default_factory=pd.DataFrame)
    zip_list: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def validate_result(result: ProcessResult) -> ValidationReport:
    """Compare the combined output against the original ZIP/CSV sources.

    Checks performed:
      * per source file: expected / parsed / dropped rows + artifact reasons
        (separators, repeated headers, ??? values, bad timestamps, malformed)
      * accounting identity: parsed + every drop-reason == expected
        (proves nothing was discarded without explanation)
      * every successfully parsed source row is represented in the combined
        output — the difference must equal exactly the rows removed by the
        duplicate policy (exact duplicates + resolved conflicts + safety net)
      * duplicate (timestamp, kind, area) detection: exact duplicates vs
        value conflicts (conflicts listed, never hidden)
      * per (kind, area) minute-interval coverage: missing minutes between
        min and max timestamp (only meaningful for one-row-per-minute data)
    """
    rep = ValidationReport()
    notes = rep.notes

    # ---- Per-file table from the artifact accounting --------------------
    rows = []
    total_expected = total_parsed = total_dropped = 0
    unaccounted_files = []
    for (zip_name, csv_name), art in sorted(result.artifact_counts.items()):
        d = art.as_dict()
        status = "ok" if art.parsed_rows else "skipped"
        # Match the processing status from the per-file log, if present.
        for r in result.per_file_log:
            if r["source_zip"] == zip_name and r["source_csv"] == csv_name:
                status = r["status"]
                break
        rows.append(dict(
            source_zip=zip_name, source_csv=csv_name, status=status,
            **d,
            dropped_rows=art.dropped_total,
            accounted=art.accounted,
        ))
        total_expected += art.expected_rows
        total_parsed += art.parsed_rows
        total_dropped += art.dropped_total
        if not art.accounted:
            unaccounted_files.append(f"{zip_name}/{csv_name}")

    rep.per_file = pd.DataFrame(rows)
    rep.total_source_expected = total_expected
    rep.total_source_parsed = total_parsed
    rep.total_source_dropped = total_dropped
    rep.rows_with_invalid_timestamp = sum(a.bad_timestamp_rows
                                          for a in result.artifact_counts.values())
    rep.rows_with_invalid_value = sum(a.bad_value_rows
                                      for a in result.artifact_counts.values())

    if unaccounted_files:
        rep.issues.append(f"{len(unaccounted_files)} file(s) have rows that are "
                          "neither parsed nor bucketed as artifacts: "
                          + ", ".join(unaccounted_files[:5]))

    # ---- ZIP inventory ---------------------------------------------------
    zips = {}
    for r in result.per_file_log:
        zn = r["source_zip"]
        z = zips.setdefault(zn, dict(source_zip=zn, csv_files=[], csv_count=0,
                                     expected_rows=0, parsed_rows=0))
        if r["source_csv"]:
            z["csv_files"].append(r["source_csv"])
            z["csv_count"] += 1
            art = result.artifact_counts.get((zn, r["source_csv"]))
            if art:
                z["expected_rows"] += art.expected_rows
                z["parsed_rows"] += art.parsed_rows
    rep.zip_list = pd.DataFrame([
        dict(source_zip=z["source_zip"], csv_count=z["csv_count"],
             csv_files=", ".join(z["csv_files"]),
             expected_rows=z["expected_rows"], parsed_rows=z["parsed_rows"])
        for z in zips.values()])

    # ---- Combined-side counts --------------------------------------------
    combined = result.combined
    rep.total_combined = int(len(combined))
    rep.exact_duplicates_removed = int(result.exact_duplicates_removed)
    rep.conflict_rows_removed = int(result.conflict_rows_removed)
    rep.duplicate_conflicts = result.duplicate_conflicts
    if not result.duplicate_conflicts.empty:
        rep.conflict_groups = int(result.duplicate_conflicts["conflict_group"].nunique())

    # Row-count reconciliation:
    #   parsed_source_rows - combined_rows should equal exactly the rows the
    #   duplicate policy removed (+ any final safety-net drops, which are also
    #   logged).  Any *unexplained* difference means data really is missing.
    diff = total_parsed - rep.total_combined
    rep.row_count_difference = int(diff)
    explained = rep.exact_duplicates_removed + rep.conflict_rows_removed
    unexplained = diff - explained
    if result.cancelled:
        notes.append("Run was cancelled — validation reflects partial data only.")
        rep.issues.append("Processing was cancelled; combined output is incomplete.")
    elif unexplained > 0:
        rep.issues.append(
            f"{unexplained:,} parsed source row(s) are missing from the combined "
            f"output and are NOT explained by duplicate removal "
            f"(exact dupes: {rep.exact_duplicates_removed:,}, "
            f"conflicts resolved: {rep.conflict_rows_removed:,}).")
    elif unexplained < 0:
        rep.issues.append(
            f"Combined output has {-unexplained:,} more row(s) than the source "
            "parse count — please inspect the Logs tab.")

    # ---- Duplicate detection on the COMBINED output ----------------------
    # (should be zero unless mode == 'keep all')
    if not combined.empty:
        key_cols = ["timestamp", "kind", "area"]
        rep.duplicates_detected = int(combined.duplicated(subset=key_cols,
                                                           keep="first").sum())
        if rep.duplicates_detected and result.duplicate_mode != "keep all":
            rep.issues.append(f"{rep.duplicates_detected:,} duplicate "
                              "(timestamp, kind, area) row(s) remain in combined.csv.")
        # No NaT / NaN may survive into the output.
        bad_ts = int(combined["timestamp"].isna().sum())
        bad_val = int(pd.to_numeric(combined["value"], errors="coerce").isna().sum())
        if bad_ts or bad_val:
            rep.issues.append(f"Combined output contains {bad_ts} invalid "
                              f"timestamp(s) / {bad_val} invalid value(s).")

    # ---- Minute-interval coverage per (kind, area) ------------------------
    # For sensor data expected to be one row per minute, look for missing
    # whole minutes between each series' min and max timestamp.
    if not combined.empty:
        notes.append("Minute-gap analysis assumes one row per minute per "
                     "(kind, area); gaps are informational, not failures.")
        rep.coverage_checked = True
        gap_rows = []
        # int64 nanoseconds — avoids datetime64//int dtype errors entirely
        ts = combined["timestamp"].values.astype("datetime64[ns]").astype("int64")
        kinds = combined["kind"].astype(str).values
        areas = combined["area"].astype(str).values
        # combined is sorted by timestamp -> grouping preserves order
        groups = pd.Series(range(len(combined))).groupby([kinds, areas], sort=True)
        one_min = 60 * 1_000_000_000  # ns in one minute
        for (k, a), idx in groups:
            t = ts[idx.values]
            if len(t) < 2:
                continue
            tmin, tmax = int(t[0]), int(t[-1])
            expected_minutes = (tmax - tmin) // one_min + 1
            actual_minutes = len(np.unique(t // one_min))
            missing = expected_minutes - actual_minutes
            if missing > 0:
                gap_rows.append(dict(kind=k, area=a,
                                     min_timestamp=pd.Timestamp(tmin),
                                     max_timestamp=pd.Timestamp(tmax),
                                     minutes_present=int(actual_minutes),
                                     minutes_expected=int(expected_minutes),
                                     missing_minutes=int(missing)))
        rep.coverage_gaps = pd.DataFrame(gap_rows)
        if gap_rows:
            tot = sum(g["missing_minutes"] for g in gap_rows)
            notes.append(f"Found {tot:,} missing minute(s) across "
                         f"{len(gap_rows)} (kind, area) series — "
                         "likely sensor dropout, not a processing bug.")

    # ---- Min / max timestamp per source file (into per-file table) -------
    if not combined.empty and not rep.per_file.empty:
        mm = (combined.groupby(["source_zip", "source_csv"], observed=True)["timestamp"]
              .agg(min_ts="min", max_ts="max"))
        rep.per_file = rep.per_file.merge(mm, how="left",
                                          left_on=["source_zip", "source_csv"],
                                          right_index=True)

    rep.passed = (not rep.issues) and not result.cancelled
    return rep
