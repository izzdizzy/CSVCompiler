#!/usr/bin/env python3
"""
app.py — Streamlit UI for the ZIP -> combined CSV tool.

This file ONLY handles the user interface (upload, controls, live progress,
status messages, preview, download).  All data cleaning / combining logic
lives in processor.py so it can be reused by the CLI script
(combine_sensor_data.py) and tests.

Live progress design
--------------------
* Processing runs on a *background worker thread* so the Streamlit script
  (UI) thread is never blocked — the page keeps rendering and stays
  responsive while large archives are being crunched.
* The worker only ever touches a thread-safe ``ProgressTracker`` (in
  processor.py); the UI thread polls ``tracker.snapshot()`` (immutable dict
  copies — no shared mutable state crosses threads besides the tracker's
  internal lock) and repaints the progress bar / status text with
  ``st.empty()`` placeholders.  This is the safe Streamlit live-update
  pattern: all ``st.*`` calls happen on the main script thread.
* ETA comes from a moving average of processing speed inside the tracker:
  rows/sec when row totals are estimable, files/sec otherwise, and an
  indeterminate (spinner-style) state when neither can be computed yet.

Run locally:
    Windows:      streamlit run app.py --server.port 8501
    Mac/Linux:    streamlit run app.py --server.port 8501
Then open http://localhost:8501 in a browser.
"""

import io
import threading
import time

import pandas as pd
import streamlit as st

# All processing logic is imported from processor.py — no parsing code here.
from processor import (
    DUPLICATE_MODES,
    OUTPUT_COLUMNS,
    ProgressTracker,
    format_duration,
    process_zips,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ZIP to Combined CSV", page_icon="🗜️", layout="wide")

st.title("ZIP to Combined CSV")
st.caption(
    "Upload one or more ZIP files containing CSVs (nested folders allowed). "
    "The app cleans each CSV, merges everything into one tidy table with columns "
    "`timestamp, kind, area, value, source_zip, source_csv`, shows live progress "
    "with ETA while processing, lets you preview it, and downloads it as `combined.csv`."
)


# ---------------------------------------------------------------------------
# Background worker helpers
# ---------------------------------------------------------------------------
def _worker(zip_items, duplicate_mode, tracker):
    """Run the pipeline on a background thread; stash the outcome in session_state.

    Only plain, thread-safe objects are touched here: the processor writes into
    the locked ``ProgressTracker``, and the final result is stored under a
    single session_state key (dict assignment is atomic enough for our poller,
    which only proceeds once the 'done' flag is set).
    """
    try:
        result = process_zips(zip_items, duplicate_mode=duplicate_mode,
                              progress=tracker)
        st.session_state["run_result"] = result
    except Exception as e:
        # Last-resort guard: surface unexpected errors instead of hanging the UI.
        st.session_state["run_error"] = f"{type(e).__name__}: {e}"
    finally:
        st.session_state["run_done"] = True


def _render_progress(tracker):
    """Draw one frame of the live progress UI from a tracker snapshot.

    Returns the snapshot dict so callers can inspect terminal conditions.
    Uses pre-created placeholder containers so repeated repaints don't grow
    the page.
    """
    snap = tracker.snapshot()
    bar = st.session_state["_pb_bar"]
    txt = st.session_state["_pb_text"]
    cur = st.session_state["_pb_current"]
    rows = st.session_state["_pb_rows"]

    fraction = snap["fraction"]
    if fraction is None:
        # Indeterminate state: total unknown (or no speed samples yet).
        # Show a slowly-advancing animated bar rather than failing/lying.
        bar.progress(None, text="Processing… estimating progress")
    else:
        pct = f" · {fraction * 100:.0f}%"
        eta_txt = ""
        if snap["eta_seconds"] is not None:
            eta_txt = f" · ETA {format_duration(snap['eta_seconds'])}"
        bar.progress(fraction, text=f"Processing{pct}{eta_txt}")

    # ---- Live status line: elapsed / ETA / speed ----
    parts = [f"⏱️ Elapsed **{format_duration(snap['elapsed']) or '0s'}**"]
    if snap["eta_seconds"] is not None:
        basis = {"rows": "rows/sec", "files": "files/sec"}.get(snap["eta_basis"], "")
        parts.append(f"🔮 ETA **{format_duration(snap['eta_seconds'])}**"
                     + (f" (via {basis})" if basis else ""))
    else:
        parts.append("🔮 ETA calculating…")
    if snap["rows_per_sec"]:
        parts.append(f"⚡ {snap['rows_per_sec']:,.0f} rows/s")
    txt.markdown("  |  ".join(parts))

    # ---- Current file line ----
    total_files = snap["total_files"]
    if snap["current_zip"] or snap["current_csv"]:
        idx = min(snap["files_done"] + 1, total_files) if total_files else snap["files_done"] + 1
        counter = (f"Processing file **{idx} of {total_files}**"
                   if total_files else f"Processing file **{idx}**")
        cur.markdown(
            f"{counter}  \n"
            f"📦 ZIP: `{snap['current_zip'] or '—'}`  \n"
            f"📄 CSV: `{snap['current_csv'] or '—'}`  \n"
            f"Rows scanned in this file: **{snap['current_rows']:,}**"
        )
    else:
        cur.markdown("Opening ZIP archive(s)…")

    # ---- Running totals ----
    kept, dropped = snap["rows_kept"], snap["rows_dropped"]
    seen = snap["rows_seen"]
    total_hint = snap["total_rows_hint"]
    total_txt = f" of ~{total_hint:,} est." if total_hint else "(total unknown)"
    rows.markdown(
        f"**Rows processed:** {seen:,}{total_txt}  |  "
        f"✅ Kept so far: **{kept:,}**  |  ❌ Dropped so far: **{dropped:,}**"
    )
    return snap


# ---------------------------------------------------------------------------
# Sidebar: inputs & options
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Inputs")

    # Requirement 4: support MULTIPLE uploaded ZIP files at once.
    uploads = st.file_uploader(
        "ZIP files (may contain nested folders of CSVs)",
        type=["zip"],
        accept_multiple_files=True,
        help="Each ZIP should contain one or more .csv files. Non-CSV entries are ignored.",
    )

    st.header("Options")

    # Requirement 8: duplicate handling dropdown, default "keep last".
    # Duplicates are keyed on (timestamp, kind, area).
    duplicate_mode = st.selectbox(
        "Duplicate handling (by timestamp + kind + area)",
        options=list(DUPLICATE_MODES),
        index=0,  # "keep last" is first in DUPLICATE_MODES -> default
        help=(
            "keep last: newest processed row wins for conflicting values\n"
            "keep first: oldest processed row wins\n"
            "keep all: keep every valid row (duplicates included)"
        ),
    )

    # Requirement: an explicit Process button (nothing runs until clicked).
    running = st.session_state.get("run_thread") is not None and \
              st.session_state["run_thread"].is_alive()
    process_clicked = st.button("Process", type="primary",
                                use_container_width=True,
                                disabled=running)

    # Optional improvement: cancel button (only active while a run is going).
    cancel_clicked = st.button("⏹ Cancel", use_container_width=True,
                               disabled=not running)
    if cancel_clicked:
        tracker = st.session_state.get("run_tracker")
        if tracker is not None:
            tracker.cancel()  # cooperative: worker stops at the next chunk/file
            st.toast("Cancellation requested — stopping after current chunk…")

    st.divider()
    st.markdown(
        "**kind** is inferred from the ZIP filename (e.g. `btu` → BTU, "
        "`temp` → Temperature); otherwise `unknown`.  \n"
        "**area** is the CSV filename without extension; falls back to the "
        "tag/title column, then to `source_csv`."
    )

# ---------------------------------------------------------------------------
# Processing (triggered only by the Process button; runs on a worker thread)
# ---------------------------------------------------------------------------
if process_clicked:
    if not uploads:
        # Basic guard: nothing uploaded yet.
        st.warning("Please upload at least one ZIP file before pressing Process.")
    else:
        # Build (name, bytes) pairs for the processor; reading the uploaded
        # files happens exactly once per file (no repeated loops).
        zip_items = [(u.name, u.getvalue()) for u in uploads]

        # Fresh run state (clear any previous outcome markers).
        tracker = ProgressTracker(total_files=0, total_rows_hint=0)
        st.session_state["run_tracker"] = tracker
        st.session_state["run_thread"] = threading.Thread(
            target=_worker, args=(zip_items, duplicate_mode, tracker), daemon=True)
        st.session_state["run_done"] = False
        st.session_state.pop("run_result", None)
        st.session_state.pop("run_error", None)

        # Containers that _render_progress repaints in place (no page growth).
        st.subheader("Live progress")
        st.session_state["_pb_bar"] = st.progress(0.0, text="Starting…")
        st.session_state["_pb_text"] = st.empty()
        st.session_state["_pb_current"] = st.empty()
        st.session_state["_pb_rows"] = st.empty()
        cancel_note = st.empty()

        st.session_state["run_thread"].start()

        # ---- Poll loop on the MAIN thread: repaint ~5x/sec, UI stays alive ----
        while True:
            _render_progress(tracker)
            if st.session_state.get("run_done"):
                break
            if tracker.is_cancelled():
                cancel_note.info("⏹ Cancellation requested — finishing current chunk…")
            time.sleep(0.2)

        # Run finished: tear down live widgets and render the final outcome.
        st.session_state["_pb_bar"].empty()
        st.session_state["_pb_text"].empty()
        st.session_state["_pb_current"].empty()
        st.session_state["_pb_rows"].empty()
        cancel_note.empty()

        # Cache the outcome so preview/download survive widget reruns.
        if "run_error" in st.session_state:
            st.error(f"Unexpected error while processing: {st.session_state['run_error']}")
        else:
            result = st.session_state.get("run_result")
            if result is not None:
                st.session_state["result"] = result
                if result.cancelled:
                    st.warning("Processing was cancelled — no output produced.")
                elif result.combined.empty:
                    st.warning("No usable rows found — see the warnings below.")
                else:
                    st.success(f"Done — {result.rows_kept:,} rows kept in "
                               f"{format_duration(result.elapsed_seconds)}.")

# ---------------------------------------------------------------------------
# Summary + preview + download
# ---------------------------------------------------------------------------
result = st.session_state.get("result")

if result is not None:
    st.subheader("Summary")

    # Requirement: summary counts + errors/warnings.
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ZIP files processed", f"{result.zips_processed:,}",
              delta=None if not result.zips_failed else f"-{result.zips_failed} corrupt")
    c2.metric("CSV files found", f"{result.csv_files_found:,}")
    c3.metric("Rows read", f"{result.rows_read:,}")
    c4.metric("Rows kept", f"{result.rows_kept:,}")
    c5.metric("Rows dropped", f"{result.rows_dropped:,}")

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("CSV files processed", f"{result.csv_files_processed:,}")
    c7.metric("CSV files skipped", f"{result.csv_files_skipped:,}")
    c8.metric("Duplicates removed", f"{result.duplicates_removed:,}")
    c9.metric("Total processing time", format_duration(result.elapsed_seconds) or "0s")

    # ---- ETA accuracy note (optional improvement) -----------------------
    # The tracker remembers the last non-zero ETA it displayed.  Comparing
    # "elapsed when that ETA was shown + that ETA" against the actual total
    # processing time tells the user how accurate the estimate ended up being.
    tracker = st.session_state.get("run_tracker")
    if (tracker is not None and not result.cancelled
            and result.elapsed_seconds > 0
            and getattr(tracker, "last_eta_seconds", None)):
        snap = tracker.snapshot()
        projected = snap["elapsed"] + tracker.last_eta_seconds
        err = abs(projected - result.elapsed_seconds)
        pct = (err / result.elapsed_seconds * 100) if result.elapsed_seconds else 0
        basis = {"rows": "rows/sec", "files": "files/sec"}.get(
            tracker.last_eta_basis, "n/a")
        st.caption(f"ETA accuracy: last projection ≈ {format_duration(projected)} "
                   f"vs actual {format_duration(result.elapsed_seconds)} "
                   f"(off by {pct:.0f}%). Basis: {basis}.")

    # Errors / warnings block (collapsible so it doesn't dominate the page).
    problems = list(result.errors)
    if problems:
        with st.expander(f"⚠️ Errors / warnings ({len(problems)})", expanded=True):
            for p in problems:
                st.text(p)
    else:
        st.success("No errors — every readable CSV was processed.")

    # Skipped-files quick view (Requirement: show skipped files + reasons).
    skipped = [r for r in result.per_file_log
               if r["status"] in ("skipped", "error", "no_csv")]
    if skipped:
        with st.expander(f"🚫 Skipped / failed files ({len(skipped)})", expanded=False):
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)

    # Per-file log: which ZIP/CSV produced what, and why anything was skipped.
    if result.per_file_log:
        log_df = pd.DataFrame(result.per_file_log)
        with st.expander("Per-file processing log", expanded=False):
            st.dataframe(log_df, use_container_width=True, hide_index=True)

    st.subheader("Result preview")
    if result.combined.empty:
        st.info("Combined result is empty — check the errors/warnings above "
                "(corrupt ZIPs, missing/empty CSVs, or unparseable rows).")
    else:
        # Show the required output columns in a scrollable preview table.
        preview = result.combined[OUTPUT_COLUMNS]
        st.dataframe(preview, use_container_width=True, hide_index=True, height=420)
        st.caption(f"Showing all {len(preview):,} rows in the preview — "
                   "the downloaded file contains the same data.")

        # Download: serialize the DataFrame to CSV bytes once, in memory.
        buf = io.BytesIO()
        preview.to_csv(buf, index=False)
        st.download_button(
            label="⬇️ Download combined.csv",
            data=buf.getvalue(),
            file_name="combined.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    # First-load guidance when nothing has been processed yet.
    st.info("👈 Upload one or more ZIP files in the sidebar and press **Process**.")
