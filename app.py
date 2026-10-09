#!/usr/bin/env python3
"""
app.py — Streamlit UI for the ZIP -> combined CSV tool.

This file ONLY handles the user interface (upload, controls, live progress,
status messages, validation report, preview, download).  All data cleaning /
combining logic lives in processor.py so it can be reused by the CLI script
(combine_sensor_data.py) and tests.

Live progress design (updated)
------------------------------
* Processing runs on a *background worker thread* so the Streamlit script
  (UI) thread is never blocked — the page keeps rendering and stays
  responsive while large archives are being crunched.
* The PRIMARY progress bar is based on COMPLETED FILES / TOTAL FILES — never
  on estimated row counts.  ETA is computed from the average wall-clock time
  per completed file.  Row counters are shown only as secondary information.
* After every file finishes, the tracker records the file's success/failure
  status and a timing sample; the UI repaints immediately for that file.
* The final combining/export stage emits visible log lines via
  ``ProgressTracker.stage()``; the poll loop renders them live so the UI
  never appears frozen while combined.csv is assembled.
* The UI thread polls ``tracker.snapshot()`` (immutable dict copies — no
  shared mutable state crosses threads besides the tracker's internal lock)
  and repaints with ``st.empty()`` placeholders.  All ``st.*`` calls happen
  on the main script thread: the safe Streamlit live-update pattern.

Tabs
----
* Process              — upload, start button, file-progress bar, current
                         file, ETA, live processing log.
* Validation / Compare — source vs combined row accounting, missing rows,
                         duplicate conflicts, missing minute gaps.
* Logs                 — detailed processing logs, warnings, errors and the
                         final export-stage logs.

Run locally:
    Windows:      streamlit run app.py --server.port 8501
    Mac/Linux:    streamlit run app.py --server.port 8501
Then open http://localhost:8501 in a browser.
"""

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
    validate_result,
    write_combined_csv,
)

# Manual-validation engine (standalone "Validation" tab): validates manually
# uploaded combined CSV file(s) against manually uploaded ZIP file(s) without
# depending on any session state from the Process tab.
from validation import (
    DEFAULT_VALIDATION_TOLERANCE,
    VALIDATION_PREVIEW_ROWS,
    df_to_csv_bytes,
    run_manual_validation,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ZIP to Combined CSV", page_icon="🗜️", layout="wide")

st.title("ZIP to Combined CSV")
st.caption(
    "Upload one or more ZIP files containing CSVs (nested folders allowed). "
    "The app cleans each CSV, merges everything into one tidy table with columns "
    "`timestamp, kind, area, value, source_zip, source_csv`, shows live "
    "**file-completion** progress with ETA while processing, validates the "
    "combined output against the sources, and downloads it as `combined.csv`."
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


def _manual_validation_worker(combined_uploads, zip_items, tolerance, tracker):
    """Run the MANUAL validation on a background thread (Validation tab).

    Uses ONLY the files handed to this call — never any Process-tab session
    state — so results depend solely on what was uploaded in the Validation
    section.  The finished ManualValidationReport is stashed atomically under
    "manual_val_result" once the 'manual_val_done' flag is set.
    """
    try:
        report = run_manual_validation(combined_uploads, zip_items,
                                       tolerance=tolerance, progress=tracker)
        st.session_state["manual_val_result"] = report
    except Exception as e:
        # Last-resort guard: surface unexpected errors instead of hanging the UI.
        st.session_state["manual_val_error"] = f"{type(e).__name__}: {e}"
    finally:
        st.session_state["manual_val_done"] = True


def _render_progress(tracker):
    """Draw one frame of the live progress UI from a tracker snapshot.

    Returns the snapshot dict so callers can inspect terminal conditions.
    Uses pre-created placeholder containers so repeated repaints don't grow
    the page.  The primary bar is FILE-COMPLETION based (completed files /
    total files); rows are secondary info only.
    """
    snap = tracker.snapshot()
    bar = st.session_state["_pb_bar"]
    txt = st.session_state["_pb_text"]
    cur = st.session_state["_pb_current"]
    rows = st.session_state["_pb_rows"]
    stage_box = st.session_state["_pb_stage"]

    fraction = snap["fraction"]
    done = snap["files_done"]
    total_files = snap["total_files"]
    pct_txt = f" · {fraction * 100:.0f}%" if fraction is not None else ""

    if fraction is None:
        # Indeterminate state: total file count not known yet (pre-scan
        # still running).  Show an animated bar rather than failing/lying.
        bar.progress(None, text="Scanning ZIP archives… counting files")
    else:
        bar.progress(fraction,
                     text=(f"Files completed: **{done} of {total_files}**{pct_txt}"))

    # ---- Live status line: elapsed / ETA (avg time per file) ----
    parts = [f"🧾 Completed files: **{done}/{total_files or '?'}**",
             f"✅ {fraction * 100:.1f}% complete" if fraction is not None
             else "⏳ Estimating…"]
    parts.append(f"⏱️ Elapsed **{format_duration(snap['elapsed']) or '0s'}**")
    if snap["eta_seconds"] is not None:
        spf = snap.get("avg_seconds_per_file")
        spf_txt = f" (avg {spf:.1f}s/file)" if spf else ""
        parts.append(f"🔮 ETA **{format_duration(snap['eta_seconds'])}**{spf_txt}")
    else:
        parts.append("🔮 ETA calculating… (needs ≥1 finished file)")
    txt.markdown("  |  ".join(parts))

    # ---- Current file line (+ last file's success/failure status) ----
    status_icons = {"ok": "✅ ok", "skipped": "⚠️ skipped", "error": "❌ error"}
    last_status = status_icons.get(snap["last_file_status"], "")
    if snap["current_zip"] or snap["current_csv"]:
        idx = min(done + 1, total_files) if total_files else done + 1
        counter = (f"Processing file **{idx} of {total_files}**"
                   if total_files else f"Processing file **{idx}**")
        last_line = (f"  \\nLast file: {last_status}" if last_status else "")
        cur.markdown(
            f"{counter}  \\n"
            f"📦 ZIP: `{snap['current_zip'] or '—'}`  \\n"
            f"📄 CSV: `{snap['current_csv'] or '—'}`  \\n"
            f"Rows scanned in this file: **{snap['current_rows']:,}**"
            + last_line
        )
    else:
        cur.markdown("Opening ZIP archive(s)…")

    # ---- Secondary info: row counts (NOT the progress basis) ----
    kept, dropped = snap["rows_kept"], snap["rows_dropped"]
    seen = snap["rows_seen"]
    total_hint = snap["total_rows_hint"]
    hint_txt = f" (~{total_hint:,} est.)" if total_hint else ""
    rows.markdown(
        f"_Secondary info — rows:_ scanned **{seen:,}{hint_txt}**  |  "
        f"✅ valid so far: **{kept:,}**  |  ❌ dropped so far: **{dropped:,}**"
    )

    # ---- Final-stage visibility: live log lines during combining/export ----
    stage_logs = snap["stage_logs"]
    if stage_logs:
        tail = stage_logs[-6:]  # keep the box compact; full history in Logs tab
        lines = "\n\n".join(f"`[{t:6.1f}s]` {m}" for t, m in tail)
        stage_box.markdown("**🛠 Final stage (live):**\n" + lines)
    else:
        stage_box.markdown("")
    return snap


def _render_live_log(tracker, container, n_lines: int = 12):
    """Show the tail of the live stage/processing log feed in `container`."""
    logs = tracker.snapshot()["stage_logs"]
    if not logs:
        container.code("(waiting for log output…)", language="log")
        return
    tail = logs[-n_lines:]
    text = "\n".join(f"[{t:7.1f}s] {m}" for t, m in tail)
    container.code(text, language="log")


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

    # Requirement: an explicit Process/start button (nothing runs until clicked).
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
# Tabs: Process | Validation (manual uploads) | Validation / Compare | Logs
# ---------------------------------------------------------------------------
tab_process, tab_manual_validation, tab_validation, tab_logs = st.tabs(
    ["🛠 Process", "✅ Validation", "🔍 Validation / Compare", "📜 Logs"])

# ---------------------------------------------------------------------------
# Processing (triggered only by the Process button; runs on a worker thread)
# ---------------------------------------------------------------------------
with tab_process:
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
            st.session_state["_pb_stage"] = st.empty()
            live_log_box = st.empty()   # live processing log inside the Process tab
            cancel_note = st.empty()

            st.session_state["run_thread"].start()

            # ---- Poll loop on the MAIN thread: repaint ~5x/sec, UI stays alive ----
            while True:
                _render_progress(tracker)
                # Visible live log — also covers the final combining/export stage
                # so the page never looks frozen while combined.csv is built.
                _render_live_log(tracker, live_log_box)
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
            st.session_state["_pb_stage"].empty()
            live_log_box.empty()
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
                        # Clear completion message including the export stage time.
                        st.success(
                            f"✅ Done — processing took {format_duration(result.elapsed_seconds)} "
                            f"(final export stage: {format_duration(result.export_elapsed_seconds) or '0s'}); "
                            f"{result.rows_kept:,} rows in combined.csv.")

# ---------------------------------------------------------------------------
# Shared result object (populated after a run)
# ---------------------------------------------------------------------------
result = st.session_state.get("result")

# ---------------------------------------------------------------------------
# Process tab (post-run): summary + preview + download
# ---------------------------------------------------------------------------
with tab_process:
    if result is not None:
        st.subheader("Summary")

        # Validation report is cached on first access (recomputed only if the
        # result object changes) so switching tabs doesn't redo the work.
        if st.session_state.get("_val_for") is not id(result):
            try:
                st.session_state["validation_report"] = validate_result(result)
            except Exception as e:
                st.session_state["validation_report"] = None
                st.session_state["validation_error"] = f"{type(e).__name__}: {e}"
            st.session_state["_val_for"] = id(result)
        report = st.session_state.get("validation_report")

        # Completion banner + validation pass/fail right in the Process tab.
        if not result.cancelled and not result.combined.empty:
            if report is not None:
                if report.passed:
                    st.success("🟢 Validation PASSED — no missing rows or unexplained "
                               "differences between sources and combined.csv.")
                else:
                    st.error("🔴 Validation FAILED — see the Validation / Compare tab "
                             "for details (you can still inspect and download the output).")

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
            basis = {"files": "avg time per file"}.get(tracker.last_eta_basis, "n/a")
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
            st.caption(f"Preview shows the first/last rows of "
                       f"{len(preview):,} combined rows — the downloaded file "
                       "contains the same data.")

            # Download: serialize the DataFrame to CSV bytes ONCE (chunked
            # writer from processor.py; cached in session_state so tab
            # switches don't re-serialize a potentially huge table).
            if st.session_state.get("_csv_bytes_for") != id(result):
                st.session_state["_csv_bytes"] = write_combined_csv(result.combined)
                st.session_state["_csv_bytes_for"] = id(result)
            st.download_button(
                label="⬇️ Download combined.csv",
                data=st.session_state["_csv_bytes"],
                file_name="combined.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        # First-load guidance when nothing has been processed yet.
        st.info("👈 Upload one or more ZIP files in the sidebar and press **Process**.")

# ---------------------------------------------------------------------------
# Validation / Compare tab
# ---------------------------------------------------------------------------
with tab_validation:
    if result is None:
        st.info("Run the processor first — validation compares the combined "
                "output against the original ZIP/CSV sources.")
    else:
        report = st.session_state.get("validation_report")
        if report is None:
            st.error("Validation could not run: "
                     f"{st.session_state.get('validation_error', 'unknown error')}")
        else:
            st.subheader("Validation summary")

            # Pass/fail banner (requirement: clear status + issues listed anyway).
            if report.passed:
                st.success("🟢 VALIDATION PASSED — every parsed source row is "
                           "accounted for in combined.csv.")
            else:
                st.error("🔴 VALIDATION FAILED — issues listed below. You can "
                         "still inspect the logs and download the output.")

            if report.issues:
                with st.expander(f"❗ Issues ({len(report.issues)})", expanded=True):
                    for iss in report.issues:
                        st.markdown(f"- {iss}")
            if report.notes:
                for n in report.notes:
                    st.caption(f"ℹ️ {n}")

            # ---- Source vs combined row accounting ----
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total source rows (expected)", f"{report.total_source_expected:,}")
            m2.metric("Valid parsed source rows", f"{report.total_source_parsed:,}")
            m3.metric("Total combined rows", f"{report.total_combined:,}")
            diff_ok = (report.row_count_difference ==
                       report.exact_duplicates_removed + report.conflict_rows_removed)
            m4.metric("Row count difference (parsed − combined)",
                      f"{report.row_count_difference:,}",
                      delta="explained by dedupe ✅" if diff_ok else "UNEXPLAINED ❌")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Artifact rows ignored", f"{report.total_source_dropped:,}")
            m6.metric("Duplicate rows detected", f"{report.duplicates_detected:,}")
            m7.metric("Exact duplicates removed", f"{report.exact_duplicates_removed:,}")
            m8.metric("Conflict groups (diff. values)", f"{report.conflict_groups:,}")

            m9, m10 = st.columns(2)
            m9.metric("Rows w/ invalid date/time", f"{report.rows_with_invalid_timestamp:,}")
            m10.metric("Rows w/ missing/invalid value", f"{report.rows_with_invalid_value:,}")

            # ---- Uploaded ZIPs & CSVs found inside each ----
            st.markdown("#### Uploaded ZIP files & CSVs found inside")
            if not report.zip_list.empty:
                st.dataframe(report.zip_list, use_container_width=True, hide_index=True)
            else:
                st.info("No ZIP inventory available (nothing processed).")

            # ---- Per-source-file expectations vs results ----
            st.markdown("#### Per source file: expected / parsed / dropped rows, "
                        "min & max timestamps")
            if not report.per_file.empty:
                st.dataframe(report.per_file, use_container_width=True,
                             hide_index=True, height=320)
                bad_acct = report.per_file[~report.per_file["accounted"]]
                if not bad_acct.empty:
                    st.error(f"{len(bad_acct)} file(s) have unaccounted rows — "
                             "see the 'accounted' column.")
            else:
                st.info("No per-file accounting available.")

            # ---- Missing rows / reconciliation ----
            st.markdown("#### Missing rows check")
            unexplained = (report.row_count_difference
                           - report.exact_duplicates_removed
                           - report.conflict_rows_removed)
            if result.cancelled:
                st.warning("Run was cancelled — the combined output is intentionally "
                           "incomplete; no missing-row conclusion drawn.")
            elif unexplained > 0:
                st.error(f"❌ {unexplained:,} parsed source row(s) are MISSING from "
                         "combined.csv and cannot be explained by duplicate removal.")
            elif unexplained < 0:
                st.error(f"❌ combined.csv contains {-unexplained:,} more row(s) than "
                         "the source parse count — please inspect the Logs tab.")
            else:
                st.success(f"✅ No missing rows: parsed source rows "
                           f"({report.total_source_parsed:,}) = combined rows "
                           f"({report.total_combined:,}) + exact duplicates "
                           f"({report.exact_duplicates_removed:,}) + resolved conflicts "
                           f"({report.conflict_rows_removed:,}).")

            # ---- Duplicate conflicts ----
            st.markdown("#### Duplicate timestamp+area conflicts "
                        "(same key, different value)")
            if not report.duplicate_conflicts.empty:
                st.warning(f"{report.conflict_groups} conflict group(s) detected; "
                           f"{report.conflict_rows_removed} row(s) resolved via "
                           f"'{result.duplicate_mode}'. Rows involved:")
                st.dataframe(report.duplicate_conflicts, use_container_width=True,
                             hide_index=True, height=280)
            else:
                st.success("No conflicting duplicates (exact duplicates were "
                           "collapsed safely where applicable).")

            # ---- Missing minute gaps ----
            st.markdown("#### Missing minute gaps per (kind, area)")
            if not report.coverage_checked:
                st.info("No data available for coverage analysis.")
            elif report.coverage_gaps.empty:
                st.success("✅ No missing minutes detected — every (kind, area) series "
                           "looks continuous (one row per minute).")
            else:
                st.warning(f"⚠️ {int(report.coverage_gaps['missing_minutes'].sum()):,} "
                           "missing minute(s) across "
                           f"{len(report.coverage_gaps)} series — likely sensor "
                           "dropout in the source data, not a processing bug:")
                st.dataframe(report.coverage_gaps, use_container_width=True,
                             hide_index=True, height=280)

# ---------------------------------------------------------------------------
# Logs tab: detailed processing logs, warnings, errors, final-export stages
# ---------------------------------------------------------------------------
with tab_logs:
    tracker = st.session_state.get("run_tracker")
    if result is None and tracker is None:
        st.info("Logs appear here once you press **Process** (live during the run, "
                "full history afterwards).")
    else:
        st.subheader("Detailed processing logs")

        # Rebuild the ordered log feed: stage logs (which include mirrored
        # processing notes + final export steps) plus messages/errors.
        stage_logs = tracker.snapshot()["stage_logs"] if tracker is not None else []
        if stage_logs:
            lines = [f"[{t:7.1f}s] {m}" for t, m in stage_logs]
            st.code("\n".join(lines), language="log")
        elif result is not None:
            st.code("(stage log feed unavailable — showing result messages below)",
                    language="log")

        if result is not None:
            if result.messages:
                with st.expander(f"📝 Processing messages ({len(result.messages)})",
                                 expanded=False):
                    for m in result.messages:
                        st.text(m)

            warn_rows = [r for r in result.per_file_log
                         if r["status"] in ("skipped", "error", "no_csv")]
            with st.expander(f"⚠️ Warnings / skipped files ({len(warn_rows)})",
                             expanded=bool(warn_rows)):
                if warn_rows:
                    st.dataframe(pd.DataFrame(warn_rows), use_container_width=True,
                                 hide_index=True)
                else:
                    st.text("None.")

            with st.expander(f"❌ Errors ({len(result.errors)})",
                             expanded=bool(result.errors)):
                if result.errors:
                    for e in result.errors:
                        st.text(e)
                else:
                    st.text("None.")

            # Final export-stage logs called out separately for clarity.
            export_logs = [f"[{t:7.1f}s] {m}" for t, m in stage_logs
                           if any(k in m for k in
                                  ("Concatenat", "duplicate", "conflict", "Sorting",
                                   "Validation", "writing combined", "Finished",
                                   "Optimizing", "Filtering", "Running"))]
            with st.expander(f"🛠 Final export stage logs ({len(export_logs)})",
                             expanded=True):
                st.code("\n".join(export_logs) if export_logs
                        else "(no export-stage logs recorded)", language="log")
