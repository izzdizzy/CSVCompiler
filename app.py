#!/usr/bin/env python3
"""
app.py — Streamlit UI for the ZIP -> combined CSV tool.

This file ONLY handles the user interface (upload, controls, status messages,
preview, download).  All data cleaning / combining logic lives in processor.py
so it can be reused by the CLI script (combine_sensor_data.py) and tests.

Run locally:
    Windows:      streamlit run app.py --server.port 8501
    Mac/Linux:    streamlit run app.py --server.port 8501
Then open http://localhost:8501 in a browser.
"""

import io

import pandas as pd
import streamlit as st

# All processing logic is imported from processor.py — no parsing code here.
from processor import DUPLICATE_MODES, OUTPUT_COLUMNS, process_zips

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ZIP to Combined CSV", page_icon="🗜️", layout="wide")

st.title("ZIP to Combined CSV")
st.caption(
    "Upload one or more ZIP files containing CSVs (nested folders allowed). "
    "The app cleans each CSV, merges everything into one tidy table with columns "
    "`timestamp, kind, area, value, source_zip, source_csv`, lets you preview it, "
    "and downloads it as `combined.csv`."
)

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
    process_clicked = st.button("Process", type="primary", use_container_width=True)

    st.divider()
    st.markdown(
        "**kind** is inferred from the ZIP filename (e.g. `btu` → BTU, "
        "`temp` → Temperature); otherwise `unknown`.  \n"
        "**area** is the CSV filename without extension; falls back to the "
        "tag/title column, then to `source_csv`."
    )

# ---------------------------------------------------------------------------
# Processing (triggered only by the Process button)
# ---------------------------------------------------------------------------
if process_clicked:
    if not uploads:
        # Basic guard: nothing uploaded yet.
        st.warning("Please upload at least one ZIP file before pressing Process.")
    else:
        # Build (name, bytes) pairs for the processor; reading the uploaded
        # files happens exactly once per file (no repeated loops).
        zip_items = [(u.name, u.getvalue()) for u in uploads]

        with st.status("Processing ZIP files…", expanded=True) as status:
            st.write(f"Received {len(zip_items)} ZIP file(s): "
                     + ", ".join(name for name, _ in zip_items))

            # ---- The entire cleaning/combining pipeline lives in processor.py ----
            try:
                result = process_zips(zip_items, duplicate_mode=duplicate_mode)
            except Exception as e:
                # Last-resort guard so the UI never shows a raw stack trace.
                status.update(label="Processing failed", state="error")
                st.error(f"Unexpected error while processing: {type(e).__name__}: {e}")
                st.stop()

            # Progress/status messages from the processor (warnings, notes…)
            for msg in result.messages:
                st.write(f"ℹ️ {msg}")
            for err in result.errors:
                st.write(f"⚠️ {err}")

            if result.combined.empty:
                status.update(label="No usable rows found", state="warning")
            else:
                status.update(
                    label=f"Done — {result.rows_kept:,} rows kept",
                    state="complete",
                    expanded=False,
                )

        # Cache the outcome so preview/download survive widget reruns.
        st.session_state["result"] = result

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

    # Errors / warnings block (collapsible so it doesn't dominate the page).
    problems = list(result.errors)
    if problems:
        with st.expander(f"⚠️ Errors / warnings ({len(problems)})", expanded=True):
            for p in problems:
                st.text(p)
    else:
        st.success("No errors — every readable CSV was processed.")

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
