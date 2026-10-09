# CSVCompiler

Combine sensor data from multiple ZIP files (each containing CSVs, possibly in
nested folders) into one clean `combined.csv`.

## Web UI (Streamlit, localhost)

`app.py` is a localhost web UI for the same pipeline; all cleaning/combining
logic lives in `processor.py` (UI-free), and `combine_sensor_data.py` reuses it
for command-line batch runs.

Upload one or more ZIP files, choose duplicate handling
(**keep last** default / keep first / keep all, keyed on
`timestamp + kind + area`), press **Process**, preview the combined table and
download `combined.csv`. The summary shows ZIPs processed, CSVs found, rows
read/kept/dropped plus errors & warnings.

### Run instructions

For Windows:
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

For Mac/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Then open http://localhost:8501 in a browser.

## Command line

```bash
python combine_sensor_data.py <input_folder_with_zips> [output_dir]
```

- `input_folder` — folder containing the `.zip` files (default: current dir)
- `output_dir`   — where outputs are written (default: same as input folder)

## Outputs

| File                | Contents                                                        |
|---------------------|-----------------------------------------------------------------|
| `combined.csv`      | `timestamp, kind, area, value, source_zip, source_csv`          |
| `duplicates.csv`    | Rows dropped because of conflicting values for the same `(timestamp, kind, area)` |
| `processing_log.csv`| Per-file status: `source_zip, source_csv, status, rows_read, rows_kept, rows_dropped, error_message` |

If the combined result exceeds ~1,000,000 rows, the script instead writes one
CSV **per kind** (e.g. `BTU.csv`, `Temperature.csv`) with columns
`timestamp, area, value, source_zip, source_csv`, and `combined.csv` becomes a
small index mapping each kind to its file.

## Processing rules implemented

1. Reads every ZIP in the input folder, processed in sorted-filename order (deterministic).
2. Recursively finds all `.csv` entries inside each ZIP; ignores non-CSV files.
3. Header detection: if the first row looks like a header (`tag/title/name`,
   `value/reading`, `date`, `time` keywords), those columns are used — extra
   columns such as `11` and `Alm Disabled` are ignored. If headers are missing
   or unclear, positional columns are used: `0=title/tag, 1=value, 2=date, 3=time`.
4. Blank rows, separator rows, and rows with missing value/date/time are skipped.
5. Date + time are parsed into a single `timestamp` with
   `pd.to_datetime(..., errors="coerce")` (formats like `9/7/26` and
   `12:00:00 AM`); unparseable rows are dropped.
6. Values are converted to numeric; non-numeric values (`???`, `N/A`, …) are dropped.
7. `kind` is inferred from the ZIP filename via the `KIND_MAP` dictionary at the
   top of the script; if it cannot be inferred, `kind = "unknown"` and the file is logged.
8. `area` = CSV filename without extension; if that isn't useful, the title/tag
   column when it uniquely identifies the area; otherwise `source_csv`.
9. Exact duplicate rows on `(timestamp, kind, area, value)` are dropped.
   Conflicting values for the same `(timestamp, kind, area)` are written to
   `duplicates.csv`; the last processed row is kept in `combined.csv`.
10. Final output is sorted by `timestamp, kind, area`.
11. Error handling: corrupt ZIPs, unreadable/empty/encoding-broken CSVs are
    skipped and logged — the script never crashes on a single bad file.

## Configuration

Edit these constants near the top of `combine_sensor_data.py`:

- `KIND_MAP` — ZIP-name pattern → kind (substring match, case-insensitive)
- `IGNORE_COLUMNS` — always-dropped column names
- `MAX_COMBINED_ROWS` — per-kind split threshold (default 1,000,000)
