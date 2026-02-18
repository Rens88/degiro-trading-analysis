# AGENTS.md - Streamlit SSC template playbook

## Purpose
Provide a standalone, agent-ready specification for generating consistent Streamlit dashboards with Plotly, an explicit FSM workflow, strict filtering behavior, and export/reload capability. This document must be sufficient to scaffold a new dashboard in an empty folder without referencing any external codebases.

## How to use this file
- Treat this as the authoritative build spec for any new Streamlit dashboard in this folder.
- Follow the checklist at the end to scaffold files, implement workflow, and wire UI.
- Ask the user only minimal questions if required; otherwise, choose sensible defaults.

## Language policy
- Use English by default for all UI text, documentation text, and code comments.
- Only use another language when the user explicitly requests it.
- This applies to both newly generated content and edits to existing content.

**How to generate a new dashboard from a user description**
1) Parse the description into: inputs (file types), columns/metrics, time axis, and expected outputs (plots/tables).
2) Decide a minimal `schema.py` default configuration (filters, view range, labels).
3) Implement the read → preprocess → process → plot pipeline in the matching modules.
4) Implement the FSM and session state keys exactly as specified below.
5) Add filtering controls for each variable and ensure downstream processing uses filtered series.
6) Create at least one Plotly time-series plot and one table view.
7) Add export/reload support that reproduces UI and plots from a zip bundle.
8) Apply logos to sidebar/main panel (from `assets/` using `sidepanellogo_` / `mainpanellogo_` prefixes) and ensure layout rules are followed.

Minimal questions (ask only if absolutely necessary)
- What file types should be accepted (CSV only, or CSV/XLSX)?
- Which column is the time axis (or should it be auto-detected)?
- Which metrics should be plotted by default?
If the user does not specify, assume: CSV input, auto-detect time column, and plot all numeric metrics.

## Project structure
Required layout (all Python code in `src/`):
```
<project_root>/
  AGENTS.md
  assets/
    sidepanellogo_<name>.png
    mainpanellogo_<name>.png
  src/
    app.py
    schema.py
    read_data.py
    preprocessing.py
    processing.py
    plotting.py
    export_data.py
```
Module responsibilities (strict)
- `schema.py`: typed defaults/config schema; all default values live here.
- `read_data.py`: ingest + interpret uploads; normalize columns; validate; no UI code.
- `preprocessing.py`: generic operations (filtering, resampling, cleaning); no UI code.
- `processing.py`: project-specific computations derived from preprocessed data; no UI code.
- `plotting.py`: Plotly figures only; no data mutations.
- `export_data.py`: export/import zip bundle; no UI code.
- `app.py`: Streamlit UI + orchestration; imports functions from the modules above.

## FSM workflow
The FSM is mandatory and must cover these phases (projects may use additional or more granular states):
1) Loading data
2) Selecting params
3) Processing
4) Viewing results
5) Exporting
6) Reloading export

Recommended state enum names (use these by default; project-specific state names are allowed as long as you can map them to the phases above for gating)
```
STATE_LOADING_DATA
STATE_SELECTING_PARAMS
STATE_PROCESSING
STATE_VIEWING_RESULTS
STATE_EXPORTING
STATE_RELOADING_EXPORT
```

Transition helper (must be used for all state changes)
```python
from datetime import datetime
import streamlit as st

def transition(next_state: str, reason: str) -> None:
    if st.session_state.get("fsm_state") != next_state:
        st.session_state["fsm_state"] = next_state
        st.session_state["workflow"].setdefault("nav", {})["last_transition"] = {
            "state": next_state,
            "reason": reason,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
```

Gating logic (sidebar sections are enabled by state)
- Loading data controls: always enabled.
- Params controls: enabled only at/after the "Selecting params" phase (recommended: `STATE_SELECTING_PARAMS`).
- Processing trigger: enabled only at/after the "Selecting params" phase (recommended: `STATE_SELECTING_PARAMS`).
- Results/plots: shown only at/after the "Viewing results" phase (recommended: `STATE_VIEWING_RESULTS`).
- Export controls: enabled only at/after the "Exporting" phase (recommended: `STATE_EXPORTING`).
- Reload export: set state to the "Reloading export" phase (recommended: `STATE_RELOADING_EXPORT`) and rehydrate data + params.

If you use project-specific state names, you MUST implement a deterministic mapping from your states to these phases and use that mapping for UI gating.

## Session state design
Recommended keys (minimum set)
- `st.session_state["workflow"]`: dict storing data, params, results, and export metadata.
- `st.session_state["fsm_state"]`: current FSM state.
- `st.session_state["upload_sig"]`: tuple `(name, size)` for upload change detection.
- `st.session_state["params_sig"]`: JSON signature of current params.
- `st.session_state["processed_sig"]`: `(upload_sig, params_sig)` cache key.
- `st.session_state["export_bytes"]`: last export zip bytes.

Rules
- If `upload_sig` changes: reset workflow, clear export bytes, transition to `STATE_LOADING_DATA`.
- If params change: clear export bytes and invalidate processed cache.
- Store processed outputs in `workflow["processed"]` and summaries in `workflow["summaries"]`.

## Data loading: new vs zip
Two loading modes are mandatory:
- New data (fresh ingest)
- Reload previous export (.zip)

Required lifecycle
Load data → adjust params → process → export analysis as .zip → reload that export and reproduce results

Recommended loader behavior
- New data:
  - Parse upload in `read_data.py`.
  - Normalize column names and validate required columns.
  - Write `workflow["data"]["df"]` and set state to `STATE_SELECTING_PARAMS`.
- Reload export:
  - Load `config/config.json` and `data/*.csv` from zip.
  - Restore params to session state.
  - Rebuild UI and plots to match the export.
  - Set state to `STATE_RELOADING_EXPORT` then to `STATE_VIEWING_RESULTS` after plots render.

## Filtering (scientific)
- Filtering must follow the ordered pipeline: **(1) gap-aware segmentation + resample to a uniform grid using median dt, (2) Butterworth SOS + filtfilt on the resampled series, (3) map filtered values back to the original timestamps, (4) skip/return raw for segments that are too short or invalid.**
- Gap handling: if `max_gap_seconds` is provided use it, otherwise split segments where `Δt > max_gap_factor * dt_median` (default factor 5). Never interpolate across segments.
- Resampling: build `tu = np.arange(t0, tN + eps, dt)` per segment; interpolate only within that segment using finite points. **Do not ffill/bfill across gaps; never fill with zeros globally.**
- Filtering: clamp `fc` to `0.99 * Nyquist` (Nyquist from `fs = 1/dt_median`), use `butter(order, fc, fs, output="sos")` + `sosfiltfilt`. Catch errors and fall back to raw.
- Map-back: interpolate filtered uniform values onto the original timestamps of the segment; preserve `NaN` values at their original positions.
- Skipping rules: if `fc<=0`, invalid dt/fs, segment duration < `min_segment_seconds`, or samples < `max(20, 12*order)` on the uniform grid → return raw for that segment and record diagnostics.

UI control for each variable (inline, strict layout)
```python
cols = st.columns([1, 4])
with cols[0]:
    enabled = st.checkbox(
        "<var>",
        key="<var>_filter_enabled",
        help="Enable low-pass Butterworth filter",
    )
with cols[1]:
    cutoff = st.number_input(
        "Low-pass cutoff (Hz)",
        key="<var>_filter_cutoff",
        min_value=0.01,
        disabled=not enabled,
    )
    st.caption("Detected fs: <xx.x> Hz (dt=<yy ms>)")  # update dynamically per data
```

Behavior (strict)
- Do not use empty widget labels (Streamlit warns and may error in the future). Use a non-empty label and hide it with `label_visibility="collapsed"`.
- Raw data is always shown.
- Checkbox ON:
  - Compute filtered series via the resample→filter→map-back pipeline above.
  - Show filtered data AND raw data (raw is highly transparent).
  - All downstream processing uses the filtered series.
- Checkbox OFF:
  - Show raw data only (not transparent).
  - Downstream processing uses raw series.
  - Cutoff input must be disabled (read-only) while filtering is OFF.

Data-handling pattern (conceptual)
```python
raw = df[var].to_numpy()
t = time_seconds  # from time column / index
filtered = butter_filter_series(raw, fc=cutoff, time_index=t) if enabled else raw
raw_color = BASE_BLUE if not enabled else "rgba(1,55,138,0.25)"

# plot raw always
fig.add_trace(go.Scatter(x=t, y=raw, name=f"{var} raw", line=dict(color=raw_color)))

# plot filtered only if enabled
if enabled:
    fig.add_trace(go.Scatter(x=t, y=filtered, name=f"{var} filtered", line=dict(color=BASE_BLUE)))

# downstream must use filtered if enabled
series_for_processing = filtered if enabled else raw
```

## Processing pipeline
Required flow
1) `read_data.py`: parse/validate, normalize columns
2) `preprocessing.py`: clean, resample, filter helpers
3) `processing.py`: domain-specific metrics/aggregations
4) `plotting.py`: Plotly figures

Rules
- No Streamlit calls outside `app.py`.
- All computation must be deterministic given inputs + params.
- Store outputs in `workflow["processed"]` and `workflow["summaries"]` for export.

## Plotting standard
Must use Plotly.

Required colors
```
BASE_BLUE = "#01378A"
BASE_RED = "#E1011A"
BASE_ORANGE = "#EA6D08"
BASE_YELLOW = "#F4C300"  # optional
BASE_GREEN = "#009F3D"   # optional
```

Layout requirements
- Use `template="plotly_white"`.
- Use `hovermode="x unified"` for time series.
- Use `st.plotly_chart(fig, width="stretch")`.
- For multi-panel charts, use `make_subplots`.
- Export figures to HTML and PNG (PNG requires `kaleido`).

## Export bundle format
Zip contents (mandatory)
```
config/
  config.json          # params, fsm_state, version, and summary
plots/
  <name>.html          # Plotly HTML
  <name>.png           # Plotly PNG (kaleido)
data/
  data.csv             # data used for plots
README.txt             # reload instructions
```

Export rules
- Always include config + data + plots + README.
- If PNG export fails, warn and continue with HTML.
- Reloading the zip must restore params + data and reproduce plots.
- You MAY include additional project-specific files/folders in the zip (for example `ams/*.csv`). Extra files must not replace or omit the mandatory contents above.

## Progress + logging
Mandatory logging and progress behavior
- Print timestamped logs for each major step (load, preprocess, process, plot, export).
- Show UI progress for long steps (spinner + status messages).

Reusable log() snippet
```python
from datetime import datetime
import time

def log(message: str, start_time: float) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    elapsed = time.perf_counter() - start_time
    line = f"[{now}] [+{elapsed:.2f}s] {message}"
    print(line)
    return line
```

## Worked example: wearable dashboard
User request: “Generate a Streamlit app that reads wearable data and shows each metric over time, let the user select timeframe (number of days) plotted.”

Implementation outline
- `schema.py`: defaults for `time_window_days=7`, `filter_cutoffs` per variable, `timezone="UTC"`.
- `read_data.py`: accept CSV/XLSX, auto-detect time column (e.g., `timestamp`, `time`, `datetime`), normalize to `time` + numeric metric columns.
- `preprocessing.py`: convert time to datetime, create `time_in_seconds`, sort, and provide low-pass filter helper.
- `processing.py`: compute summary stats per metric (mean, max, min) over the selected window.
- `plotting.py`: build Plotly time-series lines for each metric with unified hover.
- `app.py`: sidebar controls (uploaders, timeframe selector, per-metric filters, export/reload); main panel shows plots + a summary table.

Minimum outputs
- One Plotly time-series plot (multi-metric).
- One table with summary stats for the selected timeframe.

Defaults (if user does not specify)
- Input: CSV
- Time column: auto-detect
- Metrics: all numeric columns
- Timeframe: last 7 days

## Common pitfalls
- Applying filters only in plots but not in downstream processing.
- Forgetting to reset processed cache when params change.
- Not exporting config/data along with plots.
- Hiding raw data when filters are ON (must show raw, transparent).
- Placing logic in `app.py` that belongs in processing modules.

## Future: packaging and run scripts
Do not generate these files automatically now. Use the following content when requested.

Typical `requirements.txt`
```
streamlit
pandas
numpy
plotly
kaleido
pydantic
scipy  # only if using Butterworth filtering
openpyxl
```

`run_app.bat` (Windows)
```bat
@echo off
setlocal

echo Locating Python...
set "ROOT=%~dp0"
cd /d "%ROOT%" || goto :cd_error

set "PYTHON="
py -3 -c "import sys" >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON=py -3"
) else (
    python -c "import sys" >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON=python"
    ) else (
        echo Python 3 is required but was not found. Please install Python 3.x and try again.
        goto :fail
    )
)

if not exist ".venv\\Scripts\\python.exe" (
    echo Creating virtual environment in .venv (could take a minute)...
    %PYTHON% -m venv ".venv" || goto :fail
)

call ".venv\\Scripts\\activate.bat" || goto :fail

echo Installing requirements (will take about 5 mins first time, internet required)...
python -m pip install -r requirements.txt --quiet || goto :fail

echo Starting Streamlit app...
python -m streamlit run src/app.py
if %errorlevel% neq 0 goto :fail

endlocal
goto :eof

:cd_error
echo ERROR: Could not change directory to the app folder.
goto :pause_fail

:fail
echo.
echo ERROR: Something went wrong. Try deleting the .venv folder and double-clicking again.
:pause_fail
echo.
pause
```

`run_app.command` (macOS)
```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

pause_with_message() {
  echo
  echo "$1"
  read -rp "Press Enter to close..." _
}

if ! cd "$SCRIPT_DIR"; then
  pause_with_message "ERROR: Could not change directory to the app folder."
  exit 1
fi

echo "Locating Python..."
if ! command -v python3 >/dev/null 2>&1; then
  pause_with_message "Python 3 is required but was not found. Please install Python 3.x and try again."
  exit 1
fi

trap 'pause_with_message "ERROR: Something went wrong. Try deleting the .venv folder and double-clicking again."' ERR

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment in .venv (could take a minute)..."
  /usr/bin/env python3 -m venv .venv
fi

source ".venv/bin/activate"

echo "Installing requirements (will take about 5 mins first time, internet required)..."
python -m pip install -r requirements.txt --quiet

echo "Starting Streamlit app..."
python -m streamlit run src/app.py
```

## Checklist for a new dashboard
- [ ] Create folders `assets/` and `src/` with the required files.
- [ ] Define defaults and schema in `schema.py`.
- [ ] Implement upload parsing + normalization in `read_data.py`.
- [ ] Implement preprocessing helpers in `preprocessing.py`.
- [ ] Implement domain logic in `processing.py`.
- [ ] Implement Plotly figures in `plotting.py`.
- [ ] Implement zip export/import in `export_data.py`.
- [ ] Wire the FSM and sidebar gating in `app.py`.
- [ ] Add filtering controls for each variable (20/80 layout) and enforce downstream use of filtered data.
- [ ] Create at least one Plotly plot and one summary table.
- [ ] Implement export/reload lifecycle and verify UI/plots match after reload.
- [ ] Add logo images in `assets/` using prefixes `sidepanellogo_` and `mainpanellogo_`.
