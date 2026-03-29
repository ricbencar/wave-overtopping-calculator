#!/usr/bin/env python3
"""
Tkinter graphical user interface for the overtopping neural-network predictor.

OBJECTIVE
=========
This script provides a desktop interface for the overtopping prediction workflow
implemented in train.py. Its purpose is to make the model usable without manual
command-line interaction for the most common engineering tasks:

1. Single-case prediction
   - Enter one scenario manually through the GUI.
   - Evaluate the predicted overtopping discharge immediately.
   - Review ensemble statistics and extrapolation warnings.

2. Batch prediction
   - Read many cases from an external input file.
   - Run all scenarios through the trained model.
   - Export the resulting prediction table to CSV.

3. Model training / refresh
   - Train a new model directly from the external database CSV.
   - Save the trained model bundle to disk.
   - Regenerate diagnostics JSON and associated plots.

4. Calibration-domain review
   - Compare all current numeric inputs against the observed parameter ranges
     from the database used for training.
   - Warn when a scenario is outside the database domain and therefore
     represents an extrapolation.

This GUI is intended to be practical for engineering use, while preserving
traceability through log messages, explicit file paths, persisted defaults, and
clear separation between input data, trained model, diagnostics, and exported
results.

BACKEND METHOD SUMMARY
======================
The GUI is aligned with the updated train.py backend, which uses a two-stage
workflow:

1. Holdout validation stage
   - The valid dataset is split so that a subset is withheld from training.
   - That holdout subset is used to evaluate predictive performance on unseen
     data.
   - Holdout plots and performance metrics help assess generalisation quality.

2. Full-data refit stage
   - After validation, the final model is refit using the full valid dataset.
   - The saved model.joblib therefore corresponds to the refitted production
     model rather than only the holdout-training subset.
   - Diagnostics include both holdout information and full-data refit outputs.

This distinction is important:
- holdout diagnostics indicate validation performance;
- full-data refit represents the final model actually used by the GUI for
  prediction.

The backend target definition used by train.py is:
- physical dimensionless target: sq = q / sqrt(g * Hm0_toe^3)
- internal training target: y = (log10(sq) - mu) / sigma

The GUI reads, displays, and uses these backend outputs through:
- model.joblib          -> trained model bundle used for prediction;
- diagnostics.json      -> summary of counts, metrics, and plot paths;
- database.csv          -> source data used when training or refreshing;
- defaults.json         -> persistent GUI state file;
- input.txt             -> default startup batch-prediction input file.

MAIN CAPABILITIES
=================
The GUI supports:
- automatic startup loading of an existing model if the selected model file is
  present;
- single-case prediction;
- batch prediction from pipe-separated, tab-separated, comma-separated, or
  semicolon-separated files;
- model training / refresh from the selected external database file;
- display of database parameter ranges;
- display of diagnostics counts and fit metrics;
- persistent GUI defaults stored in defaults.json;
- saving of single-case and batch prediction outputs to CSV files;
- logging of actions, errors, and key model information.

PHYSICAL INPUTS
===============
The GUI works with the overtopping input variables expected by train.py. These
represent wave, water-level, toe, slope, berm, and crest conditions of the
structure. The input controls are organised into groups to make engineering
review easier:

1. Case / Wave
   - Case name
   - Hm0 toe
   - Tm-1,0 toe
   - m
   - h
   - beta

2. Toe / Slope
   - ht
   - Bt
   - gf
   - cotad
   - cotau

3. Berm / Crest
   - B
   - hb
   - Rc
   - Ac
   - Gc

The case name is only an identifier written to outputs. All other fields are
numeric model inputs.

HOW THE GUI SHOULD BE USED
==========================
A. Single prediction
--------------------
Use the "Single Prediction" tab when you want to analyse one scenario.

Recommended workflow:
1. Enter or review the physical input values.
2. Confirm the file paths for:
   - model.joblib
   - database.csv
   - diagnostics.json
   - single prediction output CSV
3. Press "Predict q".
4. Review:
   - mean overtopping discharge;
   - P05 / P50 / P95 ensemble indicators;
   - SI-unit conversion;
   - warning panel;
   - current single-case input review table.
5. Save the result if required.

Interpretation of the main outputs:
- Average q [l/s/m]
  Main predicted overtopping discharge in litres per second per metre.

- P05 [l/s/m]
  Lower ensemble estimate. Useful as a conservative low-side reference.

- P50 [l/s/m]
  Central ensemble estimate, often interpreted as a median-type indicator.

- P95 [l/s/m]
  Upper ensemble estimate, useful for assessing spread or uncertainty.

- Average q [m³/s/m]
  The same average discharge expressed in SI units.

- Range warning
  Indicates whether one or more current input values lie outside the observed
  database range used to calibrate the model.

B. Batch prediction
-------------------
Use the "Batch Prediction" tab when you want to run multiple scenarios in one
operation.

Supported input styles:
- pipe-separated text;
- tab-separated text / TSV;
- CSV;
- semicolon-separated text;
- other delimiter-based text formats that can be parsed by backend helpers.

Expected content:
- Each row is one scenario.
- Columns should correspond to the same physical variables expected by train.py.
- The script normalises some common aliases where possible.
- If the file does not contain a "Name" column, default scenario names are
  generated automatically.

Recommended workflow:
1. Select the batch input file.
2. Confirm the batch output CSV path.
3. Press "Predict from batch file".
4. Inspect the preview table.
5. Save the batch result if required.

Default startup batch file:
- input.txt

C. Model training / refresh
---------------------------
Use the training function when:
- no model.joblib exists yet;
- the database has changed;
- you want updated diagnostics;
- you want to rebuild the model from scratch.

Workflow:
1. Select the desired database.csv path.
2. Confirm the model output and diagnostics JSON paths.
3. Press "Train / refresh model".
4. The GUI calls the backend training routine.
5. The trained bundle is saved.
6. Diagnostics are written.
7. The new model is loaded into the GUI.

Important counts commonly shown in the diagnostics summary:
- valid rows
  Number of database rows retained after backend validity filtering.

- n_holdout
  Number of rows reserved for holdout validation diagnostics.

- n_full
  Number of rows used to refit the final saved model on the full valid dataset.

- R²(log10 sq)
  Fit quality in transformed space.

- R²(q l/s/m)
  Fit quality in overtopping discharge units.

PARAMETER RANGE REVIEW
======================
The GUI includes a dedicated parameter-range review system.

Purpose:
- help the user detect extrapolation before prediction;
- show the observed training-domain minimum and maximum for each variable;
- make out-of-range conditions explicit.

Where it appears:
- beside the single prediction workflow;
- in the "Parameter Ranges" tab.

Possible statuses:
- inside
- outside
- invalid
- range unavailable
- model not loaded

Engineering meaning:
- "inside" means the value lies within the observed database domain;
- "outside" means the model is extrapolating beyond the calibration range and
  results should be treated with additional engineering judgement.

PERSISTENT DEFAULTS
===================
The GUI stores persistent state in defaults.json so that the previous session is
recovered automatically when the program is reopened.

Stored items include:
- all single-case input fields;
- selected model/database/output paths;
- batch input and output paths;
- selected notebook tab;
- window geometry.

This allows the GUI to reopen in nearly the same state as the previous session.

If defaults.json is missing:
- built-in default values are used.

If you want a clean reset:
- close the GUI;
- delete defaults.json;
- reopen the GUI.

RUNNING THE SCRIPT FROM SOURCE
==============================
Typical Windows workflow from the project directory:

1. Create a virtual environment
   py -m venv .venv

2. Activate the virtual environment
   .venv\\Scripts\\activate

3. Upgrade pip
   py -m pip install --upgrade pip

4. Install required packages
   pip install pyinstaller numpy pandas matplotlib scikit-learn joblib

5. Run the GUI from source
   python gui.py

If you prefer, you can also invoke the environment interpreter explicitly
without activating the virtual environment:
   .venv\\Scripts\\python gui.py

WHY A VIRTUAL ENVIRONMENT IS RECOMMENDED
========================================
A project-specific virtual environment isolates the dependencies of this GUI and
its backend from other Python projects installed on the same machine. This helps
avoid:
- version conflicts between packages;
- accidental use of the wrong interpreter;
- inconsistent build results when compiling the executable.

In practical terms, using .venv makes the runtime environment reproducible and
keeps the GUI, the backend, and the executable build process aligned.

PIP INSTALLATION AND MAINTENANCE
================================
For this GUI, pip is used for two distinct purposes:

1. Runtime dependency installation
   - numpy
   - pandas
   - matplotlib
   - scikit-learn
   - joblib

2. Build tooling installation
   - pyinstaller

Useful commands:
- upgrade pip
    py -m pip install --upgrade pip

- check pip version
    py -m pip --version

- install packages
    pip install pyinstaller numpy pandas matplotlib scikit-learn joblib

- optionally export an environment snapshot
    pip freeze > requirements.txt

If pip is not working correctly, one recovery path is:
    py -m ensurepip --upgrade
    py -m pip install --upgrade pip

EXECUTABLE COMPILATION FOR WINDOWS
==================================
This GUI can be compiled into a Windows standalone executable using PyInstaller.

Recommended build mode:
- one-file mode (--onefile)

Reason:
- this creates a single executable file named gui.exe;
- the executable can be distributed as one main launcher file;
- the project still keeps engineering files external and editable beside the
  executable;
- database.csv, model.joblib, diagnostics.json, defaults.json, input.txt, and
  output CSV files are not embedded and should remain outside gui.exe.

Recommended build command:
    pyinstaller --noconfirm --clean --windowed --onefile --name gui --paths . gui.py

Meaning of the main options:
- --noconfirm
  Overwrite previous build outputs without interactive confirmation.

- --clean
  Clear PyInstaller cache and temporary build files before building.

- --windowed
  Build the application without a console window, which is appropriate for a
  Tkinter GUI.

- --onefile
  Create a single executable file that bundles the Python interpreter and
  imported dependencies.

- --name gui
  Set the output executable name to gui.exe.

- --paths .
  Add the project directory to the import search path during analysis.

Expected output structure:
    dist/
        gui.exe
        database.csv
        model.joblib
        diagnostics.json
        input.txt
        batch_predictions.csv
        predictions.csv
        ...

DISTRIBUTION PRINCIPLE
======================
PyInstaller bundles:
- the active Python interpreter;
- imported Python modules;
- required runtime libraries.

Therefore, the target Windows machine does not need a separate Python
installation to run gui.exe.

However, this project is intentionally organised so that engineering data and
outputs remain external to the executable. This makes it easier to:
- swap models;
- replace the database;
- inspect diagnostics;
- edit input files;
- preserve user defaults across runs.

When using --onefile, PyInstaller still extracts its internal runtime support
files to a temporary folder at launch. This is normal. The external engineering
files listed above should remain beside gui.exe, not inside the temporary
runtime folder.

IMPORTANT BUILD RECOMMENDATIONS
===============================
1. Build on the same operating system you plan to run on.
   For a Windows executable, build on Windows.

2. Keep gui.py and train.py from the same project revision.
   The GUI depends directly on backend constants, feature names, diagnostics
   structure, parsers, and model-loading logic.

3. Keep external files beside gui.exe when distributing the application.
   Do not embed them if you want them to stay user-editable.

4. Use --onefile only with correct path handling in gui.py.
   The script should resolve its application directory from the executable
   location when frozen, so that external files are read and written beside
   gui.exe.

5. Test the unfrozen script before building.
   If python gui.py fails, the packaged executable will usually fail as well.

6. Rebuild the virtual environment if it is moved.
   Python virtual environments are effectively location-dependent and are best
   recreated rather than moved manually.

EXAMPLE FULL WINDOWS WORKFLOW
=============================
From the project directory:

    py -m venv .venv
    .venv\\Scripts\\activate
    py -m pip install --upgrade pip
    pip install pyinstaller numpy pandas matplotlib scikit-learn joblib
    python gui.py
    pyinstaller --noconfirm --clean --windowed --onefile --name gui --paths . gui.py

After compilation:
- copy or keep the required external files beside gui.exe;
- launch the executable from the dist folder.

GENERAL ENGINEERING USAGE NOTES
===============================
- Prefer input values inside the observed training-domain ranges whenever
  possible.
- Treat outside-range results as extrapolations.
- Retrain the model if the source database changes materially.
- Review diagnostics after retraining before relying on the updated model.
- Use the log tab for traceability of loading, training, prediction, saving, and
  error events.
- Keep defaults.json under user control if you want persistent session behaviour.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Tkinter is required to run gui.py") from exc


from pathlib import Path
import sys

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent

DEFAULTS_FILE = APP_DIR / "defaults.json"
DEFAULT_BATCH_INPUT = APP_DIR / "input.txt"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from train import (  # noqa: E402
    DEFAULT_DIAGNOSTICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_PATH,
    DATABASE_FEATURES,
    build_prediction_table,
    load_model_bundle,
    parse_batch_feature_file,
    parse_inp_file,
    save_model_bundle,
    train_model_bundle,
    write_diagnostics,
)


FIELD_SPECS: List[Dict[str, str]] = [
    {
        "key": "name",
        "label": "Case name",
        "unit": "-",
        "db_col": "Name",
        "desc": "Scenario identifier written to the output table.",
        "default": "026-004",
    },
    {
        "key": "m",
        "label": "m",
        "unit": "-",
        "db_col": "m",
        "desc": "Foreshore slope cotangent.",
        "default": "14.000",
    },
    {
        "key": "beta",
        "label": "beta",
        "unit": "deg",
        "db_col": "b",
        "desc": "Wave attack angle relative to the structure normal.",
        "default": "0.000",
    },
    {
        "key": "h",
        "label": "h",
        "unit": "m",
        "db_col": "h",
        "desc": "Water depth in front of the structure.",
        "default": "0.181",
    },
    {
        "key": "hm0_toe",
        "label": "Hm0 toe",
        "unit": "m",
        "db_col": "Hm0 toe",
        "desc": "Spectral significant wave height at the toe.",
        "default": "0.091",
    },
    {
        "key": "tm10_toe",
        "label": "Tm-1,0 toe",
        "unit": "s",
        "db_col": "Tm-1,0 toe",
        "desc": "Spectral mean wave period at the toe.",
        "default": "1.160",
    },
    {
        "key": "ht",
        "label": "ht",
        "unit": "m",
        "db_col": "ht",
        "desc": "Water depth at the toe.",
        "default": "0.181",
    },
    {
        "key": "bt",
        "label": "Bt",
        "unit": "m",
        "db_col": "Bt",
        "desc": "Toe width.",
        "default": "0.000",
    },
    {
        "key": "gf",
        "label": "gf",
        "unit": "-",
        "db_col": "gf",
        "desc": "Roughness / permeability reduction factor.",
        "default": "1.000",
    },
    {
        "key": "cotad",
        "label": "cotad",
        "unit": "-",
        "db_col": "cotad",
        "desc": "Lower slope cotangent.",
        "default": "0.350",
    },
    {
        "key": "cotau",
        "label": "cotau",
        "unit": "-",
        "db_col": "cotau",
        "desc": "Upper slope cotangent.",
        "default": "0.330",
    },
    {
        "key": "berm_width",
        "label": "B",
        "unit": "m",
        "db_col": "B",
        "desc": "Berm width.",
        "default": "0.792",
    },
    {
        "key": "hb",
        "label": "hb",
        "unit": "m",
        "db_col": "hb",
        "desc": "Berm submergence level.",
        "default": "-0.002",
    },
    {
        "key": "rc",
        "label": "Rc",
        "unit": "m",
        "db_col": "Rc",
        "desc": "Crest freeboard.",
        "default": "0.069",
    },
    {
        "key": "ac",
        "label": "Ac",
        "unit": "m",
        "db_col": "Ac",
        "desc": "Armour crest freeboard.",
        "default": "0.069",
    },
    {
        "key": "gc",
        "label": "Gc",
        "unit": "m",
        "db_col": "Gc",
        "desc": "Crest width.",
        "default": "0.000",
    },
]

FIELD_MAP: Dict[str, Dict[str, str]] = {item["key"]: item for item in FIELD_SPECS}
DEFAULTS: Dict[str, str] = {item["key"]: item["default"] for item in FIELD_SPECS}
NUMERIC_KEYS: List[str] = [item["key"] for item in FIELD_SPECS if item["key"] != "name"]

INPUT_GROUPS = [
    ("Case / Wave", ["name", "m", "beta", "h", "hm0_toe", "tm10_toe"]),
    ("Toe / Slope", ["ht", "bt", "gf", "cotad", "cotau"]),
    ("Berm / Crest", ["berm_width", "hb", "rc", "ac", "gc"]),
]

KEY_TO_DB = {item["key"]: item["db_col"] for item in FIELD_SPECS if item["key"] != "name"}


def safe_float(value: str) -> float:
    text = str(value).strip().replace(",", ".")
    if text == "":
        raise ValueError("Empty numeric input.")
    return float(text)


def fmt_value(value: object) -> str:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def _interp_zero_q_threshold(hm0_m: Optional[float]) -> Optional[float]:
    """Interpolated effective-zero overtopping screening threshold in l/s/m."""
    if hm0_m is None:
        return None
    try:
        hm0 = float(hm0_m)
    except Exception:
        return None
    if not math.isfinite(hm0) or hm0 <= 0.0:
        return None

    points = [
        (1.0, 0.03),
        (3.0, 0.16),
        (5.0, 0.40),
        (7.0, 0.60),
    ]
    if hm0 <= points[0][0]:
        return points[0][1]
    if hm0 >= points[-1][0]:
        return points[-1][1]

    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if x0 <= hm0 <= x1:
            t = (hm0 - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


def _average_q_band_message(q_lpsm: float, limits_and_messages: List[tuple[float, str]], fallback: str) -> str:
    for limit, message in limits_and_messages:
        if q_lpsm <= limit:
            return message
    return fallback


def build_average_q_consequence_assessment(q_lpsm: float, hm0_m: Optional[float]) -> str:
    q = max(0.0, float(q_lpsm))
    zero_q = _interp_zero_q_threshold(hm0_m)

    parts: List[str] = []
    if zero_q is None:
        parts.append(
            f"Average overtopping discharge q = {fmt_value(q)} l/s/m. "
            "The effective-zero threshold could not be interpolated because Hm0 toe is unavailable or invalid."
        )
    else:
        ratio = q / zero_q if zero_q > 0.0 else float('inf')
        if q <= zero_q:
            parts.append(
                f"Average overtopping discharge q = {fmt_value(q)} l/s/m with Hm0 toe = {fmt_value(hm0_m)} m. "
                f"The interpolated effective-zero threshold is {fmt_value(zero_q)} l/s/m, so the predicted mean overtopping is below that screening level."
            )
        else:
            parts.append(
                f"Average overtopping discharge q = {fmt_value(q)} l/s/m with Hm0 toe = {fmt_value(hm0_m)} m. "
                f"The interpolated effective-zero threshold is {fmt_value(zero_q)} l/s/m, so the predicted mean overtopping is about {fmt_value(ratio)} times higher and overtopping consequences should be treated as relevant."
            )

    parts.append(
        "This consequence assessment uses average overtopping discharge q only. "
        "It does not include peak individual overtopping volume Vmax, impulsiveness, local flow depth, or local flow velocity, so it is a screening-level interpretation and should be read conservatively."
    )

    pedestrian_msg = _average_q_band_message(
        q,
        [
            (max(zero_q or 0.0, 0.03), "below the highly safe / effective-zero screening level for pedestrian exposure."),
            (0.3, "below general public-pedestrian screening limits for non-violent overtopping."),
            (1.0, "above public-pedestrian screening; only trained prepared staff may be considered, and only for low-level non-impulsive overtopping."),
        ],
        "well above pedestrian screening limits; unrestricted pedestrian access should be considered unsafe.",
    )
    vehicle_msg = _average_q_band_message(
        q,
        [
            (1.0, "within low-risk screening for normal traffic, although wetness and splash may still reduce visibility."),
            (5.0, "above light-vehicle / motorcycle comfort screening; speed reduction and operational control are advisable."),
            (10.0, "high enough that exposed traffic should normally be restricted or stopped."),
        ],
        "high enough that vehicles may lose control or suffer direct water-impact damage on exposed routes.",
    )
    buildings_msg = _average_q_band_message(
        q,
        [
            (1.0, "no average-q screening exceedance for typical buildings, facades, and set-back equipment."),
            (5.0, "able to damage light facade elements, doors, windows, lighting columns, or exposed equipment."),
        ],
        "high enough that building elements and exposed equipment should be treated as at clear risk of damage.",
    )
    boats_msg = _average_q_band_message(
        q,
        [
            (1.0, "below the revised average-q trigger commonly used for small boats or yachts located some metres behind the wall."),
            (5.0, "within a caution range where small boats and light craft may be damaged and should be checked case by case."),
        ],
        "high enough that damage to small boats or light craft should be expected if they are exposed behind the wall.",
    )
    earth_msg = _average_q_band_message(
        q,
        [
            (0.1, "compatible with poor grass-cover screening on earth dikes or embankments."),
            (5.0, "within the usual maintained grass-cover tolerance range for earth dikes or rear slopes."),
            (10.0, "in the upper maintained-grass range; crest, grass quality, and local detailing should be checked carefully."),
        ],
        "above the usual maintained-grass screening range; erosion or damage of earth embankments and rear slopes becomes likely.",
    )
    rigid_msg = _average_q_band_message(
        q,
        [
            (1.0, "no average-q rear-side damage screening exceedance for rigid structures, rear slopes, and robust revetment details."),
            (10.0, "within a controlled overtopping range where local detailing, drainage, and rear-face protection need explicit checking."),
        ],
        "high enough that rear-face or back-slope structural damage should be checked explicitly for rigid structures and revetments.",
    )
    pavement_msg = _average_q_band_message(
        q,
        [
            (1.0, "no average-q pavement screening exceedance for promenades and paved areas."),
            (10.0, "high enough that drainage, jointing, edge restraint, spray nuisance, and slip hazard need explicit checking."),
        ],
        "high enough that promenade or pavement use becomes operationally problematic and local damage mechanisms may develop.",
    )

    parts.extend([
        f"Pedestrians: {pedestrian_msg}",
        f"Vehicles: {vehicle_msg}",
        f"Buildings / equipment: {buildings_msg}",
        f"Boats / yachts: {boats_msg}",
        f"Earth embankments / grassed rear slopes: {earth_msg}",
        f"Rigid structures / rear slopes / revetments: {rigid_msg}",
        f"Promenades / pavements: {pavement_msg}",
    ])
    return "\n\n".join(parts)


class OvertoppingGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Overtopping Neural Network Predictor")
        self.root.geometry("1550x1000")
        self.root.minsize(1280, 840)

        self.defaults_path = DEFAULTS_FILE
        self._defaults_after_id: Optional[str] = None
        self._persistence_ready = False
        self._persisted_defaults = self._read_defaults_payload()
        self.loaded_model_path: Optional[Path] = None

        self.model_path_var = tk.StringVar(value=str(APP_DIR / DEFAULT_MODEL_PATH))
        self.database_path_var = tk.StringVar(value=str(APP_DIR / "database.csv"))
        self.output_path_var = tk.StringVar(value=str(APP_DIR / DEFAULT_OUTPUT_PATH))
        self.diagnostics_path_var = tk.StringVar(value=str(APP_DIR / DEFAULT_DIAGNOSTICS_PATH))
        self.batch_inp_path_var = tk.StringVar(value=str(DEFAULT_BATCH_INPUT))
        self.batch_output_path_var = tk.StringVar(value=str(APP_DIR / "batch_predictions.csv"))

        self.status_var = tk.StringVar(value="Ready.")
        self.model_info_var = tk.StringVar(value="Model not loaded yet.")
        self.diagnostics_info_var = tk.StringVar(value="Diagnostics not loaded yet.")
        self.case_summary_var = tk.StringVar(value="No prediction run yet.")
        self.interpretation_var = tk.StringVar(
            value="Run a prediction to assess consequences from the average overtopping discharge."
        )

        self.input_vars: Dict[str, tk.StringVar] = {
            key: tk.StringVar(value=value) for key, value in DEFAULTS.items()
        }
        self.range_vars: Dict[str, tk.StringVar] = {
            key: tk.StringVar(value="Observed range: model not loaded") for key in NUMERIC_KEYS
        }
        self.output_vars: Dict[str, tk.StringVar] = {
            "mean_lpsm": tk.StringVar(value="-"),
            "p05_lpsm": tk.StringVar(value="-"),
            "p50_lpsm": tk.StringVar(value="-"),
            "p95_lpsm": tk.StringVar(value="-"),
        }
        self.batch_summary_vars: Dict[str, tk.StringVar] = {
            "count": tk.StringVar(value="-"),
            "mean_q": tk.StringVar(value="-"),
            "mean_p05": tk.StringVar(value="-"),
            "mean_p50": tk.StringVar(value="-"),
            "mean_p95": tk.StringVar(value="-"),
        }

        self.bundle = None
        self.last_single_result: Optional[pd.DataFrame] = None
        self.last_batch_result: Optional[pd.DataFrame] = None
        self.last_diagnostics_payload: Optional[Dict[str, Any]] = None

        self._apply_persisted_defaults_to_vars(self._persisted_defaults)

        self._configure_style()
        self._build_header()
        self._build_body()
        self._build_footer()
        self._register_persistence_hooks()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Configure>", self._on_root_configure, add="+")

        self._restore_window_geometry(self._persisted_defaults)
        self._restore_notebook_selection(self._persisted_defaults)
        self._append_log("GUI started.")
        if self._persisted_defaults:
            self._append_log(f"Previous GUI configuration restored from: {self.defaults_path}")
        else:
            self._append_log("No previous defaults.json found. Using built-in defaults.")
        self._refresh_range_views()

        self._persistence_ready = True
        self._schedule_defaults_save()
        self.root.after_idle(self._startup_initialize)

    # ------------------------------------------------------------------
    # Persistent defaults
    # ------------------------------------------------------------------
    def _read_defaults_payload(self) -> Dict[str, Any]:
        if not self.defaults_path.exists():
            return {}
        try:
            payload = json.loads(self.defaults_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {}

    def _apply_persisted_defaults_to_vars(self, payload: Dict[str, Any]) -> None:
        if not payload:
            return

        paths = payload.get("paths", {}) or {}
        inputs = payload.get("inputs", {}) or {}

        path_var_map = {
            "model_path": self.model_path_var,
            "database_path": self.database_path_var,
            "output_path": self.output_path_var,
            "diagnostics_path": self.diagnostics_path_var,
            "batch_input_path": self.batch_inp_path_var,
            "batch_output_path": self.batch_output_path_var,
        }
        for key, var in path_var_map.items():
            value = paths.get(key)
            if value not in (None, ""):
                var.set(str(value))

        for key, var in self.input_vars.items():
            value = inputs.get(key)
            if value is not None:
                var.set(str(value))

    def _collect_defaults_payload(self) -> Dict[str, Any]:
        selected_tab = 0
        try:
            selected_tab = int(self.notebook.index(self.notebook.select()))
        except Exception:
            selected_tab = 0

        return {
            "window_geometry": self.root.geometry(),
            "selected_tab": selected_tab,
            "paths": {
                "model_path": self.model_path_var.get().strip(),
                "database_path": self.database_path_var.get().strip(),
                "output_path": self.output_path_var.get().strip(),
                "diagnostics_path": self.diagnostics_path_var.get().strip(),
                "batch_input_path": self.batch_inp_path_var.get().strip(),
                "batch_output_path": self.batch_output_path_var.get().strip(),
            },
            "inputs": {
                key: var.get() for key, var in self.input_vars.items()
            },
        }

    def _save_defaults_now(self) -> None:
        self._defaults_after_id = None
        if not self._persistence_ready:
            return
        try:
            payload = self._collect_defaults_payload()
            self.defaults_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            self._append_log(f"Could not save defaults.json: {exc}")

    def _schedule_defaults_save(self, *_args: object) -> None:
        if not self._persistence_ready:
            return
        if self._defaults_after_id is not None:
            try:
                self.root.after_cancel(self._defaults_after_id)
            except Exception:
                pass
        self._defaults_after_id = self.root.after(400, self._save_defaults_now)

    def _register_persistence_hooks(self) -> None:
        watched_vars = [
            self.model_path_var,
            self.database_path_var,
            self.output_path_var,
            self.diagnostics_path_var,
            self.batch_inp_path_var,
            self.batch_output_path_var,
        ] + list(self.input_vars.values())

        for var in watched_vars:
            var.trace_add("write", self._schedule_defaults_save)

        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed, add="+")

    def _restore_window_geometry(self, payload: Dict[str, Any]) -> None:
        geometry = str((payload.get("window_geometry") or "")).strip()
        if geometry:
            try:
                self.root.geometry(geometry)
            except Exception:
                pass

    def _restore_notebook_selection(self, payload: Dict[str, Any]) -> None:
        try:
            index = int(payload.get("selected_tab", 0))
        except Exception:
            index = 0
        try:
            if 0 <= index < self.notebook.index("end"):
                self.notebook.select(index)
        except Exception:
            pass

    def _on_notebook_tab_changed(self, _event=None) -> None:
        self._schedule_defaults_save()

    def _on_close(self) -> None:
        try:
            self._save_defaults_now()
        finally:
            self.root.destroy()

    def _on_root_configure(self, _event=None) -> None:
        self._sync_single_panes()
        self._schedule_defaults_save()

    # ------------------------------------------------------------------
    # Startup / layout
    # ------------------------------------------------------------------
    def _startup_initialize(self) -> None:
        self._sync_single_panes()
        self._load_existing_model_on_startup()
        self._refresh_range_views()

    def _load_existing_model_on_startup(self) -> None:
        model_path = Path(self.model_path_var.get().strip())
        diagnostics_path = Path(self.diagnostics_path_var.get().strip())

        if model_path.exists():
            try:
                self.bundle = load_model_bundle(model_path)
                self.loaded_model_path = model_path.resolve()
                self._append_log(f"Loaded existing model on startup: {model_path}")
                diagnostics_payload = self._read_diagnostics_json(diagnostics_path)
                self._set_model_info_from_bundle(diagnostics_payload)
                self.status_var.set(f"Existing model loaded on startup: {model_path}")
            except Exception as exc:
                self.bundle = None
                self.loaded_model_path = None
                self.model_info_var.set(f"Startup model load failed: {exc}")
                self.status_var.set(f"Startup model load failed: {exc}")
                self._append_log(f"Startup model load failed: {exc}")
                self._append_log(traceback.format_exc())
        else:
            self.model_info_var.set(f"Model file not found at startup: {model_path}")
            diagnostics_payload = self._read_diagnostics_json(diagnostics_path)
            if diagnostics_payload is not None:
                self.last_diagnostics_payload = diagnostics_payload
            self.status_var.set(
                "Ready. No existing model file was found at startup; the GUI will load or train one when needed."
            )
            self._append_log(f"Model file not found at startup: {model_path}")

    def _sync_single_panes(self) -> None:
        if not hasattr(self, "single_panes"):
            return
        try:
            total_width = self.single_panes.winfo_width()
            if total_width <= 100:
                return
            target_x = max(480, min(total_width - 480, total_width // 2))
            current_x = self.single_panes.sash_coord(0)[0]
            if abs(current_x - target_x) > 2:
                self.single_panes.sash_place(0, target_x, 0)
        except Exception:
            pass

    def _update_dynamic_wraplengths(self) -> None:
        return

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        for theme_name in ("vista", "xpnative", "clam", "alt", "default"):
            if theme_name in style.theme_names():
                try:
                    style.theme_use(theme_name)
                    break
                except Exception:
                    pass

        self.root.configure(bg="#edf3f9")
        style.configure("TNotebook", background="#edf3f9", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 10), font=("Segoe UI", 11, "bold"))
        style.configure("Card.TLabelframe", padding=12)
        style.configure("Card.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("MetricTitle.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("HugeMetric.TLabel", font=("Segoe UI", 26, "bold"))
        style.configure("LargeMetric.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 9))
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), padding=(12, 8))
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg="#1f3b5b", padx=18, pady=14)
        header.pack(fill="x")

        tk.Label(
            header,
            text="Overtopping Neural Network Predictor",
            bg="#1f3b5b",
            fg="white",
            font=("Segoe UI", 22, "bold"),
            anchor="w",
        ).pack(fill="x")

        tk.Label(
            header,
            text=(
                "Single prediction, batch prediction, model training / refresh, observed-range review, automatic startup model load, and persistent GUI defaults."
            ),
            bg="#1f3b5b",
            fg="#d8e5f2",
            font=("Segoe UI", 10),
            anchor="w",
        ).pack(fill="x", pady=(4, 0))

    def _build_body(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)

        self.single_tab = ttk.Frame(self.notebook, padding=14)
        self.ranges_tab = ttk.Frame(self.notebook, padding=14)
        self.batch_tab = ttk.Frame(self.notebook, padding=14)
        self.log_tab = ttk.Frame(self.notebook, padding=14)
        self.help_tab = ttk.Frame(self.notebook, padding=14)

        self.notebook.add(self.single_tab, text="Single Prediction")
        self.notebook.add(self.ranges_tab, text="Parameter Ranges")
        self.notebook.add(self.batch_tab, text="Batch Prediction")
        self.notebook.add(self.help_tab, text="Readme")
        self.notebook.add(self.log_tab, text="Log")

        self._build_single_tab()
        self._build_ranges_tab()
        self._build_batch_tab()
        self._build_log_tab()
        self._build_help_tab()
        self.notebook.select(self.single_tab)

    def _build_single_tab(self) -> None:
        self.single_panes = tk.PanedWindow(
            self.single_tab,
            orient="horizontal",
            bd=0,
            bg="#edf3f9",
            sashwidth=6,
            sashrelief="flat",
            showhandle=False,
            opaqueresize=True,
        )
        self.single_panes.pack(fill="both", expand=True)

        self.single_left = ttk.Frame(self.single_panes)
        self.single_right = ttk.Frame(self.single_panes)
        self.single_panes.add(self.single_left, minsize=560, stretch="always")
        self.single_panes.add(self.single_right, minsize=560, stretch="always")

        left = self.single_left
        right = self.single_right
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        input_book = ttk.Notebook(left)
        input_book.grid(row=0, column=0, sticky="nsew")

        for group_name, keys in INPUT_GROUPS:
            tab = ttk.Frame(input_book, padding=12)
            tab.columnconfigure(0, weight=1)
            card = ttk.LabelFrame(tab, text=group_name, style="Card.TLabelframe")
            card.grid(row=0, column=0, sticky="nsew")
            card.columnconfigure(1, weight=1)
            row = 0
            for key in keys:
                row = self._add_input_row(card, row, key)
            input_book.add(tab, text=group_name)

        files_tab = ttk.Frame(input_book, padding=12)
        files_tab.columnconfigure(0, weight=1)
        files_card = ttk.LabelFrame(files_tab, text="Model / files / actions", style="Card.TLabelframe")
        files_card.grid(row=0, column=0, sticky="nsew")
        files_card.columnconfigure(1, weight=1)

        self._add_path_row(files_card, 0, "Model file", self.model_path_var, self.browse_model)
        self._add_path_row(files_card, 1, "Database CSV", self.database_path_var, self.browse_database)
        self._add_path_row(files_card, 2, "Single CSV output", self.output_path_var, self.browse_output_csv)
        self._add_path_row(files_card, 3, "Diagnostics JSON", self.diagnostics_path_var, self.browse_diagnostics)

        self.model_info_label_files = ttk.Label(files_card, textvariable=self.model_info_var, justify="left")
        self.model_info_label_files.grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 4))
        self.diagnostics_info_label_files = ttk.Label(files_card, textvariable=self.diagnostics_info_var, justify="left")
        self.diagnostics_info_label_files.grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 8))

        button_row = ttk.Frame(files_card)
        button_row.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        for idx in range(3):
            button_row.columnconfigure(idx, weight=1)
        ttk.Button(button_row, text="Predict q", style="Primary.TButton", command=self.predict_single).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(button_row, text="Save single result", command=self.save_single_result).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(button_row, text="Train / refresh model", command=self.train_or_refresh_model).grid(
            row=0, column=2, sticky="ew", padx=(6, 0)
        )

        button_row_2 = ttk.Frame(files_card)
        button_row_2.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        for idx in range(3):
            button_row_2.columnconfigure(idx, weight=1)
        ttk.Button(button_row_2, text="Reset defaults", command=self.reset_defaults).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(button_row_2, text="Open ranges tab", command=lambda: self.notebook.select(self.ranges_tab)).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(button_row_2, text="Open batch tab", command=lambda: self.notebook.select(self.batch_tab)).grid(
            row=0, column=2, sticky="ew", padx=(6, 0)
        )

        input_book.add(files_tab, text="Model / Files")

        self.summary_card = ttk.LabelFrame(right, text="Prediction summary", style="Card.TLabelframe")
        self.summary_card.grid(row=0, column=0, sticky="nsew")
        self.summary_card.columnconfigure(0, weight=1, uniform="prediction_summary")
        self.summary_card.columnconfigure(1, weight=1, uniform="prediction_summary")
        self.summary_card.rowconfigure(0, weight=1)

        self.numeric_column = ttk.Frame(self.summary_card, padding=(10, 10, 8, 10))
        self.numeric_column.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.numeric_column.columnconfigure(0, weight=1, uniform="numeric_metrics")
        self.numeric_column.columnconfigure(1, weight=1, uniform="numeric_metrics")
        self.numeric_column.rowconfigure(1, weight=1)
        self.numeric_column.rowconfigure(2, weight=1)

        self.consequence_column = ttk.Frame(self.summary_card, padding=(8, 10, 10, 10))
        self.consequence_column.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.consequence_column.columnconfigure(0, weight=1)
        self.consequence_column.rowconfigure(1, weight=1)

        hero = ttk.Frame(self.numeric_column, padding=(0, 0, 0, 8))
        hero.grid(row=0, column=0, columnspan=2, sticky="ew")
        hero.columnconfigure(0, weight=1)
        ttk.Label(hero, text="Average q [l/s/m]", style="MetricTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(hero, textvariable=self.output_vars["mean_lpsm"], style="HugeMetric.TLabel").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self.case_summary_label = ttk.Label(
            hero,
            textvariable=self.case_summary_var,
            justify="left",
            wraplength=250,
            padding=(0, 0, 8, 0),
        )
        self.case_summary_label.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self._add_metric_box(self.numeric_column, 1, 0, "P05 q [l/s/m]", self.output_vars["p05_lpsm"], large=False)
        self._add_metric_box(self.numeric_column, 1, 1, "P50 q [l/s/m]", self.output_vars["p50_lpsm"], large=False)
        self._add_metric_box(self.numeric_column, 2, 0, "P95 q [l/s/m]", self.output_vars["p95_lpsm"], large=False)

        ttk.Label(
            self.consequence_column,
            text="Consequence assessment",
            style="MetricTitle.TLabel",
            justify="left",
            wraplength=280,
        ).grid(row=0, column=0, sticky="ew")

        self.interpretation_box = tk.Text(
            self.consequence_column,
            wrap="word",
            height=14,
            font=("Segoe UI", 10),
            padx=0,
            pady=0,
            borderwidth=0,
            relief="flat",
            background="white",
            foreground="#202020",
            highlightthickness=0,
            spacing1=0,
            spacing2=4,
            spacing3=4,
        )
        self.interpretation_box.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.interpretation_box.insert("1.0", self.interpretation_var.get())
        self.interpretation_box.configure(state="disabled")

        diagnostics_card = ttk.LabelFrame(right, text="Model / diagnostics summary", style="Card.TLabelframe")
        diagnostics_card.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        diagnostics_card.columnconfigure(0, weight=1)
        self.model_info_label_right = ttk.Label(diagnostics_card, textvariable=self.model_info_var, justify="left")
        self.model_info_label_right.grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 4))
        self.diagnostics_info_label_right = ttk.Label(diagnostics_card, textvariable=self.diagnostics_info_var, justify="left")
        self.diagnostics_info_label_right.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))

        warning_card = ttk.LabelFrame(right, text="Warnings / extrapolation check", style="Card.TLabelframe")
        warning_card.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        self.warning_box = scrolledtext.ScrolledText(
            warning_card,
            wrap="word",
            font=("Consolas", 12),
            height=10,
            padx=10,
            pady=10,
            borderwidth=0,
            relief="flat",
            background="white",
            foreground="#7a1010",
        )
        self.warning_box.pack(fill="both", expand=True)
        self.warning_box.insert("1.0", "No prediction run yet.")
        self.warning_box.configure(state="disabled")

        review_card = ttk.LabelFrame(right, text="Current single-case input review", style="Card.TLabelframe")
        review_card.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        review_card.columnconfigure(0, weight=1)
        review_card.rowconfigure(0, weight=1)

        review_columns = ("parameter", "value", "unit", "observed_range", "status")
        self.review_tree = ttk.Treeview(review_card, columns=review_columns, show="headings", height=12)
        review_widths = {"parameter": 180, "value": 120, "unit": 80, "observed_range": 260, "status": 120}
        for col in review_columns:
            self.review_tree.heading(col, text=col.replace("_", " ").title())
            self.review_tree.column(col, width=review_widths[col], anchor="w")
        review_y = ttk.Scrollbar(review_card, orient="vertical", command=self.review_tree.yview)
        review_x = ttk.Scrollbar(review_card, orient="horizontal", command=self.review_tree.xview)
        self.review_tree.configure(yscrollcommand=review_y.set, xscrollcommand=review_x.set)
        self.review_tree.grid(row=0, column=0, sticky="nsew")
        review_y.grid(row=0, column=1, sticky="ns")
        review_x.grid(row=1, column=0, sticky="ew")

        self._dynamic_wrap_widgets = [
            self.case_summary_label,
            self.model_info_label_files,
            self.diagnostics_info_label_files,
            self.model_info_label_right,
            self.diagnostics_info_label_right,
        ]

    def _build_ranges_tab(self) -> None:
        self.ranges_tab.columnconfigure(0, weight=1)
        self.ranges_tab.rowconfigure(1, weight=1)

        top_card = ttk.LabelFrame(
            self.ranges_tab,
            text="Observed database ranges for all prediction parameters",
            style="Card.TLabelframe",
        )
        top_card.grid(row=0, column=0, sticky="ew")
        top_card.columnconfigure(0, weight=1)

        ttk.Label(
            top_card,
            text=(
                "These min/max values come from the trained database domain. "
                "Outside-range values trigger extrapolation warnings."
            ),
            wraplength=1200,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(top_card, text="Refresh range check", command=self._refresh_range_views).grid(
            row=0, column=1, sticky="e", padx=(12, 0)
        )

        table_card = ttk.LabelFrame(self.ranges_tab, text="Range table", style="Card.TLabelframe")
        table_card.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        table_card.columnconfigure(0, weight=1)
        table_card.rowconfigure(0, weight=1)

        columns = ("parameter", "current_value", "unit", "minimum", "maximum", "status", "description")
        self.range_tree = ttk.Treeview(table_card, columns=columns, show="headings", height=18)
        widths = {
            "parameter": 180,
            "current_value": 120,
            "unit": 70,
            "minimum": 110,
            "maximum": 110,
            "status": 120,
            "description": 560,
        }
        for col in columns:
            self.range_tree.heading(col, text=col.replace("_", " ").title())
            self.range_tree.column(col, width=widths[col], anchor="w")
        yscroll = ttk.Scrollbar(table_card, orient="vertical", command=self.range_tree.yview)
        xscroll = ttk.Scrollbar(table_card, orient="horizontal", command=self.range_tree.xview)
        self.range_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.range_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

    def _build_batch_tab(self) -> None:
        self.batch_tab.columnconfigure(0, weight=1)
        self.batch_tab.rowconfigure(2, weight=1)

        files_card = ttk.LabelFrame(self.batch_tab, text="Batch input / output", style="Card.TLabelframe")
        files_card.grid(row=0, column=0, sticky="ew")
        files_card.columnconfigure(1, weight=1)
        self._add_path_row(files_card, 0, "Batch file", self.batch_inp_path_var, self.browse_inp)
        self._add_path_row(files_card, 1, "Batch CSV output", self.batch_output_path_var, self.browse_batch_output)

        ttk.Label(
            files_card,
            text=(
                "Default startup batch file is input.txt. The file must contain the same physical input columns used by train.py."
            ),
            wraplength=1100,
            justify="left",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        actions = ttk.Frame(files_card)
        actions.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        for idx in range(3):
            actions.columnconfigure(idx, weight=1)
        ttk.Button(actions, text="Predict from batch file", style="Primary.TButton", command=self.predict_batch).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(actions, text="Save batch result", command=self.save_batch_result).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(actions, text="Use default input.txt", command=self.load_example_inp_path).grid(
            row=0, column=2, sticky="ew", padx=(6, 0)
        )

        table_card = ttk.LabelFrame(self.batch_tab, text="Batch results", style="Card.TLabelframe")
        table_card.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        table_card.columnconfigure(0, weight=1)
        table_card.rowconfigure(0, weight=1)

        columns = (
            "Name",
            "q_l_per_s_per_m",
            "q_p05_l_per_s_per_m",
            "q_p50_l_per_s_per_m",
            "q_p95_l_per_s_per_m",
            "range_warning",
        )
        self.batch_tree = ttk.Treeview(table_card, columns=columns, show="headings", height=18)
        widths = {
            "Name": 170,
            "q_l_per_s_per_m": 130,
            "q_p05_l_per_s_per_m": 130,
            "q_p50_l_per_s_per_m": 130,
            "q_p95_l_per_s_per_m": 130,
            "range_warning": 650,
        }
        for col in columns:
            self.batch_tree.heading(col, text=col)
            self.batch_tree.column(col, width=widths[col], anchor="w")
        yscroll = ttk.Scrollbar(table_card, orient="vertical", command=self.batch_tree.yview)
        xscroll = ttk.Scrollbar(table_card, orient="horizontal", command=self.batch_tree.xview)
        self.batch_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.batch_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

    def _build_log_tab(self) -> None:
        log_frame = ttk.LabelFrame(self.log_tab, text="Execution log", style="Card.TLabelframe")
        log_frame.pack(fill="both", expand=True)
        self.log_box = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            font=("Consolas", 11),
            padx=12,
            pady=12,
            borderwidth=0,
            relief="flat",
            background="white",
        )
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

    def _build_help_tab(self) -> None:
        text = (
            "OVERVIEW\n"
            "========\n"
            "This GUI is the front-end for the overtopping predictor defined by train.py. It is intended for four main workflows: \n"
            "(1) evaluating one overtopping scenario interactively, \n"
            "(2) evaluating many scenarios from a batch file, \n"
            "(3) loading / refreshing the trained model and diagnostics, and \n"
            "(4) checking whether the entered parameters are inside or outside the calibration domain represented by the database.\n\n"
            "AUTOMATIC STARTUP BEHAVIOUR\n"
            "==========================\n"
            "- When the GUI starts, it immediately tries to load the model file currently shown in the 'Model file' field.\n"
            "- If that model exists and is readable, the GUI loads it before you press any prediction button.\n"
            "- If the model is missing, the GUI remains ready and will load or train a model later when required.\n"
            "- The full GUI state is persisted in defaults.json, including file paths, input values, current tab, and window geometry.\n"
            "- On the next launch, the GUI restores those values automatically.\n\n"
            "SINGLE PREDICTION TAB\n"
            "=====================\n"
            "The left half of the window contains the case definition and file actions. The right half contains the prediction summary, diagnostics summary, warning panel, and a detailed review table of the current single case.\n\n"
            "Input groups:\n"
            "- Case / Wave: case name, Hm0 toe, Tm-1,0 toe, m, h, beta.\n"
            "- Toe / Slope: ht, Bt, gf, cotad, cotau.\n"
            "- Berm / Crest: B, hb, Rc, Ac, Gc.\n\n"
            "How to run one scenario:\n"
            "1. Fill or edit the physical parameters.\n"
            "2. Confirm the model/database/output paths if needed.\n"
            "3. Press 'Predict q'.\n"
            "4. Read the two-column 'Prediction summary' panel: numeric overtopping results on the left and the average-q consequence assessment on the right.\n"
            "5. Inspect the warnings panel and the review table for extrapolation checks.\n"
            "6. Press 'Save single result' if you want a CSV output file.\n\n"
            "Interpretation of the main single-case outputs:\n"
            "- Average q [l/s/m]: mean ensemble prediction in litres per second per metre.\n"
            "- P05 / P50 / P95 [l/s/m]: ensemble spread indicators to show low, central, and high estimates.\n"
            "- Warnings / extrapolation check: indicates whether any current input exceeds the trained database range.\n"
            "- Current single-case input review: lists each parameter with current value, observed min/max, and inside/outside status.\n\n"
            "MODEL / FILES PANEL\n"
            "===================\n"
            "- Model file: joblib bundle produced by train.py.\n"
            "- Database CSV: source data used to train or refresh the model.\n"
            "- Single CSV output: destination for the current single-case prediction table.\n"
            "- Diagnostics JSON: summary file written by the backend after training.\n\n"
            "Buttons:\n"
            "- Predict q: runs one scenario using the current model.\n"
            "- Train / refresh model: trains from the database, saves the model, and regenerates diagnostics.\n"
            "- Save single result: exports the most recent single prediction to CSV.\n"
            "- Reset defaults: restores the built-in default physical inputs.\n"
            "- Open ranges tab / Open batch tab: quick navigation shortcuts.\n\n"
            "PARAMETER RANGES TAB\n"
            "====================\n"
            "This tab compares the currently entered numeric inputs against the trained database domain stored in the loaded model bundle.\n"
            "- 'minimum' and 'maximum' are the observed limits from the database used for training.\n"
            "- 'status' reports whether the current entry is inside, outside, invalid, or unavailable.\n"
            "- This tab is useful before prediction if you want to reduce extrapolation risk.\n\n"
            "BATCH PREDICTION TAB\n"
            "====================\n"
            "The batch tab is intended for many scenarios at once.\n"
            "- The default startup file is input.txt.\n"
            "- Accepted formats: pipe-separated text, tab-separated text/TSV, CSV, semicolon-separated text, and files that parse through the backend helpers.\n"
            "- The batch file must contain the same physical variables expected by train.py. Column aliases are normalised when possible.\n"
            "- Press 'Predict from batch file' to run all rows.\n"
            "- The result preview table shows each scenario name, mean q, P05, P50, P95, and any range warning.\n"
            "- Press 'Save batch result' to export the latest batch output as CSV.\n"
            "- Press 'Use default input.txt' to restore the standard initial batch path.\n\n"
            "TRAINING AND DIAGNOSTICS\n"
            "========================\n"
            "The updated backend distinguishes two stages:\n"
            "1. Holdout validation stage: used to evaluate model quality on withheld data.\n"
            "2. Full-data refit stage: used to train the final saved model on the full valid dataset.\n\n"
            "Current backend target setup:\n"
            "- physical dimensionless target: sq = q / sqrt(g * Hm0_toe^3)\n"
            "- internal training target: y = (log10(sq) - mu) / sigma\n\n"
            "Important counters and terms:\n"
            "- valid rows: rows in the database that remain after all backend validity filters.\n"
            "- n_holdout: number of rows used only for holdout validation diagnostics.\n"
            "- n_full: number of rows used in the final refit that produces the saved model.\n"
            "- R²(log10 sq): fit quality in transformed space.\n"
            "- R²(q l/s/m): fit quality in physical overtopping discharge units.\n\n"
            "PERSISTENT GUI DEFAULTS\n"
            "=======================\n"
            "The GUI writes defaults.json in the script folder. This file is used to restore your previous session. Stored items include:\n"
            "- all single-case input values;\n"
            "- model/database/output/diagnostics paths;\n"
            "- batch input and output paths;\n"
            "- selected notebook tab;\n"
            "- last window geometry.\n\n"
            "If you want to reset everything manually, close the GUI and delete defaults.json. The next launch will return to the built-in defaults.\n\n"
            "GENERAL USAGE NOTES\n"
            "===================\n"
            "- Prefer predictions inside the observed parameter ranges whenever possible.\n"
            "- Treat outside-range results as extrapolations requiring engineering judgement.\n"
            "- After retraining, review both the model summary and diagnostics summary.\n"
            "- Keep train.py and gui.py from the same revision so the GUI matches the backend feature definitions and diagnostics schema.\n"
            "- Use the Log tab whenever you need traceability of what was loaded, predicted, trained, or saved.\n"
        )

        info_frame = ttk.LabelFrame(self.help_tab, text="Instructions", style="Card.TLabelframe")
        info_frame.pack(fill="both", expand=True)
        self.help_box = scrolledtext.ScrolledText(
            info_frame,
            wrap="word",
            font=("Consolas", 12),
            padx=12,
            pady=12,
            borderwidth=0,
            relief="flat",
            background="white",
        )
        self.help_box.pack(fill="both", expand=True)
        self.help_box.insert("1.0", text)
        self.help_box.configure(state="disabled")

    def _build_footer(self) -> None:
        footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status_var, wraplength=1500, justify="left").pack(fill="x")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _add_input_row(self, parent, row: int, key: str) -> int:
        spec = FIELD_MAP[key]
        ttk.Label(parent, text=spec["label"], style="Section.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 10), pady=(8, 2)
        )
        entry = ttk.Entry(parent, textvariable=self.input_vars[key], width=20, font=("Segoe UI", 11))
        entry.grid(row=row, column=1, sticky="ew", pady=(8, 2))
        entry.bind("<FocusIn>", lambda _e, t=spec["desc"]: self.status_var.set(t))
        entry.bind("<KeyRelease>", lambda _e: self._refresh_range_views())
        ttk.Label(parent, text=spec["unit"]).grid(row=row, column=2, sticky="w", padx=(8, 10), pady=(8, 2))

        if key == "name":
            help_text = "Not a numeric field. Used as the scenario identifier in outputs."
            ttk.Label(parent, text=help_text, style="Small.TLabel", wraplength=760, justify="left").grid(
                row=row + 1, column=0, columnspan=3, sticky="w", padx=(0, 6), pady=(0, 8)
            )
        else:
            ttk.Label(parent, textvariable=self.range_vars[key], style="Small.TLabel", wraplength=760, justify="left").grid(
                row=row + 1, column=0, columnspan=3, sticky="w", padx=(0, 6), pady=(0, 2)
            )
            ttk.Label(parent, text=spec["desc"], style="Small.TLabel", wraplength=760, justify="left").grid(
                row=row + 2, column=0, columnspan=3, sticky="w", padx=(0, 6), pady=(0, 8)
            )
            row += 1
        return row + 2

    def _add_path_row(self, parent, row: int, label: str, var: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=6)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="ew", padx=(8, 0), pady=6)

    def _add_metric_box(self, parent, row: int, column: int, title: str, value_var: tk.StringVar, large: bool) -> None:
        frame = ttk.Frame(parent, padding=(8, 8))
        frame.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text=title, style="MetricTitle.TLabel", justify="left", wraplength=180).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Label(
            frame,
            textvariable=value_var,
            style=("LargeMetric.TLabel" if large else "Section.TLabel"),
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    def _set_interpretation_text(self, text: str) -> None:
        self.interpretation_var.set(text)
        interpretation_box = getattr(self, "interpretation_box", None)
        if interpretation_box is None:
            return
        interpretation_box.configure(state="normal")
        interpretation_box.delete("1.0", "end")
        interpretation_box.insert("1.0", text)
        interpretation_box.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message.rstrip() + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        self.root.update_idletasks()

    def _set_warning_text(self, message: str) -> None:
        self.warning_box.configure(state="normal")
        self.warning_box.delete("1.0", "end")
        self.warning_box.insert("1.0", message)
        self.warning_box.configure(state="disabled")

    def _get_numeric_inputs(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for key in NUMERIC_KEYS:
            values[key] = safe_float(self.input_vars[key].get())
        return values

    def _build_single_scenario_frame(self) -> pd.DataFrame:
        values = self._get_numeric_inputs()
        record = {"Name": self.input_vars["name"].get().strip() or "scenario_1"}
        for key, db_col in KEY_TO_DB.items():
            record[db_col] = values[key]
        return pd.DataFrame([record])

    def _read_diagnostics_json(self, diagnostics_path: Path) -> Optional[Dict[str, Any]]:
        if not diagnostics_path.exists():
            return None
        try:
            return json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._append_log(f"Could not read diagnostics JSON '{diagnostics_path}': {exc}")
            return None

    def _set_model_info_from_bundle(self, diagnostics_payload: Optional[Dict[str, Any]] = None) -> None:
        if self.bundle is None:
            self.model_info_var.set("Model not loaded yet.")
            self.diagnostics_info_var.set("Diagnostics not loaded yet.")
            return

        metrics = dict(getattr(self.bundle, "metrics", {}) or {})
        valid_rows = metrics.get("database_rows_with_valid_q_and_hm0", "n/a")
        holdout_rows = metrics.get("holdout_rows", metrics.get("test_rows", "n/a"))
        full_rows = metrics.get("full_rows", "n/a")
        r2_log10_sq = metrics.get("r2_log10_sq", float("nan"))
        r2_q_lpsm = metrics.get("r2_q_lpsm", float("nan"))
        n_models = metrics.get("n_models", "n/a")
        hidden_layers = metrics.get("hidden_layers", [])
        metadata = dict(getattr(self.bundle, "metadata", {}) or {})
        target_definition = str(metadata.get("target_definition", "sq = q / sqrt(g * Hm0_toe^3)")).strip()
        training_target_transform = str(
            metadata.get("training_target_transform", "y = (log10(sq) - mu) / sigma")
        ).strip()
        target_mean_log10_sq = metrics.get(
            "target_mean_log10_sq",
            getattr(self.bundle, "target_mean_log10_sq", float("nan")),
        )
        target_std_log10_sq = metrics.get(
            "target_std_log10_sq",
            getattr(self.bundle, "target_std_log10_sq", float("nan")),
        )
        uses_correct_target = "sqrt" in target_definition and "Hm0_toe^3" in target_definition
        uses_expected_transform = "log10(sq)" in training_target_transform and "sigma" in training_target_transform

        self.model_info_var.set(
            "Model ready. "
            f"valid rows={valid_rows}, "
            f"n_holdout={holdout_rows}, "
            f"n_full={full_rows}, "
            f"R²(log10 sq)={fmt_value(r2_log10_sq)}, "
            f"R²(q l/s/m)={fmt_value(r2_q_lpsm)}, "
            f"hidden_layers={hidden_layers}, "
            f"n_models={n_models}, "
            f"target={target_definition}, "
            f"train_target={training_target_transform}, "
            f"mu={fmt_value(target_mean_log10_sq)}, "
            f"sigma={fmt_value(target_std_log10_sq)}."
        )

        plot_paths = {}
        if diagnostics_payload is not None:
            plot_paths = diagnostics_payload.get("plot_paths", {}) or {}
            self.last_diagnostics_payload = diagnostics_payload

        if plot_paths:
            holdout_plot_count = sum(1 for key in plot_paths if str(key).startswith("holdout_"))
            full_plot_count = sum(1 for key in plot_paths if str(key).startswith("full_"))
            self.diagnostics_info_var.set(
                "Diagnostics loaded. "
                f"holdout plots={holdout_plot_count}, "
                f"full plots={full_plot_count}, "
                f"JSON={self.diagnostics_path_var.get().strip()}"
            )
        else:
            self.diagnostics_info_var.set(
                "Diagnostics summary unavailable or not loaded yet. "
                "Train / refresh the model to regenerate holdout and full-data plots."
            )

        if not uses_correct_target:
            legacy_message = (
                "Loaded model metadata indicates a legacy sq definition. "
                "Retrain model.joblib with the corrected train.py so the GUI and backend both use sq = q / sqrt(g * Hm0_toe^3)."
            )
            self.status_var.set(legacy_message)
            self.diagnostics_info_var.set(f"{self.diagnostics_info_var.get()} WARNING: legacy target detected.")
        elif not uses_expected_transform:
            legacy_message = (
                "Loaded model metadata indicates a legacy training-target transform. "
                "Retrain model.joblib with the corrected train.py so the backend uses y = (log10(sq) - mu) / sigma internally."
            )
            self.status_var.set(legacy_message)
            self.diagnostics_info_var.set(f"{self.diagnostics_info_var.get()} WARNING: legacy target transform detected.")

    def _normalize_batch_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        rename_map: Dict[str, str] = {}
        for column in frame.columns:
            raw = str(column).strip()
            norm = normalize_name(column)

            if raw == "Name" or norm in {"name", "scenario", "scenarioid", "testid", "id"}:
                rename_map[column] = "Name"
            elif raw == "m" or norm == "m":
                rename_map[column] = "m"
            elif raw in {"b", "beta", "Beta"} or norm in {"beta"}:
                rename_map[column] = "b"
            elif raw == "B" or norm in {"bermwidth", "bermb", "bermwidthb"}:
                rename_map[column] = "B"
            elif raw == "h" or norm == "h":
                rename_map[column] = "h"
            elif raw in {"Hm0 toe", "Hm0toe", "Hm0_t", "Hm0,t"} or norm in {"hm0toe", "hm0t", "hm0_toe"}:
                rename_map[column] = "Hm0 toe"
            elif raw in {"Tm-1,0 toe", "Tm-1,0toe", "Tm10t", "Tm-1,0,t"} or norm in {"tm10toe", "tm10t", "tm10_toe"}:
                rename_map[column] = "Tm-1,0 toe"
            elif raw == "ht" or norm == "ht":
                rename_map[column] = "ht"
            elif raw == "Bt" or norm == "bt":
                rename_map[column] = "Bt"
            elif raw == "gf" or norm == "gf":
                rename_map[column] = "gf"
            elif raw == "cotad" or norm == "cotad":
                rename_map[column] = "cotad"
            elif raw == "cotau" or norm == "cotau":
                rename_map[column] = "cotau"
            elif raw == "hb" or norm == "hb":
                rename_map[column] = "hb"
            elif raw == "Rc" or norm == "rc":
                rename_map[column] = "Rc"
            elif raw == "Ac" or norm == "ac":
                rename_map[column] = "Ac"
            elif raw == "Gc" or norm == "gc":
                rename_map[column] = "Gc"

        frame = frame.rename(columns=rename_map).copy()
        if "Name" not in frame.columns:
            frame["Name"] = [f"scenario_{idx + 1}" for idx in range(len(frame))]

        for col in DATABASE_FEATURES:
            if col not in frame.columns:
                frame[col] = float("nan")
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        return frame[["Name"] + DATABASE_FEATURES]

    def _load_batch_scenarios(self, path: Path) -> pd.DataFrame:
        sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
        suffix = path.suffix.lower()

        if "|" in sample:
            return parse_inp_file(path)

        if "\t" in sample and "|" not in sample:
            frame = pd.read_csv(path, sep="\t")
            return self._normalize_batch_frame(frame)

        if suffix in {".csv", ".txt", ".dat", ".tsv"}:
            try:
                return parse_batch_feature_file(path)
            except Exception:
                frame = pd.read_csv(path, sep=None, engine="python")
                return self._normalize_batch_frame(frame)

        frame = pd.read_csv(path, sep=None, engine="python")
        return self._normalize_batch_frame(frame)

    def _train_backend(self, database_path: Path, model_path: Path, diagnostics_path: Path):
        self._append_log(f"Training model from database: {database_path}")
        artifacts = train_model_bundle(database_path=database_path)
        self.bundle = artifacts.bundle
        self.loaded_model_path = model_path.resolve()
        save_model_bundle(self.bundle, model_path)
        write_diagnostics(
            self.bundle,
            diagnostics_path,
            holdout=artifacts.holdout,
            full_data=artifacts.full_data,
        )
        diagnostics_payload = self._read_diagnostics_json(diagnostics_path)
        self._set_model_info_from_bundle(diagnostics_payload)
        self._append_log(f"Model saved: {model_path}")
        self._append_log(f"Diagnostics saved: {diagnostics_path}")
        self._append_log(json.dumps(self.bundle.metrics, indent=2))
        self._schedule_defaults_save()
        return artifacts

    def _ensure_model(self):
        model_path = Path(self.model_path_var.get().strip())
        database_path = Path(self.database_path_var.get().strip())
        diagnostics_path = Path(self.diagnostics_path_var.get().strip())

        resolved_model_path = None
        try:
            resolved_model_path = model_path.resolve()
        except Exception:
            resolved_model_path = model_path

        if self.bundle is not None and self.loaded_model_path == resolved_model_path:
            self._refresh_range_views()
            return self.bundle

        if model_path.exists():
            self.bundle = load_model_bundle(model_path)
            self.loaded_model_path = resolved_model_path
            self._append_log(f"Loaded model: {model_path}")
            diagnostics_payload = self._read_diagnostics_json(diagnostics_path)
            self._set_model_info_from_bundle(diagnostics_payload)
        else:
            if not database_path.exists():
                raise FileNotFoundError(
                    f"Model file '{model_path}' does not exist and database file '{database_path}' was not found."
                )
            self._append_log(f"Model not found. Training new model from: {database_path}")
            self._train_backend(database_path, model_path, diagnostics_path)

        self._refresh_range_views()
        return self.bundle

    def _status_from_value_and_range(self, key: str, value_text: str) -> str:
        if self.bundle is None:
            return "model not loaded"
        try:
            value = safe_float(value_text)
        except Exception:
            return "invalid"
        db_col = KEY_TO_DB[key]
        min_v, max_v = self.bundle.feature_ranges.get(db_col, (float("nan"), float("nan")))
        if pd.isna(min_v) or pd.isna(max_v):
            return "range unavailable"
        if value < min_v or value > max_v:
            return "outside"
        return "inside"

    def _refresh_range_views(self) -> None:
        values = {key: self.input_vars[key].get().strip() for key in NUMERIC_KEYS}

        if hasattr(self, "review_tree"):
            for item in self.review_tree.get_children():
                self.review_tree.delete(item)
        if hasattr(self, "range_tree"):
            for item in self.range_tree.get_children():
                self.range_tree.delete(item)

        for key in NUMERIC_KEYS:
            spec = FIELD_MAP[key]
            value_text = values[key]
            status = self._status_from_value_and_range(key, value_text)

            if self.bundle is not None:
                db_col = KEY_TO_DB[key]
                min_v, max_v = self.bundle.feature_ranges.get(db_col, (float("nan"), float("nan")))
                observed_text = f"Observed range: [{fmt_value(min_v)}, {fmt_value(max_v)}]"
                min_text = fmt_value(min_v)
                max_text = fmt_value(max_v)
            else:
                observed_text = "Observed range: model not loaded"
                min_text = "-"
                max_text = "-"

            self.range_vars[key].set(observed_text + f" | Current status: {status}")

            review_row = (
                spec["label"],
                value_text or "-",
                spec["unit"],
                observed_text.replace("Observed range: ", ""),
                status,
            )
            if hasattr(self, "review_tree"):
                self.review_tree.insert("", "end", values=review_row)

            range_row = (
                spec["label"],
                value_text or "-",
                spec["unit"],
                min_text,
                max_text,
                status,
                spec["desc"],
            )
            if hasattr(self, "range_tree"):
                self.range_tree.insert("", "end", values=range_row)

    def _update_output_panel(self, row: pd.Series) -> None:
        mean_lpsm = float(row["q_l_per_s_per_m"])
        p05_lpsm = float(row["q_p05_l_per_s_per_m"])
        p50_lpsm = float(row["q_p50_l_per_s_per_m"])
        p95_lpsm = float(row["q_p95_l_per_s_per_m"])

        hm0_m: Optional[float]
        try:
            hm0_m = float(row.get("Hm0 toe", self.input_vars["hm0_toe"].get()))
        except Exception:
            try:
                hm0_m = safe_float(self.input_vars["hm0_toe"].get())
            except Exception:
                hm0_m = None

        self.output_vars["mean_lpsm"].set(fmt_value(mean_lpsm))
        self.output_vars["p05_lpsm"].set(fmt_value(p05_lpsm))
        self.output_vars["p50_lpsm"].set(fmt_value(p50_lpsm))
        self.output_vars["p95_lpsm"].set(fmt_value(p95_lpsm))

        case_name = str(row.get("Name", "scenario_1"))
        self.case_summary_var.set(f"Case: {case_name}")
        self._set_interpretation_text(build_average_q_consequence_assessment(mean_lpsm, hm0_m))

        warnings_text = str(row.get("range_warning", "") or "").strip()
        if warnings_text:
            message = (
                "Range warning detected.\n\n"
                f"{warnings_text}\n\n"
                "One or more inputs are outside the database range used for model training. "
                "Treat this result as an extrapolation."
            )
        else:
            message = (
                "No range warnings.\n\n"
                "All current inputs are inside the observed database range used for model training."
            )
        self._set_warning_text(message)
        self._refresh_range_views()

    def _update_batch_summary(self, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            for var in self.batch_summary_vars.values():
                var.set("-")
            return
        self.batch_summary_vars["count"].set(str(len(frame)))
        self.batch_summary_vars["mean_q"].set(fmt_value(frame["q_l_per_s_per_m"].mean()))
        self.batch_summary_vars["mean_p05"].set(fmt_value(frame["q_p05_l_per_s_per_m"].mean()))
        self.batch_summary_vars["mean_p50"].set(fmt_value(frame["q_p50_l_per_s_per_m"].mean()))
        self.batch_summary_vars["mean_p95"].set(fmt_value(frame["q_p95_l_per_s_per_m"].mean()))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def reset_defaults(self) -> None:
        for key, value in DEFAULTS.items():
            self.input_vars[key].set(value)
        self.case_summary_var.set("No prediction run yet.")
        self._set_interpretation_text(
            "Run a prediction to assess consequences from the average overtopping discharge."
        )
        for key in self.output_vars:
            self.output_vars[key].set("-")
        self._set_warning_text("No prediction run yet.")
        self._refresh_range_views()
        self.status_var.set("Built-in default inputs restored.")
        self._append_log("Default single-case inputs restored.")
        self._schedule_defaults_save()

    def train_or_refresh_model(self) -> None:
        try:
            model_path = Path(self.model_path_var.get().strip())
            database_path = Path(self.database_path_var.get().strip())
            diagnostics_path = Path(self.diagnostics_path_var.get().strip())
            if not database_path.exists():
                raise FileNotFoundError(f"Database file not found: {database_path}")
            self._train_backend(database_path, model_path, diagnostics_path)
            self._refresh_range_views()
            self.status_var.set(f"Model trained and saved to: {model_path}")
        except Exception as exc:
            self._handle_exception("Training failed", exc)

    def predict_single(self) -> None:
        try:
            self.notebook.select(self.single_tab)
            bundle = self._ensure_model()
            scenario = self._build_single_scenario_frame()
            result = build_prediction_table(bundle, scenario)
            self.last_single_result = result
            row = result.iloc[0]
            self._update_output_panel(row)
            self._append_log("Single prediction completed.")
            self._append_log(result.to_string(index=False))
            self.status_var.set(f"Prediction complete. Average q = {float(row['q_l_per_s_per_m']):.6g} l/s/m")
        except Exception as exc:
            self._handle_exception("Prediction failed", exc)

    def predict_batch(self) -> None:
        try:
            batch_path = Path(self.batch_inp_path_var.get().strip())
            if not batch_path.exists():
                raise FileNotFoundError(f"Batch file not found: {batch_path}")
            bundle = self._ensure_model()
            scenarios = self._load_batch_scenarios(batch_path)
            result = build_prediction_table(bundle, scenarios)
            self.last_batch_result = result
            self._populate_batch_tree(result)
            self._update_batch_summary(result)
            self._append_log(f"Batch prediction completed for {len(result)} scenario(s).")
            self.status_var.set(f"Batch prediction complete for {len(result)} scenario(s).")
            self.notebook.select(self.batch_tab)
        except Exception as exc:
            self._handle_exception("Batch prediction failed", exc)

    def save_single_result(self) -> None:
        try:
            if self.last_single_result is None or self.last_single_result.empty:
                raise ValueError("No single prediction result is available yet.")
            output_path = Path(self.output_path_var.get().strip())
            self.last_single_result.to_csv(output_path, index=False)
            self._append_log(f"Single prediction saved: {output_path}")
            self.status_var.set(f"Saved single prediction to: {output_path}")
        except Exception as exc:
            self._handle_exception("Save failed", exc)

    def save_batch_result(self) -> None:
        try:
            if self.last_batch_result is None or self.last_batch_result.empty:
                raise ValueError("No batch prediction result is available yet.")
            output_path = Path(self.batch_output_path_var.get().strip())
            self.last_batch_result.to_csv(output_path, index=False)
            self._append_log(f"Batch prediction saved: {output_path}")
            self.status_var.set(f"Saved batch prediction to: {output_path}")
        except Exception as exc:
            self._handle_exception("Save failed", exc)

    def _populate_batch_tree(self, frame: pd.DataFrame) -> None:
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)
        for _, row in frame.iterrows():
            self.batch_tree.insert(
                "",
                "end",
                values=(
                    str(row["Name"]),
                    fmt_value(row["q_l_per_s_per_m"]),
                    fmt_value(row["q_p05_l_per_s_per_m"]),
                    fmt_value(row["q_p50_l_per_s_per_m"]),
                    fmt_value(row["q_p95_l_per_s_per_m"]),
                    str(row.get("range_warning", "") or ""),
                ),
            )

    def load_example_inp_path(self) -> None:
        candidate = DEFAULT_BATCH_INPUT
        self.batch_inp_path_var.set(str(candidate))
        self.status_var.set(f"Batch file path set to default input.txt: {candidate}")
        self._append_log(f"Batch input path set to: {candidate}")

    # ------------------------------------------------------------------
    # Browse actions
    # ------------------------------------------------------------------
    def browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select model file",
            initialdir=str(APP_DIR),
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        )
        if path:
            self.model_path_var.set(path)
            self.bundle = None
            self.loaded_model_path = None

    def browse_database(self) -> None:
        path = filedialog.askopenfilename(
            title="Select database CSV",
            initialdir=str(APP_DIR),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.database_path_var.set(path)

    def browse_output_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save single prediction CSV",
            initialdir=str(APP_DIR),
            initialfile=Path(self.output_path_var.get() or "single_prediction.csv").name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if path:
            self.output_path_var.set(path)

    def browse_diagnostics(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save diagnostics JSON",
            initialdir=str(APP_DIR),
            initialfile=Path(self.diagnostics_path_var.get() or "diagnostics.json").name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if path:
            self.diagnostics_path_var.set(path)

    def browse_inp(self) -> None:
        path = filedialog.askopenfilename(
            title="Select batch file",
            initialdir=str(APP_DIR),
            filetypes=[
                ("Batch files", "*.txt *.inp *.csv *.tsv *.dat"),
                ("Text files", "*.txt *.inp *.tsv"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.batch_inp_path_var.set(path)

    def browse_batch_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save batch prediction CSV",
            initialdir=str(APP_DIR),
            initialfile=Path(self.batch_output_path_var.get() or "batch_predictions.csv").name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if path:
            self.batch_output_path_var.set(path)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    def _handle_exception(self, prefix: str, exc: Exception) -> None:
        detail = f"{prefix}: {exc}"
        self.status_var.set(detail)
        self._append_log(detail)
        self._append_log(traceback.format_exc())
        messagebox.showerror(prefix, str(exc))



def main() -> int:
    os.chdir(APP_DIR)
    root = tk.Tk()
    app = OvertoppingGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
