"""Physical and numerical constants for pvtend."""

import numpy as np

# --- Fundamental constants ---
R_EARTH: float = 6_371_000.0      # Earth mean radius [m]
OMEGA_E: float = 7.2921e-5        # Earth angular velocity [rad/s]
G0: float = 9.81                  # Gravitational acceleration [m/s²]
R_DRY: float = 287.05             # Specific gas constant for dry air [J/(kg·K)]
CP_DRY: float = 1004.0            # Specific heat at constant pressure [J/(kg·K)]
KAPPA: float = R_DRY / CP_DRY     # Poisson constant ≈ 0.286
H_SCALE: float = 7000.0           # Scale height [m]

# --- QG omega equation defaults ---
SIGMA0_CONST: float = 2.0e-6      # Constant static stability [m² Pa⁻² s⁻²]
F_MIN_LAT: float = 5.0            # Min latitude for f clamping [deg]
GEO_SMOOTH_SIGMA: float = 1.5     # Gaussian smoothing sigma for geostrophic wind [grid pts]
LAT_QG_LO: float = 15.0           # QG taper: zero below this [deg]
LAT_QG_HI: float = 25.0           # QG taper: full above this [deg]
LAT_QG_POLAR: float = 80.0        # QG taper: polar taper start [deg]

# --- Default grid ---
DEFAULT_LEVELS: list[int] = [1000, 850, 700, 500, 400, 300, 250, 200, 100]
WAVG_LEVELS: list[int] = [300, 250, 200]

# --- Climatology ---
CLIM_VARIABLES: list[str] = ["u", "v", "w", "t", "pv", "z"]
MONTH_ABBREVS: list[str] = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]

# --- Target NH grid (1.5° resolution) ---
TARGET_LAT = np.arange(90.0, -0.1, -1.5)   # (61,) 90°N → 0°N
TARGET_LON = np.arange(-180.0, 180.0, 1.5)  # (240,)

# --- Default event-centred patch size ---
LAT_HALF: float = 21.0   # half-width in latitude [deg]
LON_HALF: float = 36.0   # half-width in longitude [deg]

# --- RWB classification ---
RWB_CLASSIFY_LEVELS: list[int] = [300, 250, 200]
RWB_CLASSIFY_THRESHOLD: int = 3  # all levels must agree
