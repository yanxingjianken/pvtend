"""Physical and numerical constants for pvtend."""

import numpy as np

# --- Fundamental constants ---
R_EARTH: float = 6_371_000.0      # Earth mean radius [m]
OMEGA_E: float = 7.2921e-5        # Earth angular velocity [rad/s]
G0: float = 9.81                  # Gravitational acceleration [m/s²]
R_DRY: float = 287.05             # Specific gas constant for dry air [J/(kg·K)]
CP_DRY: float = 1004.0            # Specific heat at constant pressure [J/(kg·K)]
KAPPA: float = R_DRY / CP_DRY     # Poisson constant ≈ 0.286
L_V: float = 2.501e6              # Latent heat of vapourisation [J/kg]
R_V: float = 461.5                # Gas constant for water vapour [J/(kg·K)]
H_SCALE: float = 7000.0           # Scale height [m]

# --- QG omega equation defaults ---
SP19_DRY_FRACTION: float = 1.0 / 3.0  # ω_dry / ω_total ratio (Steinfeld & Pfahl 2019)
F_MIN_LAT: float = 5.0            # Min latitude for f clamping [deg]
GEO_SMOOTH_SIGMA: float = 1.5     # Gaussian smoothing sigma for geostrophic wind [grid pts]
LAT_QG_LO: float = 15.0           # QG taper: zero below this [deg]
LAT_QG_HI: float = 25.0           # QG taper: full above this [deg]
LAT_QG_POLAR: float = 85.0        # QG taper: zero poleward of this [deg] (omega.lat_taper)

# --- Default grid ---
DEFAULT_LEVELS: list[int] = [1000, 850, 700, 500, 400, 300, 250, 200, 100]
# Levels the 3-D piece winds are weighted-averaged over for the 2-D npz keys.
# Widened from [300, 250, 200] on 2026-08-10 so the 2-D summary spans the
# upper piece's tropopause band rather than only its top three levels:
# measured on CESM2 f09, the 350 K isentrope sits at 169-186 hPa and the
# 2 PVU surface near 250-300 hPa, so 400-200 brackets the tropopause without
# pulling the lower stratosphere into the average.
WAVG_LEVELS: list[int] = [400, 300, 250, 200]

# --- Mask threshold for negative PV anomaly region ---
# Only grid points with q' < 0 (SI) are included in the orthogonal-basis mask.
MASK_PV_THRESHOLD: float = 0.0   # [PVU in SI]  (strict < 0)

# --- Climatology ---
CLIM_VARIABLES: list[str] = ["u", "v", "w", "t", "pv", "z", "q"]
MONTH_ABBREVS: list[str] = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]

# --- Target NH grid (1.5° resolution) ---
TARGET_LAT = np.arange(90.0, -0.1, -1.5)   # (61,) 90°N → 0°N
TARGET_LON = np.arange(-180.0, 180.0, 1.5)  # (240,)

# --- Default event-centred patch size ---
# npz patch half-widths.  Widened from 21/36 deg on 2026-08-10.
# Latitude stops at 30 rather than 40: at +-40 deg, 69.4 % of blocking peaks
# (median 56.9 N, p90 72.4 N) would extend past the pole and be NaN-padded;
# +-30 brings that to 41.3 % with a mean overrun of 8.4 deg.  Longitude is
# periodic so +-60 costs nothing there -- but it does force the inversion box
# wider (see INV_LON_HALF), or the patch edge would sit on the box boundary
# where psi is prescribed rather than solved.
LAT_HALF: float = 30.0   # half-width in latitude [deg]
LON_HALF: float = 60.0   # half-width in longitude [deg]

# --- RWB classification ---
RWB_CLASSIFY_LEVELS: list[int] = [300, 250, 200]
RWB_CLASSIFY_THRESHOLD: int = 3  # all levels must agree
