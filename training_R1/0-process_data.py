import glob
from pathlib import Path

import xarray as xr

from project_utils import parameters as param

NCEP_PATH = Path("./input_data/NCEP-NCAR-R1")
OUTPUT_DIR = Path("./data")
VARIABLES = ["shum", "uwnd", "vwnd"]

def compute_and_save_anomalies(var, ncep_path=NCEP_PATH, output_dir=OUTPUT_DIR):
    """
    Load NCEP/NCAR R1 daily fields for `var`, subset to the JJAS season and
    the study domain/level, and compute standardized day-of-year anomalies.

    Parameters
    ----------
    var : str
        Variable name as used in the source file names and in-file variable
        name (e.g. "shum", "uwnd", "vwnd").
    ncep_path : path-like
        Directory containing the raw NCEP/NCAR R1 files.
    output_dir : path-like
        Directory to write the anomaly file to.

    Returns
    -------
    xarray.DataArray
        Standardized anomalies, named "<var>_anom", also written to
        `<output_dir>/<var>_anomalies_r1_1979-2022.nc`.
    """
    files = sorted(glob.glob(str(ncep_path / f"{var}*.nc")))
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time").sel(
        lat=param.lat_bbox,
        lon=param.lon_bbox,
        level=param.hgt_level,
        time=param.time_period,
    )

    is_jjas = (ds["time.month"] >= 6) & (ds["time.month"] <= 9)
    da = ds[var].where(is_jjas, drop=True)

    clim_mean = da.groupby("time.dayofyear").mean(dim="time")
    clim_std = da.groupby("time.dayofyear").std(dim="time")
    anom = (da.groupby("time.dayofyear") - clim_mean).groupby("time.dayofyear") / clim_std
    anom.name = f"{var}_anom"

    output_dir.mkdir(parents=True, exist_ok=True)
    anom.to_netcdf(output_dir / f"{var}_anomalies_r1_1979-2022.nc")
    return anom

for var in VARIABLES:
    print(f"Processing {var}...")
    compute_and_save_anomalies(var)
