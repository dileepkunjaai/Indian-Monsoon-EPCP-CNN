#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# In[ ]:


regions = ['IP', 'WCI', 'NWI', 'CNE', 'NEI', 'HR']

# ── Data location ──────────────────────────────────────────────────────────
# Point this at your own data directory (or set the EPCP_DATA_DIR env var).
# Expected layout:
#   <EPCP_DATA_DIR>/omega/omega.*.nc
#   <EPCP_DATA_DIR>/predicted_class/{region}/predicted_class_data_{region}-850hpa.csv
BASE_DIR = os.environ.get("EPCP_DATA_DIR", "./data")

omega_path = os.path.join(BASE_DIR, "omega")
class_prob_path = os.path.join(
    BASE_DIR, "predicted_class", "{region}", "predicted_class_data_{region}-850hpa.csv"
)

# Domain
LAT_MIN, LAT_MAX = -10, 30
LON_MIN, LON_MAX = 70, 90

# JJAS months
JJAS = [6, 7, 8, 9]

PRESSURE_LEVELS = [1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100.]


# In[ ]:


import warnings

def load_omega_anomalies(omega_path, lat_min, lat_max, lon_min, lon_max, jjas_months, pressure_levels):

    files = sorted(glob.glob(os.path.join(omega_path, "omega.*.nc")))
    if not files:
        raise FileNotFoundError(f"No omega files found in {omega_path}")
    print(f"Found {len(files)} omega files.")

    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', chunks={'time': 100})

    # --- Robust variable detection ---
    candidate_names = ['omega', 'OMEGA', 'w', 'W', 'vvel', 'VVEL']
    omega_var = None
    for name in candidate_names:
        if name in ds.data_vars:
            omega_var = name
            break
    if omega_var is None:
        omega_var = list(ds.data_vars)[0]
        print(f"Warning: Using first variable found: '{omega_var}'")
    else:
        print(f"Omega variable: '{omega_var}'")

    # --- Robust coordinate detection ---
    def find_coord(ds, candidates):
        for c in candidates:
            if c in ds.coords or c in ds.dims:
                return c
        return None

    lat_name  = find_coord(ds, ['lat', 'latitude', 'LAT', 'LATITUDE', 'y'])
    lon_name  = find_coord(ds, ['lon', 'longitude', 'LON', 'LONGITUDE', 'x'])
    lev_name  = find_coord(ds, ['level', 'lev', 'plev', 'pressure', 'LEVEL', 'LEV', 'PLEV'])
    time_name = find_coord(ds, ['time', 'TIME', 'Time'])

    print(f"Coordinates — lat:'{lat_name}', lon:'{lon_name}', lev:'{lev_name}', time:'{time_name}'")

    rename_map = {}
    if lat_name  and lat_name  != 'lat':   rename_map[lat_name]  = 'lat'
    if lon_name  and lon_name  != 'lon':   rename_map[lon_name]  = 'lon'
    if lev_name  and lev_name  != 'level': rename_map[lev_name]  = 'level'
    if time_name and time_name != 'time':  rename_map[time_name] = 'time'
    if rename_map:
        ds = ds.rename(rename_map)
        print(f"Renamed: {rename_map}")

    # --- Detect lat/lon axis order and slice accordingly ---
    lats_all = ds['lat'].values
    lons_all = ds['lon'].values

    # Handle descending lat (e.g. 90 -> -90)
    if lats_all[0] > lats_all[-1]:
        print(f"Latitude is descending ({lats_all[0]:.1f} -> {lats_all[-1]:.1f}). Reversing slice order.")
        lat_slice = slice(lat_max, lat_min)  # reversed
    else:
        print(f"Latitude is ascending ({lats_all[0]:.1f} -> {lats_all[-1]:.1f}).")
        lat_slice = slice(lat_min, lat_max)

    if lons_all[0] > lons_all[-1]:
        print(f"Longitude is descending. Reversing slice order.")
        lon_slice = slice(lon_max, lon_min)
    else:
        lon_slice = slice(lon_min, lon_max)

    # --- Subset level: match whichever values exist ---
    available_levs = ds['level'].values
    matched_levs   = [lv for lv in pressure_levels if lv in available_levs]
    print(f"Matched pressure levels: {matched_levs}")

    ds = ds.sel(lat=lat_slice, lon=lon_slice, level=matched_levs)

    # Confirm non-empty lat/lon after subset
    print(f"After spatial subset — lat size: {ds.sizes['lat']}, lon size: {ds.sizes['lon']}, level size: {ds.sizes['level']}")
    if ds.sizes['lat'] == 0 or ds.sizes['lon'] == 0:
        raise ValueError(
            f"Spatial subset returned empty lat/lon dimension!\n"
            f"Full lat range in data: {lats_all.min():.1f} to {lats_all.max():.1f}\n"
            f"Full lon range in data: {lons_all.min():.1f} to {lons_all.max():.1f}\n"
            f"Requested lat: {lat_min} to {lat_max}, lon: {lon_min} to {lon_max}"
        )

    # --- JJAS subset ---
    ds = ds.sel(time=ds['time'].dt.month.isin(jjas_months))
    print(f"JJAS timesteps: {ds.sizes['time']}")

    # --- Load into memory ---
    print("Loading omega data into memory (this may take a moment)...")
    da = ds[omega_var].load()
    print(f"Loaded. Shape: {da.shape}, dims: {da.dims}")

    # --- Standardized anomalies: bypass flox entirely using numpy ---
    # Convert to numpy for reliable groupby operations
    time_vals  = da['time'].values
    doy_vals   = da['time'].dt.dayofyear.values  # shape: (time,)
    data_np    = da.values                        # shape: (time, level, lat, lon)

    unique_doys = np.unique(doy_vals)
    clim_mean_np = np.full_like(data_np, np.nan)
    clim_std_np  = np.full_like(data_np, np.nan)

    for doy in unique_doys:
        idx = np.where(doy_vals == doy)[0]
        subset = data_np[idx]                          # (n_days_this_doy, level, lat, lon)
        mu  = np.nanmean(subset, axis=0)               # (level, lat, lon)
        sig = np.nanstd(subset, axis=0, ddof=1)        # (level, lat, lon)
        sig[sig == 0] = np.nan                         # avoid division by zero
        clim_mean_np[idx] = mu[np.newaxis, ...]
        clim_std_np[idx]  = sig[np.newaxis, ...]

    anom_np = (data_np - clim_mean_np) / clim_std_np

    # Wrap back into DataArray
    omega_anom = xr.DataArray(
        anom_np,
        coords=da.coords,
        dims=da.dims,
        name='omega_anom'
    )

    print("Standardized omega anomalies computed successfully.")
    return omega_anom


omega_anom = load_omega_anomalies(
    omega_path, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, JJAS, PRESSURE_LEVELS
)


# In[ ]:


def parse_dates_robustly(date_series):
    """
    Try multiple date formats to robustly parse a date column.
    Returns a pandas DatetimeIndex.
    """
    formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y',
        '%Y%m%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S'
    ]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(date_series, format=fmt)
            print(f"Dates parsed with format: '{fmt}'")
            return parsed
        except Exception:
            continue
    # fallback: let pandas infer
    parsed = pd.to_datetime(date_series, infer_datetime_format=True)
    print("Dates parsed with pandas inference.")
    return parsed


# In[ ]:


def get_region_composite(region, omega_anom, class_prob_path):
    """
    For a given region, load the predicted-class CSV, extract 'ones' dates
    (predicted_class == 1), and return the mean omega anomaly cross-section
    (level x lat), averaged over 70-90E.
    """
    # Load predicted class
    cp_file = class_prob_path.format(region=region)
    cp_df = pd.read_csv(cp_file)

    # Robustly find date column
    date_col = None
    for col in cp_df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    if date_col is None:
        date_col = cp_df.columns[0]
        print(f"[{region}] Warning: date column not found by name. Using first column: '{date_col}'")
    else:
        print(f"[{region}] Using date column: '{date_col}'")

    # Robustly find predicted class column
    class_col = None
    for col in cp_df.columns:
        if 'predicted' in col.lower() or 'class' in col.lower():
            class_col = col
            break
    if class_col is None:
        raise ValueError(f"[{region}] Could not find predicted class column in {cp_file}. Columns: {cp_df.columns.tolist()}")
    print(f"[{region}] Using class column: '{class_col}'")

    cp_df[date_col] = parse_dates_robustly(cp_df[date_col])

    # Get ones dates
    ones_dates = cp_df.loc[cp_df[class_col] == 1, date_col]
    print(f"[{region}] Number of ones dates: {len(ones_dates)}")

    # Select omega anomaly on ones dates
    ones_dates_np = ones_dates.values.astype('datetime64[ns]')
    
    # match available times
    available_times = omega_anom.time.values
    matched_dates = np.intersect1d(ones_dates_np, available_times)
    print(f"[{region}] Matched {len(matched_dates)} dates in omega dataset.")

    if len(matched_dates) == 0:
        raise ValueError(f"[{region}] No matching dates found between ones_dates and omega anomaly dataset.")

    # composite: mean over ones dates, then mean over lon (70-90E already subsetted)
    omega_ones = omega_anom.sel(time=matched_dates)
    composite = omega_ones.mean(dim='time').mean(dim='lon')  # (level x lat)

    return composite


# In[ ]:


composites = {}

for region in regions:
    print(f"\n--- Processing region: {region} ---")
    try:
        comp = get_region_composite(region, omega_anom, class_prob_path)
        composites[region] = comp
        print(f"[{region}] Composite shape: {comp.shape}")
    except Exception as e:
        print(f"[{region}] ERROR: {e}")
        composites[region] = None


# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
axes_flat = axes.flatten()

STANDARD_PLEVS  = [1000, 700, 500, 300, 200, 100]
STANDARD_LABELS = ['1000', '700', '500', '300', '200', '100']

region_titles = {
    'IP':  'IP', 'WCI': 'WCI', 'NWI': 'NWI',
    'CNE': 'CNE', 'NEI': 'NEI', 'HR':  'HR'
}

cf_list = []

for idx, region in enumerate(regions):
    ax       = axes_flat[idx]
    row, col = divmod(idx, 3)
    comp     = composites.get(region)

    if comp is None:
        ax.set_visible(False)
        continue

    # Ensure (level x lat) orientation
    if comp.dims[0] == 'lat':
        comp = comp.T

    lats = comp['lat'].values
    levs = comp['level'].values
    data = comp.values  # (n_levels, n_lats)

    # Symmetric autoscale at 98th percentile of abs values
    vmax      = np.nanpercentile(np.abs(data), 98)
    vmin      = -vmax
    levels_cf = np.linspace(vmin, vmax, 21)
    levels_cl = np.linspace(vmin, vmax, 11)

    # Filled contours
    cf = ax.contourf(lats, levs, data,
                     levels=levels_cf, cmap='RdBu', extend='both')
    cf_list.append(cf)

    # Contour lines
    cl = ax.contour(lats, levs, data,
                    levels=levels_cl, colors='k',
                    linewidths=0.5, alpha=0.45)
    ax.clabel(cl, inline=True, fontsize=10, fmt='%.1f')

    # --- Y-axis: log scale, standard pressure ticks ---
    ax.set_yscale('log')
    ax.set_ylim(1000, 100)
    ax.set_yticks(STANDARD_PLEVS)
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(STANDARD_LABELS))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Show y-axis label and ticks only on left column (col == 0)
    if col == 0:
        ax.set_ylabel('Pressure (hPa)', fontsize=14)
        ax.tick_params(axis='y', labelsize=14, left=True, labelleft=True)
    else:
        ax.tick_params(axis='y', labelsize=14, left=True, labelleft=False)
        ax.set_ylabel('')

    # --- X-axis: latitude label and tick labels only on second row (row == 1) ---
    ax.set_xlim(LAT_MIN, LAT_MAX)
    ax.set_xticks(np.arange(LAT_MIN, LAT_MAX + 1, 10))
    
    if row == 1:                 # second row: show label and tick labels
        ax.set_xlabel('Latitude (°N)', fontsize=14)
        ax.tick_params(axis='x', labelsize=14, labelbottom=True)
    else:                        # first row: hide label and tick labels
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=14, labelbottom=False)

    ax.set_title(region_titles.get(region, region), fontsize=15, fontweight='bold')
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)

# --- Common horizontal colorbar ---
valid_cfs = [c for c in cf_list if c is not None]
if valid_cfs:
    cbar = fig.colorbar(
        valid_cfs[-1], ax=axes,
        orientation='horizontal',
        fraction=0.03, pad=0.06, aspect=50,
        label='Standardized Omega Anomaly (σ)'
    )
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Standardized Std. Omega Anomaly (σ)', fontsize=14)

fig.suptitle(
    'Vertical Cross-Section of Omega Anomaly (70–90°E mean)\n 1979–2022',
    fontsize=17, fontweight='bold'
)

os.makedirs("figures", exist_ok=True)
plt.savefig("./figures/omega_vertical_crosssection_2x3.png", dpi=200, bbox_inches='tight')
plt.show()
print("Figure saved.")

