#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

print("Imports OK")


# In[ ]:


# Point this at your own data directory (or set the EPCP_DATA_DIR env var).
# Expected layout:
#   <EPCP_DATA_DIR>/anomalies/era5/*.nc
#   <EPCP_DATA_DIR>/anomalies/r1/*.nc
#   <EPCP_DATA_DIR>/predicted_class/era5/{combo}/{region}/predicted_class_data_*.csv
#   <EPCP_DATA_DIR>/predicted_class/r1/{combo}/{region}/predicted_class_data_*.csv
#   <EPCP_DATA_DIR>/precip/{region}/region_mean_precip_{region}.csv
BASE = Path(os.environ.get("EPCP_DATA_DIR", "./data"))

regions = ['IP', 'WCI', 'NWI', 'CNE', 'NEI', 'HR']

# ── ERA5 anomaly files ────────────────────────────────────────────────────────
_ERA5_DIR = BASE / "anomalies/era5"

ERA5_FILES = {
    'slp' : _ERA5_DIR / "era5_mslp_anomaly.nc",
    'gph' : _ERA5_DIR / "era5_850_gph_anomaly.nc",
    'u'   : _ERA5_DIR / "era5_850_uwnd_anomaly.nc",
    'v'   : _ERA5_DIR / "era5_850_vwnd_anomaly.nc",
    'shum': _ERA5_DIR / "era5_850_shum_anomaly.nc",
}

# ── R1 anomaly files ──────────────────────────────────────────────────────────
R1_FILES = {
    'slp' : BASE / "anomalies/r1/slp_anomalies.nc",
    'gph' : BASE / "anomalies/r1/hgt_anomalies_850.nc",
    'u'   : BASE / "anomalies/r1/uwnd_anomalies.nc",
    'v'   : BASE / "anomalies/r1/vwnd_anomalies.nc",
    'shum': BASE / "anomalies/r1/shum_anomalies.nc",
}

# ── Precipitation template (same across all combos per region) ────────────────
PRECIP_TMPL = str(BASE / "precip/{region}/region_mean_precip_{region}.csv")

# ── Six combos: 3 ERA5 runs + 3 R1 runs ──────────────────────────────────────
# For plotting we always need: SLP anomaly (shading) + U/V anomaly (vectors)
# Those are pulled from the respective dataset's files, regardless of training vars.

COMBOS = [
    dict(
        label        = "850hPa Q+850hPa GPH+SLP\n",
        dataset      = "ERA5",
        anom_files   = ERA5_FILES,
        pred_tmpl    = str(BASE / "predicted_class/era5/Q-GPH-SLP/{region}/predicted_class_data_{region}.csv"),
    ),
    dict(
        label        = "850hPa U+850hPa V+SLP\n",
        dataset      = "ERA5",
        anom_files   = ERA5_FILES,
        pred_tmpl    = str(BASE / "predicted_class/era5/U-V-SLP/{region}/predicted_class_data_{region}.csv"),
    ),
    dict(
        label        = "850hPa GPH+SLP\n", #ERA5\n
        dataset      = "ERA5",
        anom_files   = ERA5_FILES,
        pred_tmpl    = str(BASE / "predicted_class/era5/GPH-SLP/{region}/predicted_class_data_{region}.csv"),
    ),
    dict(
        label        = "850hPa Q+850hPa GPH+SLP\n", #R1\n
        dataset      = "R1",
        anom_files   = R1_FILES,
        pred_tmpl    = str(BASE / "predicted_class/r1/Q-GPH-SLP/{region}/predicted_class_data_{region}.csv"),
    ),
    dict(
        label        = "850hPa U+850hPa V+SLP\n",
        dataset      = "R1",
        anom_files   = R1_FILES,
        pred_tmpl    = str(BASE / "predicted_class/r1/U-V-SLP/{region}/predicted_class_data_{region}.csv"),
    ),
    dict(
        label        = "850hPa GPH+SLP\n",
        dataset      = "R1",
        anom_files   = R1_FILES,
        pred_tmpl    = str(BASE / "predicted_class/r1/GPH-SLP/{region}/predicted_class_data_{region}.csv"),
    ),
]

print(f"Configured {len(regions)} regions × {len(COMBOS)} combos = {len(regions)*len(COMBOS)} panels")


# In[ ]:


def standardize_field(da_composite, da_full):
    """
    Standardise a composite DataArray by the temporal std of the full anomaly field.
    da_composite : 2-D (lat, lon) composite mean
    da_full      : 3-D (time, lat, lon) full anomaly time series
    Returns a dimensionless (σ) DataArray on the same grid.
    """
    std = da_full.std(dim='time')
    # avoid division by zero
    std = std.where(std > 0)
    return da_composite / std


# In[ ]:


# ── skip these names when looking for the main data variable ─────────────────
_SKIP_VARS = {
    'lat', 'lon', 'latitude', 'longitude', 'time', 'level', 'plev',
    'pressure_level', 'height', 'valid_time', 'expver', 'number',
    'step', 'surface', 'mean', 'sd', 'lev',
}

# priority order when multiple candidates remain
_PRIORITY_VARS = [
    'slp_anom', 'msl_anom', 'mslp_anom', 'psl_anom', 'sp_anom',
    'uwnd_anom', 'vwnd_anom', 'u_anom', 'v_anom',
    'hgt_anom', 'z_anom', 'gph_anom', 'shum_anom', 'q_anom',
    'slp', 'msl', 'mslp', 'psl', 'sp',
    'uwnd', 'vwnd', 'u', 'v',
    'hgt', 'z', 'shum', 'q',
]


def get_varname(ds):
    """Return the primary meteorological variable name in a dataset robustly."""
    candidates = [v for v in ds.data_vars if v.lower() not in _SKIP_VARS]
    if len(candidates) == 1:
        return candidates[0]
    for p in _PRIORITY_VARS:
        if p in candidates:
            return p
    # last resort: longest name (often the most specific)
    return max(candidates, key=len)


def standardize_coords(ds):
    """Rename latitude/longitude → lat/lon; ensure lat is ascending."""
    rmap = {}
    if 'latitude' in ds.coords  and 'lat' not in ds.coords:
        rmap['latitude']  = 'lat'
    if 'longitude' in ds.coords and 'lon' not in ds.coords:
        rmap['longitude'] = 'lon'
    if rmap:
        ds = ds.rename(rmap)
    # ERA5 often has lat descending (90→-90); sort ascending for slice consistency
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))
    return ds


def load_anom(filepath):
    """Open a NetCDF file and return (dataset, varname)."""
    ds = xr.open_dataset(str(filepath))
    ds = standardize_coords(ds)
    vname = get_varname(ds)
    return ds, vname


def get_ones_dates(pred_csv):
    """
    Read predicted-class CSV; return numpy array of datetime64 dates
    where predicted_class == 1.
    """
    df = pd.read_csv(pred_csv)
    # find the date column flexibly
    date_col = next(
        (c for c in df.columns if c.lower() in ('date', 'time', 'datetime')),
        None
    )
    if date_col is None:
        # fall back: first column that parses as dates
        for c in df.columns:
            try:
                pd.to_datetime(df[c].iloc[:5])
                date_col = c
                break
            except Exception:
                continue
    if date_col is None:
        raise ValueError(f"No date column found in {pred_csv}. Columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    ones = df.loc[df['predicted_class'] == 1, date_col].values
    return ones


def composite(ds, vname, dates):
    """
    Select `dates` from ds[vname] and return the time mean.
    Converts everything to numpy.datetime64 before nearest-neighbour fallback.
    """
    da = ds[vname]
    # Always work in numpy.datetime64[ns] to avoid Timestamp vs datetime64 mismatch
    dates_np = np.array(dates, dtype='datetime64[ns]')
    
    try:
        # Try exact selection first (works when time axis already matches)
        out = da.sel(time=dates_np).mean(dim='time')
    except (KeyError, TypeError):
        # Nearest-neighbour fallback — compare datetime64 vs datetime64
        time_np = da.time.values.astype('datetime64[ns]')
        idx = [np.abs(time_np - t).argmin() for t in dates_np]
        out = da.isel(time=idx).mean(dim='time')
    return out


# ── quick file-existence check ────────────────────────────────────────────────
def check_files():
    missing = []
    for key, fpath in {**ERA5_FILES, **R1_FILES}.items():
        if not Path(fpath).exists():
            missing.append(str(fpath))
    for combo in COMBOS:
        for reg in regions:
            p = combo['pred_tmpl'].format(region=reg)
            if not Path(p).exists():
                missing.append(p)
            pc = PRECIP_TMPL.format(region=reg)
            if not Path(pc).exists():
                missing.append(pc)
    if missing:
        print(f"⚠ {len(missing)} missing file(s):")
        for m in missing:
            print("  ", m)
    else:
        print("✓ All files found.")

check_files()


# In[ ]:


print("Loading anomaly datasets (this may take a moment)…")

era5_slp_ds, era5_slp_var = load_anom(ERA5_FILES['slp'])
era5_u_ds,   era5_u_var   = load_anom(ERA5_FILES['u'])
era5_v_ds,   era5_v_var   = load_anom(ERA5_FILES['v'])

r1_slp_ds,  r1_slp_var   = load_anom(R1_FILES['slp'])
r1_u_ds,    r1_u_var     = load_anom(R1_FILES['u'])
r1_v_ds,    r1_v_var     = load_anom(R1_FILES['v'])

print(f"ERA5 → SLP: '{era5_slp_var}'  U: '{era5_u_var}'  V: '{era5_v_var}'")
print(f"R1   → SLP: '{r1_slp_var}'   U: '{r1_u_var}'   V: '{r1_v_var}'")

# Bundle for lookup by dataset tag
ANOM_CACHE = {
    'ERA5': dict(slp=(era5_slp_ds, era5_slp_var),
                 u  =(era5_u_ds,   era5_u_var),
                 v  =(era5_v_ds,   era5_v_var)),
    'R1'  : dict(slp=(r1_slp_ds,  r1_slp_var),
                 u  =(r1_u_ds,    r1_u_var),
                 v  =(r1_v_ds,    r1_v_var)),
}
print("Done.")


# In[ ]:


from matplotlib.lines import Line2D


# In[ ]:


import string
import cartopy.mpl.ticker as cticker
import matplotlib.colors as mcolors

# ── Spatial domain ────────────────────────────────────────────────────────────
LON_MIN, LON_MAX = 40, 110
LAT_MIN, LAT_MAX =  0,  35

# ── Contour levels ────────────────────────────────────────────────────────────
CLEV      = np.arange(-1.0, 1.01, 0.1)
CLEV_LINE = CLEV[::2]
CMAP      = 'RdBu_r'
norm      = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)

# ── Quiver ────────────────────────────────────────────────────────────────────
# With larger panels (~5.5" wide) we can afford slightly denser vectors
Q_SKIP  = {'ERA5': 6, 'R1': 1}
Q_SCALE = 7.5
Q_REF   = 1.0

# ── Ticks ─────────────────────────────────────────────────────────────────────
LAT_TICKS = [0, 10, 20, 30]
LON_TICKS = [40, 60, 80, 100]

PROJ = ccrs.PlateCarree()

# ── Panel alphabet labels (independent per figure) ───────────────────────────
single = list(string.ascii_lowercase)
double = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
PANEL_LABELS = single + double   # a–z, aa, ab, … enough for any grid

print("Shared settings ready.")


# In[ ]:


def plot_circulation_figure(combos_subset, dataset_label, out_path,
                             anom_cache, regions,
                             figsize=(16.5, 20)):
    """
    Draw a 6-row × 3-col composite circulation figure for one dataset
    (either ERA5 or R1) and save to out_path.

    Parameters
    ----------
    combos_subset  : list of 3 combo dicts (all same dataset)
    dataset_label  : str, e.g. 'ERA5' or 'R1'  — used in the suptitle
    out_path       : str, output PNG path
    anom_cache     : ANOM_CACHE dict
    regions        : list of region strings
    figsize        : (width, height) in inches
    """
    nrows = len(regions)
    ncols = len(combos_subset)   # should be 3
    is_era5 = (dataset_label == 'ERA5')

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=figsize,
        subplot_kw={'projection': PROJ},
    )
    fig.subplots_adjust(hspace=0.06, wspace=0.04)

    cache = anom_cache[dataset_label]

    for col, combo in enumerate(combos_subset):
        for row, region in enumerate(regions):
            ax        = axes[row, col]
            is_left   = (col == 0)
            is_bottom = (row == nrows - 1)

            # ── map background ────────────────────────────────────────────────
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=PROJ)
            ax.coastlines(resolution='50m', linewidth=0.8, color='k')
            ax.add_feature(cfeature.BORDERS, linewidth=0.4,
                           linestyle=':', edgecolor='0.4')
            ax.add_feature(cfeature.STATES,  linewidth=0.25, edgecolor='0.6')

            ax.gridlines(
                draw_labels=False, linewidth=0.35,
                color='gray', alpha=0.5, linestyle='--',
                xlocs=LON_TICKS, ylocs=LAT_TICKS,
            )

            # ── lat ticks (left column only) ──────────────────────────────────
            if is_left:
                ax.set_yticks(LAT_TICKS, crs=PROJ)
                ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
                ax.tick_params(axis='y', labelsize=12, left=True, labelleft=True)
            else:
                ax.set_yticks([])

            # ── lon ticks (bottom row only) ───────────────────────────────────
            if is_bottom:
                ax.set_xticks(LON_TICKS, crs=PROJ)
                ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
                ax.tick_params(axis='x', labelsize=12, bottom=True, labelbottom=True)
            else:
                ax.set_xticks([])

            # ── column title (top row only) ───────────────────────────────────
            if row == 0:
                ax.set_title(combo['label'].strip(), fontsize=14,
                             fontweight='bold', pad=7)

            # ── row label (left column only) ──────────────────────────────────
            if is_left:
                ax.text(-0.20, 0.5, region,
                        transform=ax.transAxes,
                        fontsize=14, fontweight='bold',
                        va='center', ha='right', rotation=90)

            # ── panel label (a), (b), … ───────────────────────────────────────
            idx = row * ncols + col
            ax.text(
                0.01, 0.97, f'({PANEL_LABELS[idx]})',
                transform=ax.transAxes,
                fontsize=13, fontweight='bold',
                va='top', ha='left', color='white',
                bbox=dict(facecolor='black', alpha=0.35,
                          pad=1.5, boxstyle='round,pad=0.2'),
                zorder=10,
            )

            # ── load predicted dates ──────────────────────────────────────────
            pred_csv = combo['pred_tmpl'].format(region=region)

            try:
                ones_dates = get_ones_dates(pred_csv)

                if len(ones_dates) == 0:
                    ax.text(0.5, 0.5, 'No class-1\ndays',
                            transform=ax.transAxes,
                            ha='center', va='center',
                            fontsize=11, color='gray')
                    continue

                # ── composites ───────────────────────────────────────────────
                slp_comp = composite(*cache['slp'], ones_dates)
                u_comp   = composite(*cache['u'],   ones_dates)
                v_comp   = composite(*cache['v'],   ones_dates)

                # ── standardise ERA5 (Pa → σ) ─────────────────────────────────
                if is_era5:
                    slp_comp = standardize_field(
                        slp_comp, cache['slp'][0][cache['slp'][1]])
                    u_comp   = standardize_field(
                        u_comp,   cache['u'][0][cache['u'][1]])
                    v_comp   = standardize_field(
                        v_comp,   cache['v'][0][cache['v'][1]])

                # ── spatial subset ────────────────────────────────────────────
                kw    = dict(lat=slice(LAT_MIN, LAT_MAX),
                             lon=slice(LON_MIN, LON_MAX))
                slp_s = slp_comp.sel(**kw)
                u_s   = u_comp.sel(**kw)
                v_s   = v_comp.sel(**kw)

                lons = slp_s.lon.values
                lats = slp_s.lat.values

                # ── filled contour ────────────────────────────────────────────
                ax.contourf(
                    lons, lats, slp_s.values,
                    levels=CLEV, cmap=CMAP, norm=norm,
                    extend='both', transform=PROJ,
                )

                # ── contour lines ─────────────────────────────────────────────
                cs = ax.contour(
                    lons, lats, slp_s.values,
                    levels=CLEV_LINE, colors='k',
                    linewidths=0.6, transform=PROJ,
                )
                ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

                # ── wind vectors ──────────────────────────────────────────────
                sk = Q_SKIP[dataset_label]
                qv = ax.quiver(
                    lons[::sk], lats[::sk],
                    u_s.values[::sk, ::sk],
                    v_s.values[::sk, ::sk],
                    scale=Q_SCALE, scale_units='inches',
                    width=0.003, headwidth=4, headlength=4,
                    color='k', transform=PROJ, zorder=5,
                )

                # quiver key in last column, first row
                if row == 0 and col == ncols - 1:
                    ax.quiverkey(qv, X=0.90, Y=1.07, U=Q_REF,
                                 label=f'{Q_REF} σ', labelpos='E',
                                 coordinates='axes',
                                 fontproperties={'size': 11})

                # ── N annotation ──────────────────────────────────────────────
                ax.text(0.02, 0.04, f'N={len(ones_dates)}',
                        transform=ax.transAxes, fontsize=12,
                        color='k',
                        bbox=dict(fc='white', alpha=0.6, pad=1))

            except FileNotFoundError as exc:
                ax.text(0.5, 0.5, 'File not\nfound',
                        transform=ax.transAxes,
                        ha='center', va='center',
                        fontsize=8, color='red')
                print(f"[MISSING]  {exc}")

            except Exception as exc:
                ax.text(0.5, 0.5, f'Error:\n{str(exc)[:55]}',
                        transform=ax.transAxes,
                        ha='center', va='center',
                        fontsize=7, color='darkred')
                print(f"[ERROR] {region}/{combo['label'].strip()}: {exc}")

    # ── shared colourbar ──────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes,
        orientation='vertical',
        fraction=0.013, pad=0.02,
        shrink=0.4, aspect=33,
    )
    cbar.set_label('SLP Standardised Anomaly (σ)', fontsize=12.5)
    cbar.set_ticks(np.arange(-1.0, 1.01, 0.25))
    cbar.ax.tick_params(labelsize=13)

    # ── title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f'EPCP Anomalous Circulation Composites — {dataset_label}  |  JJAS\n'
        'Shading: SLP std. anomaly   |   Vectors: 850 hPa wind std. anomaly',
        fontsize=16, fontweight='bold', y=0.93,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {out_path}")


# In[ ]:


ERA5_COMBOS = [c for c in COMBOS if c['dataset'] == 'ERA5']   # first 3

plot_circulation_figure(
    combos_subset  = ERA5_COMBOS,
    dataset_label  = 'ERA5',
    out_path       = './figures/EPCP_composite_ERA5_6x3.png',
    anom_cache     = ANOM_CACHE,
    regions        = regions,
    figsize        = (16.5, 20),
)


# In[ ]:


R1_COMBOS = [c for c in COMBOS if c['dataset'] == 'R1']   # last 3

plot_circulation_figure(
    combos_subset  = R1_COMBOS,
    dataset_label  = 'R1',
    out_path       = './figures/EPCP_composite_R1_6x3.png',
    anom_cache     = ANOM_CACHE,
    regions        = regions,
    figsize        = (16.5, 20),
)

