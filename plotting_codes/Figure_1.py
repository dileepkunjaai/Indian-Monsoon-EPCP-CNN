#!/usr/bin/env python
# coding: utf-8

# # Figure 1 — IMD rainfall climatology and regional annual cycles
# 
# Builds the combined climatology figure: (row 0) spatial maps of annual
# climatology, JJAS climatology, and JJAS p95 daily rainfall over India; (rows
# 1-2) daily-mean annual cycle per IMD homogeneous region, with JJAS months
# highlighted.
# 
# **Inputs** (set `EPCP_DATA_DIR`, see next cell):
# - IMD gridded daily rainfall NetCDF (1979-2022 or your own period)
# - India national outline shapefile
# - Per-region homogeneous-zone shapefiles (IP, WCI, NWI, CNE, NEI, HR)
# 
# **Output:** `figures/fig1_imd_climatology.png`
# 

# In[ ]:


import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
from glob import glob
import os
import warnings
warnings.filterwarnings('ignore')

# ── Data location ──────────────────────────────────────────────────────────────
# Point this at your own data directory (or set the EPCP_DATA_DIR env var).
# Expected layout:
#   <EPCP_DATA_DIR>/imd_rainfall.nc
#   <EPCP_DATA_DIR>/shp/india_outline/india_JK.shp
#   <EPCP_DATA_DIR>/shp/homogeneous_india/{region}/*.shp
BASE_DIR = os.environ.get("EPCP_DATA_DIR", "./data")

IMD_FULL        = os.path.join(BASE_DIR, "imd_rainfall.nc")
INDIA_SHP       = os.path.join(BASE_DIR, "shp", "india_outline", "india_JK.shp")
REGION_SHP_BASE = os.path.join(BASE_DIR, "shp", "homogeneous_india")

REGIONS = ['IP', 'WCI', 'NWI', 'CNE', 'NEI', 'HR']
JJAS    = [6, 7, 8, 9]

print("Imports OK")


# In[ ]:


ds = xr.open_dataset(IMD_FULL)
print(ds)

# Auto-detect rainfall variable
rain_var = [v for v in ds.data_vars][0]
print(f"\nRainfall variable detected: '{rain_var}'")
da = ds[rain_var]

# Standardise coordinate names → lat, lon, time
rename_map = {}
for c in list(da.coords):
    if c.lower() == 'latitude':
        rename_map[c] = 'lat'
    if c.lower() == 'longitude':
        rename_map[c] = 'lon'
if rename_map:
    da = da.rename(rename_map)

time_dim = [d for d in da.dims if 'time' in d.lower()][0]
if time_dim != 'time':
    da = da.rename({time_dim: 'time'})

print(da)
print("\nTime range :", str(da.time.values[0])[:10], "→", str(da.time.values[-1])[:10])
print("Lat range  :", float(da.lat.min()), "–", float(da.lat.max()))
print("Lon range  :", float(da.lon.min()), "–", float(da.lon.max()))


# In[ ]:


# ── JJAS subset ───────────────────────────────────────────────────────────────
da_jjas = da.sel(time=da['time.month'].isin(JJAS))

# (a) Annual climatology — daily → monthly sum → long-term mean [mm/month]
da_monthly  = da.resample(time='1MS').sum(skipna=True)
annual_clim = da_monthly.mean('time')

# (b) JJAS climatology [mm/month]
da_jjas_monthly = da_jjas.resample(time='1MS').sum(skipna=True)
jjas_clim       = da_jjas_monthly.mean('time')

# (c) p95 of daily rainfall during JJAS — all days (wet_threshold=0)
wet_threshold = 0.0
da_jjas_wet   = da_jjas.where(da_jjas > wet_threshold)
p95_jjas      = da_jjas_wet.quantile(0.95, dim='time', skipna=True)

print("Annual clim :", float(annual_clim.min()), "–", float(annual_clim.max()), "mm/month")
print("JJAS clim   :", float(jjas_clim.min()),   "–", float(jjas_clim.max()),   "mm/month")
print("p95 JJAS    :", float(p95_jjas.min()),     "–", float(p95_jjas.max()),    "mm/day")


# In[ ]:


india_shp = gpd.read_file(INDIA_SHP)
if india_shp.crs and india_shp.crs.to_epsg() != 4326:
    india_shp = india_shp.to_crs(epsg=4326)
print("India CRS:", india_shp.crs)

region_gdfs = {}
for reg in REGIONS:
    shp_files = glob(f"{REGION_SHP_BASE}/{reg}/*.shp")
    assert len(shp_files) == 1, f"Expected 1 shp for {reg}, found: {shp_files}"
    gdf = gpd.read_file(shp_files[0])
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    region_gdfs[reg] = gdf
    print(f"  {reg}: {shp_files[0]}  ({len(gdf)} polygon(s))")


# In[ ]:


def mask_to_india(data_array, gdf):
    """
    Set all pixels outside gdf to NaN using regionmask.
    Works correctly regardless of lat orientation (ascending or descending).
    """
    lons = data_array.lon.values
    lats = data_array.lat.values

    gdf_union = gdf.dissolve().reset_index(drop=True)
    regions   = regionmask.from_geopandas(gdf_union)
    mask      = regions.mask(lons, lats)   # NaN outside, 0 inside

    inside  = ~np.isnan(mask.values)       # True = inside India
    masked  = data_array.copy().astype(float)
    masked.values[~inside] = np.nan
    return masked


print("Masking climatologies to India boundary...")
annual_clim_masked = mask_to_india(annual_clim, india_shp)
jjas_clim_masked   = mask_to_india(jjas_clim,   india_shp)
p95_jjas_masked    = mask_to_india(p95_jjas,     india_shp)

for name, arr in [('Annual clim', annual_clim_masked),
                  ('JJAS clim',   jjas_clim_masked),
                  ('p95 JJAS',    p95_jjas_masked)]:
    n_valid = (~np.isnan(arr.values)).sum()
    n_nan   = np.isnan(arr.values).sum()
    print(f"  {name}: {n_valid} valid pixels, {n_nan} NaN (outside India)")


# In[ ]:


def get_region_daily_mean(da_full, gdf):
    """
    Spatially average da_full over the region defined by gdf.
    Returns a 1-D DataArray (time).
    """
    lons = da_full.lon.values
    lats = da_full.lat.values

    gdf_union = gdf.dissolve().reset_index(drop=True)
    regions   = regionmask.from_geopandas(gdf_union)
    mask      = regions.mask(lons, lats)
    inside    = ~np.isnan(mask.values)

    da_masked = da_full.where(inside)
    da_mean   = da_masked.mean(dim=['lat', 'lon'], skipna=True)
    return da_mean


region_full_daily = {}
for reg in REGIONS:
    print(f"Processing {reg} ...", end=' ')
    dm = get_region_daily_mean(da, region_gdfs[reg])
    region_full_daily[reg] = dm
    print(f"done  ({len(dm.time)} days)")


# In[ ]:


region_annual_cycle = {}

for reg in REGIONS:
    dm       = region_full_daily[reg].to_series()
    dm.index = pd.to_datetime(dm.index)

    # Climatological daily mean grouped by month → 12-value series
    monthly  = dm.groupby(dm.index.month).mean()
    monthly  = monthly.reindex(range(1, 13))
    region_annual_cycle[reg] = monthly

print("Annual cycles built.")
for reg in REGIONS:
    jjas_vals = [f"{region_annual_cycle[reg][m]:.2f}" for m in JJAS]
    print(f"  {reg} JJAS means (J/J/A/S): {jjas_vals}")


# In[ ]:


month_labels       = ['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec']
JJAS_months        = [6, 7, 8, 9]
panel_labels_cycle = ['(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

lon_min, lon_max = float(da.lon.min()), float(da.lon.max())

lat_min, lat_max = float(da.lat.min()), float(da.lat.max())
extent = [lon_min - 0.5, lon_max + 0.5, lat_min - 0.5, lat_max + 0.5]

# ── Discrete colormap helper ──────────────────────────────────────────────────
def discrete_cmap_norm(vmin, vmax, n_bins, base_cmap='jet'):
    base   = mpl.colormaps[base_cmap].resampled(n_bins)
    colors = base(np.linspace(0, 1, n_bins))
    cmap   = mpl.colors.ListedColormap(colors)
    cmap.set_bad(color='white')   # NaN (outside India) → white
    bounds = np.linspace(vmin, vmax, n_bins + 1)
    norm   = mpl.colors.BoundaryNorm(bounds, ncolors=n_bins, clip=True)
    return cmap, norm, bounds

# ── Panel specs ───────────────────────────────────────────────────────────────
panels = [
    (annual_clim_masked, 'Annual Climatology',  0, 400, 10, 'mm/month', '(a)'),
    (jjas_clim_masked,   'JJAS Climatology',    0, 400, 10, 'mm/month', '(b)'),
    (p95_jjas_masked,    'p95 During JJAS',     0, 100, 10, 'mm/day',   '(c)'),
]

# ── Figure & GridSpec ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 18))
gs  = gridspec.GridSpec(3, 3, figure=fig,
                         height_ratios=[1.5, 1, 1],
                         hspace=0.17,
                         wspace=0.04)

# ── Row 0: Spatial maps ───────────────────────────────────────────────────────
for col, (data, title, vmin, vmax, n_bins, unit, label) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col], projection=ccrs.PlateCarree())

    cmap, norm, bounds = discrete_cmap_norm(vmin, vmax, n_bins)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor('white')
    ax.spines['geo'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    im = ax.pcolormesh(da.lon.values, da.lat.values,
                       data.values,
                       cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), zorder=2)

    # India outer boundary
    india_shp.boundary.plot(ax=ax, color='black', linewidth=0.9,
                             transform=ccrs.PlateCarree(), zorder=6)
    # Homogeneous region boundaries
    for reg, gdf in region_gdfs.items():
        gdf.boundary.plot(ax=ax, color='black', linewidth=0.8,
                          linestyle='--',
                          transform=ccrs.PlateCarree(), zorder=5)

    # Discrete colorbar — ticks at bin centres
    tick_locs   = (bounds[:-1] + bounds[1:]) / 2
    tick_labels = [f'{int(v)}' for v in tick_locs]
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.04, shrink=0.95, aspect=22, ticks=tick_locs)
    cbar.ax.set_xticklabels(tick_labels, fontsize=11, rotation=45, ha='right')
    cbar.set_label(unit, fontsize=16, labelpad=3)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=6)
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')

# ── Rows 1–2: Annual cycle panels ────────────────────────────────────────────
all_vals = np.concatenate([region_annual_cycle[r].values for r in REGIONS])
y_max    = np.nanmax(all_vals) * 1.08

for idx, reg in enumerate(REGIONS):
    row = 1 + idx // 3
    col = idx % 3
    ax  = fig.add_subplot(gs[row, col])

    cycle = region_annual_cycle[reg]
    x     = np.arange(1, 13)
    y     = cycle.values.astype(float)

    ax.plot(x, y, color='black', linewidth=2.0, zorder=3)

    # Gold dots on JJAS months
    jjas_y = [cycle[m] for m in JJAS_months]
    ax.scatter(JJAS_months, jjas_y, color='gold', edgecolors='black',
               s=70, zorder=5, linewidths=0.7)

    ax.set_xticks(x)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, y_max)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.2)

    ax.set_title(reg, fontsize=16, fontweight='bold', pad=6)
    ax.text(0.02, 0.97, panel_labels_cycle[idx], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')

    # x-tick labels: bottom row only
    if row == 2:
        ax.set_xticklabels(month_labels, fontsize=12, rotation=45, ha='right')
    else:
        ax.set_xticklabels([])

    # y-label and ticks: left column only
    if col == 0:
        ax.set_ylabel('Rainfall (mm/day)', fontsize=13)
    else:
        ax.set_yticklabels([])

    # x-label: centre panel of bottom row only
    if idx == 4:
        ax.set_xlabel('Month', fontsize=13)

plt.suptitle("IMD Rainfall: Climatology and Regional Annual Cycles (1979–2022)",
             fontsize=18, fontweight='bold', y=0.92)

os.makedirs("figures", exist_ok=True)
out_path = "figures/fig1_imd_climatology.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved: {out_path}")

