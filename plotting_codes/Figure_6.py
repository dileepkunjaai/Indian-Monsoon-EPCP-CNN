#!/usr/bin/env python
# coding: utf-8

# # ERA5/IMD precipitation vs. moisture-flux convergence anomalies on EPCP days
# 
# For each of the six IMD homogeneous regions, composites the anomaly (EPCP-day
# mean minus JJAS climatology) of ERA5 precipitation, IMD precipitation, and
# ERA5 vertically integrated moisture flux convergence (VIMFC), and plots a
# 6x3 panel figure with region and national outline overlays.
# 
# **Inputs** (set `EPCP_DATA_DIR`, see next cell):
# - IMD gridded daily rainfall NetCDF
# - VIMFC daily NetCDF
# - ERA5 daily total precipitation NetCDFs (one or more files)
# - Per-region predicted-class CSVs (CNN output)
# - India national outline + per-region homogeneous-zone shapefiles
# 
# **Output:** `figures/era5_anomaly_common_cbar_labeled.png`
# 

# In[ ]:


import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


regions = ['IP', 'CNE', 'HR', 'NWI', 'WCI', 'NEI']

# As requested, using the box 65-100E and 0-40N strictly for plotting extent
plot_extent = [65, 100, 0, 40]  # [lon_min, lon_max, lat_min, lat_max]


# In[ ]:


# Point this at your own data directory (or set the EPCP_DATA_DIR env var).
# Expected layout:
#   <EPCP_DATA_DIR>/imd_rainfall.nc
#   <EPCP_DATA_DIR>/vimfc.nc
#   <EPCP_DATA_DIR>/era5_precip/era5_daily_total_precip_*.nc
#   <EPCP_DATA_DIR>/predicted_class/{region}/predicted_class_data_{region}-850hpa.csv
#   <EPCP_DATA_DIR>/shp/india_outline/india_JK.shp
#   <EPCP_DATA_DIR>/shp/homogeneous_india/{region}/*.shp
BASE_DIR = os.environ.get('EPCP_DATA_DIR', './data')
CSV_BASE = os.path.join(BASE_DIR, 'predicted_class')

imd_path = os.path.join(BASE_DIR, 'imd_rainfall.nc')
vimfc_path = os.path.join(BASE_DIR, 'vimfc.nc')
precip_dir = os.path.join(BASE_DIR, 'era5_precip')

# 1. Load IMD
print("Loading IMD...")
imdjjas = xr.open_dataset(imd_path)

# 2. Load VIMFC
print("Loading VIMFC...")
vimfc_ds = xr.open_dataset(vimfc_path)
if 'time' in vimfc_ds.dims:
    vimfc_ds = vimfc_ds.rename({'time': 'valid_time'})

# 3. Load ERA5 Precip (Modified to prevent kernel crash)
print("Loading ERA5 Precip files...")
era5_files = sorted(glob.glob(os.path.join(precip_dir, 'era5_daily_total_precip_*.nc')))

# Remove parallel=True to avoid thread-safe netCDF crashes.
# Add chunks to use Dask lazily, preventing RAM overload.
era5_precip = xr.open_mfdataset(
    era5_files, 
    combine='by_coords', 
    parallel=False,              # Turned off parallel to prevent NetCDF ID crashes
    chunks={'valid_time': 30}    # Read 30 days (~1 month) at a time
)

print("Datasets loaded successfully.")
print(f"ERA5 Precip size: {era5_precip.sizes}")


# In[ ]:


def get_dim_name(ds_or_da, target):
    """Robustly find dimension name containing target string."""
    for dim in ds_or_da.dims:
        if target in dim.lower():
            return dim
    return None

def get_var_name(ds, preferred_list):
    """Robustly find variable name from preferred list or fallback to first."""
    for pref in preferred_list:
        if pref in ds.data_vars:
            return pref
    return list(ds.data_vars)[0]

def convert_units(da, var_name):
    """Robustly check units and convert to mm/day."""
    # Take a small sample to check max value without loading everything
    time_dim = get_dim_name(da, 'time')
    sample = da.isel({time_dim: 0})
    max_val = float(sample.max())
    
    if var_name in ['precip', 'rain', 'tp']:
        if max_val < 1.0:  # likely m/day
            print(f"  -> Converting {var_name} (max={max_val:.4f}) from m to mm (x1000)")
            return da * 1000.0
    elif var_name in ['vimfc', 'mfc']:
        if max_val < 1.0:  # likely kg/m2/s
            print(f"  -> Converting {var_name} (max={max_val:.4f}) from kg/m2/s to mm/day (x86400)")
            return da * 86400.0
    return da

def jjas_subset(ds):
    time_name = get_dim_name(ds, 'time')
    return ds.sel({time_name: ds[time_name].dt.month.isin([6, 7, 8, 9])})

def get_class1_dates(region):
    csv_path = os.path.join(CSV_BASE, region, f'predicted_class_data_{region}-850hpa.csv')
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df.loc[df['predicted_class'] == 1, 'date'].values


# In[ ]:


# Subset JJAS
imd_jjas = jjas_subset(imdjjas)
vimfc_jjas = jjas_subset(vimfc_ds)
era5p_jjas = jjas_subset(era5_precip)

# Get variable names robustly
imd_var = get_var_name(imd_jjas, ['rain', 'precip', 'rf'])
era5_var = get_var_name(era5p_jjas, ['precip', 'tp'])
vimfc_var = get_var_name(vimfc_jjas, ['vimfc', 'mfc'])

print(f"IMD variable: {imd_var}")
print(f"ERA5 variable: {era5_var}")
print(f"VIMFC variable: {vimfc_var}")

# Apply robust unit conversion
print("Checking and converting units...")
imd_jjas[imd_var] = convert_units(imd_jjas[imd_var], imd_var)
era5p_jjas[era5_var] = convert_units(era5p_jjas[era5_var], era5_var)
vimfc_jjas[vimfc_var] = convert_units(vimfc_jjas[vimfc_var], vimfc_var)


# In[ ]:


anomalies = {}

for region in regions:
    print(f"Processing region: {region}")
    dates = pd.to_datetime(get_class1_dates(region))
    
    # ERA5 Precip anomaly
    era5_time = get_dim_name(era5p_jjas, 'time')
    era5_sel = era5p_jjas.sel({era5_time: dates}, method='nearest')
    era5_anom = era5_sel.mean(dim=era5_time) - era5p_jjas.mean(dim=era5_time)
    
    # IMD Precip anomaly
    imd_time = get_dim_name(imd_jjas, 'time')
    imd_sel = imd_jjas.sel({imd_time: dates}, method='nearest')
    imd_anom = imd_sel.mean(dim=imd_time) - imd_jjas.mean(dim=imd_time)
    
    # VIMFC anomaly
    vimfc_time = get_dim_name(vimfc_jjas, 'time')
    vimfc_sel = vimfc_jjas.sel({vimfc_time: dates}, method='nearest')
    vimfc_anom = vimfc_sel.mean(dim=vimfc_time) - vimfc_jjas.mean(dim=vimfc_time)
    
    # Store as DataArrays
    anomalies[region] = {
        'era5p': era5_anom[era5_var],
        'imd': imd_anom[imd_var],
        'vimfc': vimfc_anom[vimfc_var]
    }
print("Anomalies computed successfully.")


# In[ ]:


import cartopy.io.shapereader as shpreader
import matplotlib.ticker as mticker
import string

# Plotting parameters
levels = np.linspace(0, 10, 11) 
cmap = 'Purples'  # Standard matplotlib colormap

# Load the India_JandK shapefile
india_shp_path = os.path.join(BASE_DIR, 'shp', 'india_outline', 'india_JK.shp')
india_reader = shpreader.Reader(india_shp_path)

# Create figure and axes
fig, axes = plt.subplots(len(regions), 3, 
                         figsize=(15, 32), 
                         subplot_kw={'projection': ccrs.PlateCarree()})

col_titles = ['Precipitation (ERA5)', 'Precipitation (IMD)', 'Calculated Convergence (ERA5)']
var_keys = ['era5p', 'imd', 'vimfc']

panel_idx = 0
cf = None  # To store the contourf plot object for the colorbar

for i, region in enumerate(regions):
    # Find regional shapefile
    shp_pattern = os.path.join(BASE_DIR, 'shp', 'homogeneous_india', region, '*.shp')
    shp_files = glob.glob(shp_pattern)
    
    for j, vname in enumerate(var_keys):
        panel_idx += 1
        ax = axes[i, j]
        da = anomalies[region][vname]
        
        lat_name = get_dim_name(da, 'lat')
        lon_name = get_dim_name(da, 'lon')
        
        # Plot data with extend='max' (only extend the upper bound)
        cf = ax.contourf(da[lon_name], da[lat_name], da.values, 
                         levels=levels, cmap=cmap, extend='max', 
                         transform=ccrs.PlateCarree())
        
        # Add coastlines and borders
        ax.coastlines('10m', linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle='--')
        
        # Set plot extent
        ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
        
        # 2. Overlay India_JK shapefile for all plots (black solid line)
        ax.add_geometries(india_reader.geometries(), ccrs.PlateCarree(), 
                          edgecolor='black', facecolor='none', linewidth=1.8)
        
        # Overlay regional shapefile (solid black, thicker line to highlight)
        if shp_files:
            reg_reader = shpreader.Reader(shp_files[0])
            ax.add_geometries(reg_reader.geometries(), ccrs.PlateCarree(), 
                              edgecolor='indianred', facecolor='none', linewidth=2.5, linestyle='-')
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Decrease tick density: set ticks every 10 degrees
        gl.xlocator = mticker.MultipleLocator(10)
        #gl.ylocator = mticker.MultipleLocator(10)
        
        gl.xlabel_style = {'size': 14}  # Increased font
        gl.ylabel_style = {'size': 14}  # Increased font
        
        # 3. Add panel labels (a, b, c, ...)
        panel_label = f'({string.ascii_lowercase[panel_idx - 1]})'
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, 
                fontsize=18, fontweight='bold', va='top', color='black',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Row and Column titles - Moved x to -0.15 to shift further outward
        if j == 0:
            ax.text(-0.25, 0.5, region, transform=ax.transAxes, 
                    fontsize=18, fontweight='bold', va='center', rotation=90)
        if i == 0:
            ax.set_title(col_titles[j], fontsize=16, fontweight='bold')

# 4. Make one common colorbar on the foot of the plots
# [left, bottom, width, height] in figure coordinates
cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.015]) 
cb = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', extend='max')
cb.set_label('Anomaly (mm/day)', fontsize=16, fontweight='bold')
cb.ax.tick_params(labelsize=14)

# 5. Increase font size for suptitle
plt.suptitle('Anomaly: Precipitation vs Moisture Convergence (EPCP Days)', 
             fontsize=22, fontweight='bold', y=0.98)

# Adjust layout to make room for the common colorbar and the outward shifted labels
# Changed 'left' to 0.05 to accommodate the -0.15 text shift
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.07, hspace=0.25, wspace=0.15)
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/era5_anomaly_common_cbar_labeled.png', dpi=200, bbox_inches='tight')
plt.show()

