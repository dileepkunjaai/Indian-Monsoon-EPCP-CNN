#!/usr/bin/env python
# coding: utf-8

# # ERA5 vertically integrated moisture transport (VIMT) anomaly on EPCP days
# 
# For each of the six IMD homogeneous regions, composites the anomaly
# (EPCP-day mean minus full-period climatology) of ERA5 VIMT magnitude and
# direction (u/v components) on EPCP days (predicted_class == 1), and plots a
# 2x3 panel figure with shaded magnitude and wind-vector direction.
# 
# **Inputs** (set `EPCP_DATA_DIR`, see next cell):
# - ERA5 VIMT daily NetCDF (`vimt`, `vimt_u`, `vimt_v` variables)
# - Per-region predicted-class CSVs (CNN output)
# - India national outline + per-region homogeneous-zone shapefiles
# 
# **Output:** `figures/era5_flux_anom_composite_epcp.png`
# 

# In[ ]:


import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
import glob
import os


# In[ ]:


# 1. Define Paths
# Point this at your own data directory (or set the EPCP_DATA_DIR env var).
# Expected layout:
#   <EPCP_DATA_DIR>/vimt.nc
#   <EPCP_DATA_DIR>/shp/india_outline/india_JK.shp
#   <EPCP_DATA_DIR>/shp/homogeneous_india/{region}/*.shp
#   <EPCP_DATA_DIR>/predicted_class/{region}/predicted_class_data_{region}-850hpa.csv
BASE_DIR = os.environ.get("EPCP_DATA_DIR", "./data")

mflux_path = os.path.join(BASE_DIR, "vimt.nc")
INDIA_SHP = os.path.join(BASE_DIR, "shp", "india_outline", "india_JK.shp")
REGION_SHP_BASE = os.path.join(BASE_DIR, "shp", "homogeneous_india")
CSV_BASE = os.path.join(BASE_DIR, "predicted_class")

regions = ['IP', 'WCI', 'NWI', 'CNE', 'NEI', 'HR']

# 2. Open ERA5 moisture flux dataset
ds = xr.open_dataset(mflux_path)

# Extract lat/lon grids directly from the new ERA5 dataset
lons = ds['lon'].values
lats = ds['lat'].values
LON, LAT = np.meshgrid(lons, lats)

# 3. Load Shapefiles using geopandas
india_gdf = gpd.read_file(INDIA_SHP)

region_gdfs = {}
for r in regions:
    # Find the single .shp file inside the region's folder
    shp_files = glob.glob(os.path.join(REGION_SHP_BASE, r, "*.shp"))
    if len(shp_files) == 1:
        region_gdfs[r] = gpd.read_file(shp_files[0])
    else:
        print(f"Warning: Expected 1 shapefile for {r}, found {len(shp_files)}.")


# In[ ]:


# Calculate the base climatology (mean over all time steps) to subtract later
clim_u = ds['vimt_u'].mean(dim='time')
clim_v = ds['vimt_v'].mean(dim='time')
clim_mag = ds['vimt'].mean(dim='time')


# In[ ]:


# ==========================================
# 4. Setup Figure
# ==========================================
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14, 11),
                        subplot_kw={'projection': ccrs.PlateCarree()})
axs = axs.flatten()
panel_label = 'a'
map_extent = [65, 100, 0, 40]  # [lon_min, lon_max, lat_min, lat_max]
stride = 10


# ==========================================
# 5. Main Loop - Process Data and Plot
# ==========================================
for i, region in enumerate(regions):
    ax = axs[i]
    
    # Read predicted class CSV and extract EPCP dates (predicted_class == 1)
    csv_path = f"{CSV_BASE}/{region}/predicted_class_data_{region}-850hpa.csv"
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    epcp_dates = df.loc[df['predicted_class'] == 1, 'date'].values
    
    # Select ERA5 data for those dates
    ds_epcp = ds.sel(time=epcp_dates)
    
    # Compute anomalies (Composite mean - Climatology)
    anom_u = ds_epcp['vimt_u'].mean(dim='time') - clim_u
    anom_v = ds_epcp['vimt_v'].mean(dim='time') - clim_v
    anom_mag = ds_epcp['vimt'].mean(dim='time') - clim_mag
    
    # ── pcolormesh ──────────────────────────────────────────────────────────────
    p = ax.pcolormesh(LON, LAT, anom_mag,
                      cmap='Purples',       # ← was 'RdBu_r'
                      vmin=0, vmax=200,     # ← was -150/150; now 0–200 (magnitude only)
                      shading='auto',
                      transform=ccrs.PlateCarree())
    
    # Plot wind anomalies using quiver
# Subsampling stride — adjust between 3–6 depending on resolution

    q = ax.quiver(LON[::stride, ::stride], LAT[::stride, ::stride],
                  anom_u[::stride, ::stride], anom_v[::stride, ::stride],
                  transform=ccrs.PlateCarree(),
                  #color='#d9604a',          # ← salmon-red matching reference
                  scale=1400,
                  width=0.003,
                  headwidth=4, headlength=4)
    
    # Set map extent
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    
    # Add shapefiles
    india_gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1.0)
    region_gdfs[region].plot(ax=ax, edgecolor='indianred', facecolor='none', linewidth=1.5)
    
    # Map aesthetics
    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.MaxNLocator(integer=True, nbins=5)
    gl.ylocator = plt.MaxNLocator(integer=True, nbins=5)
    gl.xlabel_style = {'size': 13}  # Increased font
    gl.ylabel_style = {'size': 13}  # Increased font
    
    # Title and panel labels
    ax.set_title(region, fontsize=15, weight='bold')
    ax.text(-0.08, 1.05, f"({panel_label})", transform=ax.transAxes, 
            fontsize=14, weight='bold', va='top')
    panel_label = chr(ord(panel_label) + 1)
    
    # Add quiver key only to the first panel
    if i == 0:
        ax.quiverkey(q, X=0.79, Y=1.05, U=100,
                     label='100 kg m$^{-1}$ s$^{-1}$',
                     labelpos='E', coordinates='axes',
                     #color='#d9604a',                    # ← key arrow also red
                     fontproperties={'size': 11})

# ==========================================
# 6. Final Adjustments, Colorbar, and Save
# ==========================================
plt.subplots_adjust(hspace=0.3, wspace=0.25)
fig.suptitle('ERA5 Moisture Flux: SLP & 850hPa GPH Composite Anomaly (EPCP Days)', 
             weight='bold', size=16.8, y=0.96)

# Shared horizontal colorbar
cbar_ax = fig.add_axes([0.3, 0.04, 0.4, 0.02]) 
cbar = fig.colorbar(p, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label(r'Flux Magnitude Anomaly (kg m$^{-1}$ s$^{-1}$)', fontsize=12)
cbar.ax.tick_params(labelsize=13)

# Save the figure
output_path = "./figures/era5_flux_anom_composite_epcp.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fig.savefig(output_path, dpi=400, bbox_inches='tight')

plt.show()

