#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np


# In[ ]:


# Point this at your own data directory (or set the EPCP_DATA_DIR env var).
# Expected layout: <EPCP_DATA_DIR>/shp/homogeneous_india/{region}/*.shp
BASE_DIR = os.environ.get('EPCP_DATA_DIR', './data')
REGION_SHP_BASE = os.path.join(BASE_DIR, 'shp', 'homogeneous_india')

regions_meta = {
    'IP':  {'color': '#4CAF50', 'label': 'Peninsular India (IP)'},
    'WCI': {'color': '#7B5EA7', 'label': 'West Central India (WCI)'},
    'NWI': {'color': '#F4A03A', 'label': 'Northwest India (NWI)'},
    'CNE': {'color': '#4BACC6', 'label': 'Central Northeast (CNE)'},
    'NEI': {'color': '#E05C5C', 'label': 'Northeast India (NEI)'},
    'HR':  {'color': '#C8A97E', 'label': 'Hilly Regions (HR)'},
}

# Shapefile names inside each region folder aren't predictable, so
# auto-detect rather than hardcode each one.
gdfs = {}
for key in regions_meta:
    shp_files = glob.glob(os.path.join(REGION_SHP_BASE, key, '*.shp'))
    assert len(shp_files) == 1, f'Expected 1 shapefile for {key}, found: {shp_files}'
    gdfs[key] = gpd.read_file(shp_files[0])


# In[ ]:


fig, ax = plt.subplots(figsize=(7, 9), dpi=150)
#fig.patch.set_facecolor('#F7F9FC')
#ax.set_facecolor('#D6E8F5')   # ocean/background colour

# --- plot regions ---
for key, meta in regions_meta.items():
    gdfs[key].plot(
        ax=ax,
        color=meta['color'],
        edgecolor='#2c2c2c',
        linewidth=1,
        alpha=0.85,
        zorder=2,
    )

# --- region abbreviation labels at centroids ---
label_offsets = {          # (dx, dy) in degrees – tweak as needed
    'IP':  ( 0.0,  0.0),
    'WCI': ( 0.3,  0.0),
    'NWI': ( 0.0,  0.0),
    'CNE': ( 0.4,  0.0),
    'NEI': ( 1.7,  1.2),
    'HR':  (-2.9,  0.0),
}

for key, meta in regions_meta.items():
    centroid = gdfs[key].unary_union.centroid
    dx, dy   = label_offsets[key]
    ax.text(
        centroid.x + dx, centroid.y + dy,
        key,
        ha='center', va='center',
        fontsize=9, fontweight='bold', color='white',
        path_effects=[
            pe.withStroke(linewidth=2.5, foreground='#2c2c2c')
        ],
        zorder=5,
    )

# --- lat / lon axes ---
ax.set_xlabel('Longitude (°E)', fontsize=10, labelpad=6)
ax.set_ylabel('Latitude (°N)',  fontsize=10, labelpad=6)

lon_ticks = np.arange(65, 100, 5)
lat_ticks = np.arange(5,  40,  5)
ax.set_xticks(lon_ticks)
ax.set_yticks(lat_ticks)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g°E'))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g°N'))
ax.tick_params(axis='both', labelsize=10, length=4, direction='out')
ax.grid(True, linestyle='--', linewidth=0.4, color='white', alpha=0.7, zorder=1)

# --- bounding box tight around India ---
all_geom = gpd.pd.concat(gdfs.values())
xmin, ymin, xmax, ymax = all_geom.total_bounds
pad = 1.0
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)

# --- legend ---
legend_handles = [
    Patch(facecolor=meta['color'], edgecolor='#2c2c2c',
          linewidth=0.5, alpha=0.85, label=meta['label'])
    for meta in regions_meta.values()
]
legend = ax.legend(
    handles=legend_handles,
    loc='upper right',
    fontsize=8,
    title='Homogeneous Regions',
    title_fontsize=8.5,
    framealpha=0.92,
    edgecolor='#aaaaaa',
    handlelength=1.4,
    handleheight=1.1,
)
legend.get_title().set_fontweight('bold')

# --- north arrow ---
ax.annotate(
    'N', xy=(0.95, 0.18), xycoords='axes fraction',
    ha='center', va='bottom', fontsize=11, fontweight='bold',
    xytext=(0.95, 0.13), textcoords='axes fraction',
    arrowprops=dict(arrowstyle='->', color='black', lw=1.8),
)

# --- title ---
ax.set_title(
    'Homogeneous Precipitation Regions of India',
    fontsize=12, fontweight='bold', pad=10, color='k'
)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('./figures/homo-reg-ind-pub.jpg', dpi=300, bbox_inches='tight')
plt.show()

