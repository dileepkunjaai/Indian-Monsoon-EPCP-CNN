#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import mannwhitneyu
from glob import glob
import os

# ── regions and paths ──────────────────────────────────────────────────────────
regions   = ['IP', 'WCI', 'NWI', 'CNE', 'NEI', 'HR']

# Point this at your own data directory (or set the EPCP_DATA_DIR env var).
# Expected layout:
#   <EPCP_DATA_DIR>/moisture_flux.nc
#   <EPCP_DATA_DIR>/predicted_class/{region}/predicted_class_data_{region}-850hpa.csv
BASE_DIR = os.environ.get("EPCP_DATA_DIR", "./data")

MFLUX_PATH = os.path.join(BASE_DIR, "moisture_flux.nc")

CSV_TMPL = os.path.join(
    BASE_DIR, "predicted_class", "{region}", "predicted_class_data_{region}-850hpa.csv"
)

# ── domain ─────────────────────────────────────────────────────────────────────
LAT_SLICE  = slice(40, 0)
LON_SLICE  = slice(60, 100)

# ── period year ranges (inclusive) ─────────────────────────────────────────────
EARLY = (1979, 1999)
LATE  = (2000, 2022)


# In[ ]:


mflux_full = xr.open_dataset(MFLUX_PATH).sel(
    lat=LAT_SLICE, lon=LON_SLICE
)
print(mflux_full)


# In[ ]:


def sel_years(ds, year_range):
    """Subset an xarray Dataset to a closed year range along 'time'."""
    y0, y1 = year_range
    return ds.sel(time=(ds.time.dt.year >= y0) & (ds.time.dt.year <= y1))


# In[ ]:


def _mw_pval(x, y):
    """Scalar MW p-value; NaNs dropped before test."""
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 3 or len(y) < 3:          # guard against empty slices
        return np.nan
    return mannwhitneyu(x, y, alternative='two-sided').pvalue


def mannwhitneyu_xr(da_early, da_late):
    """
    Vectorised grid-cell-wise Mann-Whitney U test between two DataArrays
    that may have different time lengths.
    Returns a DataArray of p-values (lat × lon).
    """
    return xr.apply_ufunc(
        _mw_pval,
        da_early, da_late,
        input_core_dims=[["time"], ["time"]],
        vectorize=True,
        join="outer",
        dataset_join="outer",
        dataset_fill_value=np.nan,
        dask="parallelized",
        output_dtypes=[float],
    )


# In[ ]:


region_stats = {}   # keyed by region name

for reg in regions:
    print(f"Processing {reg} …", flush=True)

    # ── 1. load predicted classes and filter class==1 dates ───────────────────
    csv_path   = CSV_TMPL.format(region=reg)
    class_prob = pd.read_csv(csv_path)
    class_prob['date'] = pd.to_datetime(class_prob['date'])

    ones_dates = class_prob.loc[class_prob['predicted_class'] == 1, 'date'].values

    # ── 2. select moisture flux on ones dates ─────────────────────────────────
    mflux1 = mflux_full.sel(time=ones_dates)   # may drop dates not in dataset

    # ── 3. split into early / late ────────────────────────────────────────────
    mf_early = sel_years(mflux1, EARLY)
    mf_late  = sel_years(mflux1, LATE)

    # ── 4. period means ───────────────────────────────────────────────────────
    mean_early = mf_early.mean(dim='time')
    mean_late  = mf_late.mean(dim='time')
    mean_diff  = mean_late - mean_early           # recent − early

    # ── 5. Mann-Whitney on magnitude (ones dates only, early vs late) ─────────
    pval_mag = mannwhitneyu_xr(mf_early['mag'], mf_late['mag'])

    # non-significant mask: True where NOT significant (p >= 0.05)
    nonsig_mask = xr.ones_like(pval_mag).where(pval_mag >= 0.05)

    region_stats[reg] = dict(
        mean_early=mean_early,
        mean_late=mean_late,
        mean_diff=mean_diff,
        nonsig_mask=nonsig_mask,
    )

print("Done.")


# In[ ]:


import cmocean


# In[ ]:


import matplotlib as mpl
import string
labels = list(string.ascii_lowercase)

# ── colour / vector settings ───────────────────────────────────────────────────
MAG_CMAP      = 'cmo.ice_r'
DIFF_CMAP     = 'RdBu'
QUIVER_SCALE  = 3000
QUIVER_STEP   = 2
HATCH_PATTERN = '..'
HATCH_LW      = 0.3
mpl.rcParams['hatch.linewidth'] = HATCH_LW

proj   = ccrs.PlateCarree()
extent = [60, 100, 0, 40]

col_titles = ['Early (1979–1999)', 'Recent (2000–2022)', 'Difference (Recent − Early)']

# ── global colour limits ───────────────────────────────────────────────────────
all_mag_vals = np.concatenate([
    np.concatenate([
        region_stats[r]['mean_early']['mag'].values.ravel(),
        region_stats[r]['mean_late']['mag'].values.ravel()
    ]) for r in regions
])
mag_vmin = np.nanpercentile(all_mag_vals, 2)
mag_vmax = np.nanpercentile(all_mag_vals, 98)

all_diff_vals = np.concatenate([
    region_stats[r]['mean_diff']['mag'].values.ravel() for r in regions
])
diff_abs       = np.nanpercentile(np.abs(all_diff_vals), 98)
diff_vmin, diff_vmax = -diff_abs, diff_abs

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4.0 * len(regions)))

gs = fig.add_gridspec(
    nrows=len(regions), ncols=3,
    hspace=0.06, wspace=0.04,
    left=0.09, right=0.97,
    top=0.94, bottom=0.08
)

axes = np.array([
    [fig.add_subplot(gs[r, c], projection=proj) for c in range(3)]
    for r in range(len(regions))
])

for row_i, reg in enumerate(regions):
    st       = region_stats[reg]
    datasets = [st['mean_early'], st['mean_late'], st['mean_diff']]
    nonsig   = st['nonsig_mask']

    for col_j, ds in enumerate(datasets):
        ax   = axes[row_i, col_j]
        lats = ds['lat'].values
        lons = ds['lon'].values
        mag  = ds['mag'].values
        u    = ds['u_dir'].values
        v    = ds['v_dir'].values

        # ── filled contour ────────────────────────────────────────────────────
        if col_j < 2:
            ax.contourf(lons, lats, mag,
                        levels=20, vmin=mag_vmin, vmax=mag_vmax,
                        cmap=MAG_CMAP, transform=proj, extend='both')
        else:
            ax.contourf(lons, lats, mag,
                        levels=20, vmin=diff_vmin, vmax=diff_vmax,
                        cmap=DIFF_CMAP, transform=proj, extend='both')

            ns_vals = nonsig.values
            if not np.all(np.isnan(ns_vals)):
                ax.contourf(lons, lats, ns_vals,
                            levels=[0.5, 1.5],
                            hatches=[HATCH_PATTERN],
                            colors='none', alpha=0.0,
                            transform=proj)

        # ── quiver ────────────────────────────────────────────────────────────
        qs = QUIVER_STEP
        lons2d, lats2d = np.meshgrid(lons, lats)
        qv = ax.quiver(lons2d[::qs, ::qs], lats2d[::qs, ::qs],
                  u[::qs, ::qs], v[::qs, ::qs],
                  color='black', transform=proj,
                  scale=QUIVER_SCALE, width=0.006,
                  headwidth=3, headlength=4,
                  headaxislength=3.5, alpha=0.75)
        if row_i == 0 and col_j == 0:
            # Adjust U (magnitude) and label according to your typical flux values
            ax.quiverkey(qv, X=0.85, Y=1.07, U=200,
                         label='200 kg m⁻¹ s⁻¹', labelpos='E',
                         coordinates='axes', fontproperties={'size': 12})

        # ── map features ──────────────────────────────────────────────────────
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle='--')

        # ── gridlines: lat labels on col 0, lon labels on last row ───────────
        is_last_row  = (row_i == len(regions) - 1)
        is_first_col = (col_j == 0)

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.25, color='gray', alpha=0.4, linestyle=':'
        )
        gl.xlocator = mticker.FixedLocator(range(60, 101, 5))
        gl.ylocator = mticker.FixedLocator(range(0,  31,  5))

        # suppress all labels by default, then selectively enable
        gl.top_labels    = False
        gl.right_labels  = False
        gl.bottom_labels = is_last_row    # lon ticks only on row 6
        gl.left_labels   = is_first_col  # lat ticks only on col 1

        gl.xlabel_style = {'size': 13}
        gl.ylabel_style = {'size': 13}

        # ── column titles (top row only) ──────────────────────────────────────
        if row_i == 0:
            ax.set_title(col_titles[col_j], fontsize=15,
                         fontweight='bold', pad=5)

        # ── region label ──────────────────────────────────────────────────────
        if col_j == 0:
            ax.text(-0.18, 0.5, reg, va='center', ha='right',
                    transform=ax.transAxes,
                    fontsize=13, fontweight='bold')

    for row_i in range(len(regions)):
        for col_j in range(3):
            ax    = axes[row_i, col_j]
            label = labels[row_i * 3 + col_j]   # a=0, b=1, c=2, d=3 ...
            ax.text(
                0.02, 0.97,               # top-left corner, just inside the axes
                f'({label})',
                transform=ax.transAxes,
                fontsize=13,
                fontweight='bold',
                va='top', ha='left',
                color='black',            # white text so it shows on dark colormap
                bbox=dict(
                    boxstyle='round,pad=0.15',
                    facecolor='white',
                    alpha=0.45,           # semi-transparent dark box for contrast
                    edgecolor='none'
                )
            )

# ── colorbars ─────────────────────────────────────────────────────────────────
# Read axes positions AFTER drawing (positions are finalised at render time)
fig.canvas.draw()   # force layout computation so get_position() is accurate

col0_left   = axes[-1, 0].get_position().x0
col1_right  = axes[-1, 1].get_position().x1
col2_left   = axes[-1, 2].get_position().x0
col2_right  = axes[-1, 2].get_position().x1
axes_bottom = axes[-1, 0].get_position().y0

CB_HEIGHT = 0.013
CB_PAD    = 0.018   # ← reduced from 0.035 to pull bars closer

# shorten mag colorbar: indent by 12% of its total width on each side
mag_bar_width  = col1_right - col0_left
mag_bar_indent = mag_bar_width * 0.12
cbar_ax1 = fig.add_axes([
    col0_left  + mag_bar_indent,
    axes_bottom - CB_PAD - CB_HEIGHT,
    mag_bar_width - 2 * mag_bar_indent,
    CB_HEIGHT
])
sm1 = plt.cm.ScalarMappable(cmap=MAG_CMAP,
                             norm=plt.Normalize(mag_vmin, mag_vmax))
sm1.set_array([])
cb1 = fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal', extend='both')
cb1.set_label('VIMF Magnitude (kg m⁻¹ s⁻¹)', fontsize=13)
cb1.ax.tick_params(labelsize=13)

# diff colorbar: full width of col 3
cbar_ax2 = fig.add_axes([
    col2_left,
    axes_bottom - CB_PAD - CB_HEIGHT,
    col2_right - col2_left,
    CB_HEIGHT
])
sm2 = plt.cm.ScalarMappable(cmap=DIFF_CMAP,
                             norm=plt.Normalize(diff_vmin, diff_vmax))
sm2.set_array([])
cb2 = fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal', extend='both')
cb2.set_label('Difference (kg m⁻¹ s⁻¹)', fontsize=13)
cb2.ax.tick_params(labelsize=13)

# ── suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    'Vertically Integrated Moisture Flux on EPCP Days\n'
    'Early vs. Recent Period',
    fontsize=18, fontweight='bold', y=0.98
)

os.makedirs('figures', exist_ok=True)
plt.savefig('./figures/mflux_6regions_3panel.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → mflux_6regions_3panel.png")

