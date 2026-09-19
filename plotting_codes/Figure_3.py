#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import importlib


# In[ ]:


from project_utils import parameters as param
from project_utils import utils as util
from project_utils import fig_utils as figu
importlib.reload(param)
importlib.reload(util)
importlib.reload(figu)


# In[ ]:


figu.set_plt_rc_params()


# In[ ]:


# Point this at your own data directory (or set the EPCP_DATA_DIR env var).
# Expected layout: <EPCP_DATA_DIR>/precip/{region}/region_mean_precip_*.csv
BASE_DIR = os.environ.get('EPCP_DATA_DIR', './data')
PRECIP_BASE = os.path.join(BASE_DIR, 'precip')

region_names = ['IP', 'WCI', 'NWI', 'CNE', 'NEI', 'HR']

def find_precip_csv(region):
    """Auto-detect the region's precip CSV (filenames aren't uniform)."""
    region_dir = os.path.join(PRECIP_BASE, region)
    fname = next(f for f in os.listdir(region_dir)
                 if f.startswith('region_mean_precip') and f.endswith('.csv'))
    return os.path.join(region_dir, fname)

data_paths = [(find_precip_csv(r), r) for r in region_names]

# Function to create a single plot for a given data file and region name
def create_subplot(ax, data_path, region_name, label="", ylabel=False):  # Added ylabel parameter
    precip_dat = pd.read_csv(data_path)
    precip_dat['time'] = pd.to_datetime(precip_dat['time'])
    
    pr_thr = precip_dat["rain"].quantile(0.95)
    precip_dat["extreme"] = 0
    precip_dat.loc[precip_dat.rain > pr_thr, 'extreme'] = 1

    extreme_count = precip_dat.groupby(precip_dat.time.dt.year)['extreme'].sum().rename_axis('year')

    early_ind = extreme_count.index <= param.early_stop
    early_plot_ind = extreme_count.index <= param.early_stop + 1
    late_ind = extreme_count.index > param.early_stop

    x = extreme_count.index
    y = extreme_count.values
    ex_int, ex_slope, ex_pval = util.fit_ols(x, y)
    ex_int_early, ex_slope_early, ex_pval_early = util.fit_ols(x[early_ind], y[early_ind])
    ex_int_late, ex_slope_late, ex_pval_late = util.fit_ols(x[late_ind], y[late_ind])

    ax.plot(x[early_plot_ind], y[early_plot_ind], color='b', marker='.', linewidth=2, markersize=9)
    ax.plot(x[late_ind], y[late_ind], marker='.', color='g', linewidth=2, markersize=9)

    ax.plot(x, x * ex_slope + ex_int, color='k', linewidth=2)
    ax.plot(x[early_ind], x[early_ind] * ex_slope_early + ex_int_early, color='b', linewidth=2)
    ax.plot(x[late_ind], x[late_ind] * ex_slope_late + ex_int_late, color='g', linewidth=2)

    ax.text(0.98, 0.94, 'slope = {:.2f}, p = {:.3f}'.format(ex_slope, ex_pval),
            transform=ax.transAxes, size=16, horizontalalignment='right')  # Increase font size
    ax.text(0.02, 0.85, 'slope = {:.2f}\np = {:.2f}'.format(ex_slope_early, ex_pval_early),
            transform=ax.transAxes, size=16, color='b', horizontalalignment='left')  # Increase font size
    ax.text(0.98, 0.76, 'slope = {:.2f}\np = {:.3f}'.format(ex_slope_late, ex_pval_late),
            transform=ax.transAxes, size=16, color='g', horizontalalignment='right')  # Increase font size

    if ylabel:  # Conditionally set ylabel
        ax.set_ylabel("days season$^{-1}$", size=18)  # Increase font size
    
    ax.set(title=region_name, ylim=(-2, 24), xlim=(1978, 2023))
    ax.tick_params(axis='both', labelsize=18)  # Increase font size for axis ticks
    ax.set_xlabel(ax.get_xlabel(), size=18)  # Increase font size for axis label
    ax.title.set_size(18)  # Increase font size for subplot title

# Create a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

# Iterate over data files and plot on subplots
for (data_path, region_name), ax in zip(data_paths, axes.ravel()):
    # Determine the label for the leftmost panels
    label = '(a)' if ax in axes[0, :] else '(d)' if ax in axes[1, :] else ''
    create_subplot(ax, data_path, region_name, label=label, ylabel=ax in axes[:, 0])  # Pass ylabel parameter

# Show y-axis only for leftmost panels ('a' and 'd')
for ax in axes[:, 1:].ravel():
    ax.set_yticklabels([])

# Add subplot labels
for i, ax in enumerate(axes.ravel()):
    ax.text(-0.1, 1.1, f'({chr(ord("a") + i)})', transform=ax.transAxes, size=20, weight='bold', ha='center')

# Adjust subplot spacing
plt.tight_layout()

# Save the figure
os.makedirs('figures', exist_ok=True)
fig.savefig("./figures/obs-trend_multi_panel_plot.jpg", dpi=300)

