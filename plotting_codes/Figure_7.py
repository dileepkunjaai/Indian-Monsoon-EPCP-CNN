#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import importlib


# In[ ]:


from project_utils import parameters as param
from project_utils import fig_utils as figu
from project_utils import utils as util

importlib.reload(figu)
importlib.reload(param)
importlib.reload(util)


# In[ ]:


## set plot configurations
figu.set_plt_rc_params()


# In[ ]:


# Point this at your own data directory (or set the EPCP_DATA_DIR env var).
# Expected layout:
#   <EPCP_DATA_DIR>/precip/{region}/region_mean_precip_*.csv
#   <EPCP_DATA_DIR>/predicted_class/{region}/predicted_class_data_{region}-850hpa.csv
BASE_DIR = os.environ.get('EPCP_DATA_DIR', './data')

regions = ['IP', 'WCI', 'NWI', 'CNE', 'NEI', 'HR']

panel_label = 'a'


fig, axs = plt.subplots(nrows=len(regions), ncols=2, figsize=(12, 5*len(regions)))

for i, region in enumerate(regions):
    # Read region mean precipitation dataset
    file_dir = os.path.join(BASE_DIR, "precip", region)
    file_name = next(file for file in os.listdir(file_dir) if file.startswith("region_mean_precip") and file.endswith(".csv"))
    file_path = os.path.join(file_dir, file_name)
    precip_dat = pd.read_csv(file_path)
    precip_dat['time'] = pd.to_datetime(precip_dat['time'])

    # Open class probability dataset
    class_prob_file = os.path.join(
        BASE_DIR, "predicted_class", region, f"predicted_class_data_{region}-850hpa.csv"
    )
    class_prob = pd.read_csv(class_prob_file)
    class_prob['date'] = pd.to_datetime(class_prob['date'])

    # Select dates based on predicted classes
    ones_dates = class_prob.loc[class_prob.predicted_class == 1, 'date']
    ones_precip = precip_dat[precip_dat.time.isin(ones_dates.values)]

    # Calculate changes in class occurrence and precipitation:
    class1_count = ones_dates.groupby(ones_dates.dt.year).count().rename('count').to_frame()
    class1_precip = ones_precip.groupby(ones_precip.time.dt.year)['rain'].mean().rename('precip').reset_index()

    def calc_trends(dat, cutoffyear=2000):
        if isinstance(dat, pd.Series):
            x = dat.index
            y = dat.values
        else:
            x = dat.index
            y = dat.values.squeeze()

        early_ind = (x) < cutoffyear
        late_ind = (x >= cutoffyear)

        df = pd.DataFrame({"var": [dat.index.name]})
        df[['int', 'slope', 'pval']] = util.fit_ols(x, y)
        df[['int_early', 'slope_early', 'pval_early']] = util.fit_ols(x[early_ind], y[early_ind])
        df[['int_late', 'slope_late', 'pval_late']] = util.fit_ols(x[late_ind], y[late_ind])
        df["cutoff_year"] = cutoffyear
    
        return df


    trend_dat = calc_trends(class1_count)
    trend_dat = pd.concat([trend_dat, calc_trends(class1_precip.set_index('time')['precip'])])

    early_ind = class1_count.index <= param.early_stop
    late_ind = class1_count.index > param.early_stop
    early_plot_ind = class1_count.index <= param.early_stop + 1
    
    # Plotting
    x = class1_count.index
    y = class1_count.values.squeeze()

    axs[i, 0].plot(x[early_plot_ind], y[early_plot_ind], color=figu.base_col, marker='.', linewidth=1.4)
    axs[i, 0].plot(x[late_ind], y[late_ind], marker='.', color=figu.class1_col, linewidth=1.4)
    axs[i, 0].plot(x, x*trend_dat.iloc[0]['slope']+trend_dat.iloc[0]['int'], color='k', linewidth=1.4)
    axs[i, 0].plot(x[early_ind], x[early_ind]*trend_dat.iloc[0]['slope_early']+trend_dat.iloc[0]['int_early'], color=figu.base_col, linewidth=1.4)
    axs[i, 0].plot(x[late_ind], x[late_ind]*trend_dat.iloc[0]['slope_late']+trend_dat.iloc[0]['int_late'], color=figu.class1_col, linewidth=1.4)
    
    if trend_dat.iloc[0]['pval_late'] < 0.01:
        late_string = f'slope = {np.round(trend_dat.iloc[0]["slope_late"], 2)}\np < 0.01'
    else:
        late_string = f'slope = {np.round(trend_dat.iloc[0]["slope_late"], 2)}\np = {np.round(trend_dat.iloc[0]["pval_late"], 2)}'
    
    axs[i, 0].text(x=1, y=0.04, s=f'slope = {np.round(trend_dat.iloc[0]["slope"], 2)}, p = {np.round(trend_dat.iloc[0]["pval"], 2)}', transform=axs[i, 0].transAxes, size=18, horizontalalignment='right')
    axs[i, 0].text(x=0.05, y=0.85, s=f'slope = {np.round(trend_dat.iloc[0]["slope_early"], 2)}\np = {np.round(trend_dat.iloc[0]["pval_early"], 2)}', transform=axs[i, 0].transAxes, size=18, color=figu.base_col, horizontalalignment='left')
    axs[i, 0].text(x=1, y=0.85, s=late_string, transform=axs[i, 0].transAxes, size=18, color=figu.class1_col, horizontalalignment='right')
    axs[i, 0].set(ylim=(min(y)-11, max(y)+11), xlim=(1978, 2023))
    axs[i, 0].set_ylabel("days per season", fontsize=16)
    axs[i, 0].tick_params(axis='both', labelsize=16)
    axs[i, 0].set_title(f'{region}', fontsize=16)
    figu.format_plot(axs[i, 0])
    

    # right panels.
    x = class1_precip['time']
    y = class1_precip['precip']
    
    axs[i, 1].plot(x[early_plot_ind], y[early_plot_ind], color=figu.base_col, marker='.', linewidth=1.4)
    axs[i, 1].plot(x[late_ind], y[late_ind], marker='.', color=figu.class1_col, linewidth=1.4)
    axs[i, 1].plot(x, x*trend_dat.iloc[1]['slope']+trend_dat.iloc[1]['int'], color='k', linewidth=1.4)
    axs[i, 1].plot(x[early_ind], x[early_ind]*trend_dat.iloc[1]['slope_early']+trend_dat.iloc[1]['int_early'], color=figu.base_col, linewidth=1.4)
    axs[i, 1].plot(x[late_ind], x[late_ind]*trend_dat.iloc[1]['slope_late']+trend_dat.iloc[1]['int_late'], color=figu.class1_col, linewidth=1.4)
    
    axs[i, 1].text(x=1, y=0.04, s=f'slope = {np.round(trend_dat.iloc[1]["slope"], 2)}\np = {np.round(trend_dat.iloc[1]["pval"], 3)}', transform=axs[i, 1].transAxes, size=18, horizontalalignment='right')
    axs[i, 1].text(x=0.05, y=0.85, s=f'slope = {np.round(trend_dat.iloc[1]["slope_early"], 3)}\np = {np.round(trend_dat.iloc[1]["pval_early"], 2)}', transform=axs[i, 1].transAxes, size=18, color=figu.base_col, horizontalalignment='left')
    axs[i, 1].text(x=1, y=0.85, s=f'slope = {np.round(trend_dat.iloc[1]["slope_late"], 2)}\np = {np.round(trend_dat.iloc[1]["pval_late"], 3)}', transform=axs[i, 1].transAxes, size=18, color=figu.class1_col, horizontalalignment='right')
    axs[i, 1].set(ylim=(min(y)-2, max(y)+2), xlim=(1978, 2023))
    axs[i, 1].set_ylabel("precip. (mm/day)", fontsize=16)
    axs[i, 1].tick_params(axis='both', labelsize=16)
    axs[i, 1].set_title(f'{region}', fontsize=16)

    figu.format_plot(axs[i, 1])
    
    # Add panel labels
    axs[i, 0].text(-0.06, 1.1, f"({panel_label})", transform=axs[i, 0].transAxes, fontsize=18, weight='bold', va='top')
    axs[i, 1].text(-0.05, 1.09, f"({chr(ord(panel_label) + 1)})", transform=axs[i, 1].transAxes, fontsize=18, weight='bold', va='top')
    

    # Update panel label
    panel_label = chr(ord(panel_label) + 2)
    
# titles above the columns
fig.text(0.3, 0.89, "EPCP occurrences per year", fontsize=16, weight='bold', ha='center')
fig.text(0.72, 0.89, "EPCP mean precipitation", fontsize=16, weight='bold', ha='center')

os.makedirs('figures', exist_ok=True)
fig.savefig("./figures/epcp-freq-inten-850hpa-allreg.jpg", dpi=400)

plt.show()

