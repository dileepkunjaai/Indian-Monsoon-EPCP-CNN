import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from project_utils import parameters as param

datacrs = ccrs.PlateCarree()
mapcrs = ccrs.PlateCarree(central_longitude=80)
bounds = [40, 125, -3, 34]
small_bounds = [60, 100, 0, 30]

## Set colors -------------------------------------------------------------
train_col = 'navy'
val_col = 'cornflowerblue'
class1_col = 'navy' 
epcp90_col = '#057A71' #(72/255, 74/255, 154/255)
epcp75_col = '#244EA1' #(41/255, 44/255, 152/255)
p90_col = '#427ECA'
p99_col = '#188195'
class0_col = 'tan'
base_col = 'grey'
midwest_col = 'darkgreen'
region_col = 'green'
fig2c_cmap = "twilight_shifted"
fig2c_col = (68/255, 2/255, 86/255)

from matplotlib.colors import LinearSegmentedColormap

brownpurple = np.concatenate([plt.cm.get_cmap("BrBG")([0, 0.1, 0.2, 0.3, 0.4]), 
plt.cm.get_cmap("Purples")([0, 0.3, 0.5, 0.65, 0.83, 1])])

brownpurple = LinearSegmentedColormap.from_list('brownpurple', brownpurple, N=15)
purples = LinearSegmentedColormap.from_list('purples', plt.cm.get_cmap("Purples")(np.arange(0, 1.1, .1)), N = 10)



brown = plt.cm.get_cmap("BrBG")(np.linspace(0, 0.5, 128))
green = plt.cm.get_cmap("Greens")(np.linspace(0.2, 1, 128))

browngreen = np.vstack((brown, green))
browngreen = LinearSegmentedColormap.from_list('browngreen', browngreen, N=256)



whitegreen = np.concatenate([[[1,1,1,1]], plt.cm.get_cmap("Greens")(np.arange(0, 1.1, .1))])
whitegreen = LinearSegmentedColormap.from_list('whitegreen', whitegreen, N = 11)

whitered = np.concatenate([[[1,1,1,1]], plt.cm.get_cmap("Reds")(np.arange(0, 1.1, .1))])
whitered = LinearSegmentedColormap.from_list('whitered', whitered, N = 11)

bluered = LinearSegmentedColormap.from_list('bluered', plt.cm.get_cmap("RdBu_r")(np.arange(0, 1.1, 0.1)), N=10)
browngreen = LinearSegmentedColormap.from_list('browngreen', plt.cm.get_cmap("BrBG")(np.arange(0, 1.1, 0.1)), N=12)
purpleorange = LinearSegmentedColormap.from_list('purpleorange', plt.cm.get_cmap("PuOr_r")(np.arange(0, 1.1, 0.1)), N=11)

reds = LinearSegmentedColormap.from_list('reds', plt.cm.get_cmap("Reds")(np.arange(0, 1.1, .1)), N = 10)
oranges = LinearSegmentedColormap.from_list('oranges', plt.cm.get_cmap("Oranges")(np.arange(0, 1.1, .1)), N = 10)
orangered = LinearSegmentedColormap.from_list('orangered', plt.cm.get_cmap("OrRd")(np.arange(0, 1.1, 0.1)), N=10)


##-------------------------------------------------------------------------------
def set_plt_rc_params():
    plt.rcParams['figure.dpi'] = 120
    plt.rc('axes', titlesize=8)     # fontsize of the axes titles (i.e. title of each panel)
    plt.rc('axes', labelsize=7)    # fontsize of the x and y axis labels
    plt.rc('xtick', labelsize=7)    # fontsize of the x tick labels
    plt.rc('ytick', labelsize=7)    # fontsize of the y tick labels
    plt.rc('figure', titlesize = 8)
    plt.rc('legend', fontsize=7)
    plt.rc('legend', title_fontsize=7)
    plt.rc('lines', linewidth=1)

def format_map(ax, border = 'off', bounds = bounds):
    """format map with states, coastlines, and set extent"""
    ax.set_extent(bounds)
    ax.axis(border)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth = 0.5)
    ax.add_feature(cfeature.STATES.with_scale('110m'), linewidth = 0.5)

def format_map_platecarree(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth = 0.4)
    ax.set_extent([40, 125, 0, 30])
    return(ax)

def format_plot(ax):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def add_panel_label(ax, label, x = -0.1, y = 1.15, fontsize=9):
    ax.text(x, y, label, transform=ax.transAxes,
      fontsize=fontsize, fontweight='bold', va='top', ha='right')

def add_square_bracket(fig, ax, xmin, xmax, ymin, ymax, s, vertical=True, text_offset = 0.01):
    if vertical:
        ax.hlines([ymin, ymax], xmin = xmin, xmax = xmax, transform = fig.transFigure,
                color = "k", clip_on = False)
        ax.vlines([xmax], ymin = ymin, ymax = ymax, transform = fig.transFigure, color = 'k', clip_on = False)
        ax.text(xmax + text_offset, (ymin + ymax)/2, s, size = 8, transform = fig.transFigure, va = "center", ha = "center",
               rotation = 270)
