# #### for jjas

import numpy as np
import xarray as xr

## general analysis
time_period = slice("1979-01-01", "2022-12-31") # for climatology calculations
time_period_15 = slice("1979-01-01", "2015-12-31")
early_years = slice("1979-01-01", "1999-12-31")
early_stop = 1999
late_years = slice("2000-01-01", "2022-12-31")

dates = xr.open_dataset("/media/amal/CMIP6_tos_5TB/Amal_CNN_WORK/1-jjas-epcp-cnn/commondata/jjas_dates-22.nc").reset_coords().time
n = 5368 ## number of days 

mm2inch = 0.0393701 # !
g = 9.80665
mflux_levels = slice(1000, 300)
hgt_level = 850

# region
## reanalysis region india
lat_bbox = slice(40, 0)
lon_bbox = slice(40, 125)

lats = np.array([40. ,  37.5,  35. ,  32.5,  30. ,  27.5,  25. ,  22.5,  20. ,  17.5,
                 15. ,  12.5,  10. ,   7.5,   5. ,   2.5,   0.]) 
nlats = len(lats)


lons = np.array([40. ,  42.5,  45. ,  47.5, 50. ,  52.5,  55. ,  57.5,  60. ,  62.5,  65. ,  67.5,  70. ,  72.5,
        75. ,  77.5,  80. ,  82.5,  85. ,  87.5,  90. ,  92.5,  95. ,  97.5,
       100. , 102.5, 105. , 107.5, 110. , 112.5, 115. , 117.5, 120., 122.5,
       125.])
nlons = len(lons)
