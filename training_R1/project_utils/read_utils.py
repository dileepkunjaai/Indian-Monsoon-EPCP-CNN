import numpy as np
import pandas as pd
import xarray as xr
from project_utils import parameters as param

# def get_hgt_input():
#     """ create np.ndarray of hgt data from pre-processed files
#     
#     returns = numpy ndarray with dimmensions of [time, lat, lon, vars]
#     """
#     
#     hgt_xr = xr.open_dataset("/media/amal/OneTouch/1-jjas-epcp-cnn/commondata/hgt_anomalies_850hpa-jjas.nc")
#     dat_np = np.array(hgt_xr['hgt_anom']).reshape(param.n, param.nlats, param.nlons, 1) #!
#     
#     return(dat_np)

def get_hgt_slp_input(selcted_level):
    """ 
    create np.ndarray of hgt and slp data from pre-processed files 
    returns = numpy ndarray with dimmensions of [time, lat, lon, vars]
    """
    #there was a level cordinate.. so had to read as array.
    hgt_file_out_anomaly="/mnt/Data1/AmalWork/WIth_threeVariable_ERA5/ERA5_Anomaly/anomaly/era5_"+str(selcted_level)+"_geopotential_1979_to_2022_daily_JJAS_anomaly.nc"
    hgt_xr = xr.open_dataset(hgt_file_out_anomaly)['z']

    slp_xr = xr.open_dataset("/mnt/Data1/AmalWork/WIth_threeVariable_ERA5/ERA5_Anomaly/anomaly/era5_mslp_1979_to_2022_dailyMean_JJAS_anomaly.nc")['msl']
    dat_np = np.concatenate([np.array(hgt_xr).reshape(param.n, param.nlats, param.nlons, 1), 
                             np.array(slp_xr).reshape(param.n, param.nlats, param.nlons, 1)], 
                            axis = 3)
    return(dat_np)

def get_uwnd_vwnd_slp_input(selcted_level):
    """ create np.ndarray of hgt and slp data from pre-processed files 
    
    returns = numpy ndarray with dimmensions of [time, lat, lon, vars]
    """
    #there was a level cordinate.. so had to read as array.
    uwnd_file_out_anom = "/mnt/Data1/AmalWork/WIth_threeVariable_ERA5/ERA5_Anomaly/anomaly/era5_"+str(selcted_level)+"_U_1979_to_2022_daily_JJAS_anomaly.nc"
    uwnd_xr = xr.open_dataset(uwnd_file_out_anom)['u']#.drop_vars("level")
    vwnd_file_out_anom = "/mnt/Data1/AmalWork/WIth_threeVariable_ERA5/ERA5_Anomaly/anomaly/era5_"+str(selcted_level)+"_V_1979_to_2022_dailyMean_JJAS_anomaly.nc"
    vwnd_xr = xr.open_dataset(vwnd_file_out_anom)['v']#.drop_vars("level")
    slp_xr = xr.open_dataset("/mnt/Data1/AmalWork/WIth_threeVariable_ERA5/ERA5_Anomaly/anomaly/era5_mslp_1979_to_2022_dailyMean_JJAS_anomaly.nc")['msl']
    dat_np = np.concatenate([np.array(uwnd_xr).reshape(param.n, param.nlats, param.nlons, 1), np.array(vwnd_xr).reshape(param.n, param.nlats, param.nlons, 1),np.array(slp_xr).reshape(param.n, param.nlats, param.nlons, 1)], axis = 3)
    print(dat_np.shape)
    return(dat_np)

def get_hgt_slp_shum_input(selcted_level):
    """ 
    create np.ndarray of hgt and slp data from pre-processed files 
    returns = numpy ndarray with dimmensions of [time, lat, lon, vars]
    """
    #there was a level cordinate.. so had to read as array.
    hgt_file_out_anomaly="/mnt/Data1/AmalWork/WIth_threeVariable_ERA5/ERA5_Anomaly/anomaly/era5_"+str(selcted_level)+"_geopotential_1979_to_2022_daily_JJAS_anomaly.nc"
    hgt_xr = xr.open_dataarray(hgt_file_out_anomaly)

    shum_file_out_anom = "/mnt/Data1/AmalWork/WIth_threeVariable_ERA5/ERA5_Anomaly/anomaly/era5_"+str(selcted_level)+"_specifichumidity_1979_to_2022_daily_JJAS_anomaly.nc"
    shum_xr = xr.open_dataarray(shum_file_out_anom)#.drop_vars("level")

    slp_xr = xr.open_dataset("/mnt/Data1/AmalWork/WIth_threeVariable_ERA5/ERA5_Anomaly/anomaly/era5_mslp_1979_to_2022_dailyMean_JJAS_anomaly.nc")

    dat_np = np.concatenate([np.array(hgt_xr).reshape(param.n, param.nlats, param.nlons, 1), np.array(shum_xr).reshape(param.n, param.nlats, param.nlons, 1),
                         np.array(slp_xr['msl']).reshape(param.n, param.nlats, param.nlons, 1)], 
                        axis = 3)
    print(dat_np.shape)
    return(dat_np)
