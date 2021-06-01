# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:10:13 2021

@author: kkrao
"""


import pandas as pd
import gdal
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale = 0.6, style = "ticks")


def get_value(filename, mx, my, band = 1):
        ds = gdal.Open(filename)
        gt = ds.GetGeoTransform()
        data = ds.GetRasterBand(band).ReadAsArray()
        px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
        py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
        ds = None
        return data[py,px]
    
def get_lats_lons(array):
    x = range(data['pws'].shape[1])
    y = range(data['pws'].shape[0])
    
    x,y = np.meshgrid(x,y)
    
    lons = x*gt[1]+gt[0]
    lats = y*gt[5]+gt[3]
    
    return lats, lons

def create_df(array,keys):
    df = pd.DataFrame()
    ctr=0
    for key in keys:
        df[key] = array[ctr].flatten()
        ctr+=1
    return df

#%% create dataset
# data = dict()
# ds = gdal.Open(os.path.join("D:/Krishna/projects/wildfire_from_lfmc", "data","arr_pixels_lfmc_dfmc_anomalies","PAS_6_jan_2021.tif"))
# gt = ds.GetGeoTransform()
# data['pws'] = np.array(ds.GetRasterBand(1).ReadAsArray())

# lats, lons = get_lats_lons(data['pws'])

# keys = ['pws','elevation','aspect','slope','twi','silt','sand','clay',\
#         'dry_season_length','ndvi','vpd_mean','vpd_std','isohydricity','root_depth','canopy_height']

# array = np.zeros((len(keys), data['pws'].shape[0],data['pws'].shape[1])).astype('float')
# array[0] = data['pws']


# array[1] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/Elevation/usa_dem.tif",lons,lats)
# # array[2] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/ASPECT/usa_aspect_wgs1984.tif",lons,lats)
# array[3] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/Elevation/usa_slope_project.tif",lons,lats)
# array[4]= get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/twi/twi.tif",lons,lats)


# array[5]= get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242/data/Unified_NA_Soil_Map_Subsoil_Silt_Fraction.tif",lons,lats)
# array[6]= get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242/data/Unified_NA_Soil_Map_Subsoil_Sand_Fraction.tif",lons,lats)
# array[7]= get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242/data/Unified_NA_Soil_Map_Subsoil_Clay_Fraction.tif",lons,lats)

# band = 1
# ds = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/fireSeasonLength.tif")
# array[8]= ds.GetRasterBand(band).ReadAsArray()
# ds =  gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/ndvi_mean.tif")
# array[9] =ds.GetRasterBand(band).ReadAsArray()
# ds = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/vpd_mean.tif")
# array[10] = ds.GetRasterBand(band).ReadAsArray()
# ds = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/vpdStd.tif")
# array[11]= ds.GetRasterBand(band).ReadAsArray()
# ds = None

# array[12]= get_value("D:/Krishna/projects/wildfire_from_lfmc/data/traits/isohydricity.tif",lons,lats)
# array[13] = get_value("D:/Krishna/projects/wildfire_from_lfmc/data/traits/root_depth.tif",lons,lats)
# array[14] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/canopy_height/canopy_height.tif",lons,lats)

# df = create_df(array,keys)
# df.dropna(subset = ["pws"], inplace = True)
# df.head()
# store = pd.HDFStore('D:/Krishna/projects/pws_drivers/data/store.h5')
# store['df'] = df
# store.close()

#%%load dataset
store = pd.HDFStore('D:/Krishna/projects/pws_drivers/data/store.h5')
df =  store['df']   # save it
store.close()

df.drop('aspect',axis = 1,inplace = True)

# sample = df.sample(1e4)
# sns.pairplot(sample)
df.loc[df['elevation']<-1e3] = np.nan
df.loc[df['slope']<-1e3] = np.nan
df.loc[df['twi']>1e4] = np.nan
df.loc[df['silt']<-1] = np.nan
df.loc[df['sand']<-1] = np.nan
df.loc[df['clay']<-1] = np.nan
df.loc[df['isohydricity']>1e3] = np.nan
df.loc[df['root_depth']<-1] = np.nan

sns.heatmap(df.corr(),vmin = -0.5, vmax = 0.5, cmap = sns.diverging_palette(240, 10, n=8))

# for col in df.columns:
#     fig, ax= plt.subplots()
#     # df[col].hist()
#     # ax.set_xlabel(col)
#     sns.regplot(x=col, y="pws", data=df.sample(int(1e4)),ci = None,
#                   scatter_kws={"s": 0.1,'alpha':0.5},line_kws = {"linewidth":0},ax=ax)
#     ax.set_xlim(df[col].quantile(0.05),df[col].quantile(0.95))
#     plt.show()
# data['elevation'] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/Elevation/usa_dem.tif",lons,lats)
# data['aspect'] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/ASPECT/usa_aspect_wgs1984.tif",lons,lats)
# data['slope'] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/Elevation/usa_slope_project.tif",lons,lats)
# data['twi'] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/twi/twi.tif",lons,lats)


# data['silt'] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242/data/Unified_NA_Soil_Map_Subsoil_Silt_Fraction.tif",lons,lats)
# data['sand'] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242/data/Unified_NA_Soil_Map_Subsoil_Sand_Fraction.tif",lons,lats)
# data['clay'] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242/data/Unified_NA_Soil_Map_Subsoil_Clay_Fraction.tif",lons,lats)

# band = 1
# data['dry_season_length'] = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/fireSeasonLength.tif").GetRasterBand(band).ReadAsArray()
# data['ndvi'] = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/ndvi.tif").GetRasterBand(band).ReadAsArray()
# data['vpd_mean'] = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/vpd_mean.tif").GetRasterBand(band).ReadAsArray()
# data['vpd_std'] = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/vpdStd.tif").GetRasterBand(band).ReadAsArray()

# data['isohydricity'] = get_value("D:/Krishna/projects/wildfire_from_lfmc/data/traits/isohydricity.tif",lons,lats)
# data['root_depth'] = get_value("D:/Krishna/projects/wildfire_from_lfmc/data/traits/root_depth.tif",lons,lats)
# data['canopy_height'] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/canopy_height/canopy_height.tif",lons,lats)



# plt.imshow(data['aspect'].astype(int),vmin = 0, vmax = 360)
