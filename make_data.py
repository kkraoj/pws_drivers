# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:10:13 2021

@author: kkrao

Create dataframe of with input (features) and output (pws)
Shape of dataframe = (# of pixels, # of features + 1)
"""

import os

import pandas as pd
import gdal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import dirs
sns.set(font_scale = 0.6, style = "ticks")


def get_value(filename, mx, my, band = 1):
    """
    Vector implementation of query of raster value at lats and lons

    Parameters
    ----------
    filename : raster path
    mx : Lon values (list or array)
    my : lat values (list or array)
    band : band position to query. int, optional. The default is 1.

    Returns
    -------
    1D array of value of raster at lats and lons

    """
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(band).ReadAsArray()
    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
    ds = None
    return data[py,px]
    
def get_lats_lons(data, gt):
    """
    Fetch list of lats and lons corresponding to geotransform and 2D numpy array

    Parameters
    ----------
    data : 2D numpy array. This is the array whose shape will be used to
            generate the list of lats and lons
    gt : 6 size tuple of gdal geotransform

    Returns
    -------
    lats : array of latitudes
    lons : array of longitudes

    """
    x = range(data.shape[1])
    y = range(data.shape[0])
    
    x,y = np.meshgrid(x,y)
    
    lons = x*gt[1]+gt[0]
    lats = y*gt[5]+gt[3]
    
    return lats, lons

def create_df(array,keys):
    """
    Create a dataframe with a 3D numpy matrix.
    Each slice of the matrix (in 3rd dimension) is flattened into a linear
    vector and appended as a column to the dataframe.
    
    Parameters
    ----------
    array : 3d matrix of shape (rows, cols, features)
    keys : array of strings associated with each feature. This will be the 
            column name

    Returns
    -------
    df : pandas dataframe

    """
    df = pd.DataFrame()
    ctr=0
    for key in keys:
        df[key] = array[ctr].flatten()
        ctr+=1
    return df

def create_h5():
    """
    Bring together all features and labels (pws). drop nans. Store it as a h5
    file. And return it as a dataframe of shape (# of examples, # of features + 1)
    Returns
    -------
    df : pandas dataframe

    """
    data = dict()
    ds = gdal.Open(os.path.join("D:/Krishna/projects/wildfire_from_lfmc", "data","arr_pixels_lfmc_dfmc_anomalies","PAS_6_jan_2021.tif"))
    gt = ds.GetGeoTransform()
    data['pws'] = np.array(ds.GetRasterBand(1).ReadAsArray())
    
    lats, lons = get_lats_lons(data['pws'], gt)
    
    keys = ['pws','silt','sand','clay', 'ks','thetas','isohydricity',\
        'root_depth','canopy_height','hft','p50','gpmax', 'c','g1','pft',
        "elevation","aspect","slope","twi","dry_season_length","ndvi",\
            "vpd_mean","vpd_std", "dist_to_water","agb","ppt_mean","ppt_std","lc","n"]
    
    array = np.zeros((len(keys), data['pws'].shape[0],data['pws'].shape[1])).astype('float')
    array[0] = data['pws']
    
    

    array[1]= get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242/data/Unified_NA_Soil_Map_Subsoil_Silt_Fraction.tif",lons,lats)
    array[2]= get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242/data/Unified_NA_Soil_Map_Subsoil_Sand_Fraction.tif",lons,lats)
    array[3]= get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242/data/Unified_NA_Soil_Map_Subsoil_Clay_Fraction.tif",lons,lats)
    array[4]= get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/Hydraul_Param_SoilGrids_Schaap_0/Ks_30cm.tif",lons,lats)
    array[5]= get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/Hydraul_Param_SoilGrids_Schaap_0/thetas_30cm.tif",lons,lats)
    

    
    array[6]= get_value("D:/Krishna/projects/wildfire_from_lfmc/data/traits/isohydricity.tif",lons,lats)
    array[7] = get_value("D:/Krishna/projects/wildfire_from_lfmc/data/traits/root_depth.tif",lons,lats)
    array[8] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/canopy_height/canopy_height.tif",lons,lats)
    array[9]= get_value("D:/Krishna/projects/wildfire_from_lfmc/data/traits/yanlan_HESS/HFT.tif",lons,lats)
    array[10]= get_value("D:/Krishna/projects/wildfire_from_lfmc/data/traits/yanlan_HESS/P50_liu.tif",lons,lats)
    array[11]= get_value("D:/Krishna/projects/wildfire_from_lfmc/data/traits/yanlan_HESS/gpmax_50.tif",lons,lats)
    array[12]= get_value("D:/Krishna/projects/wildfire_from_lfmc/data/traits/yanlan_HESS/C_50.tif",lons,lats)
    array[13]= get_value("D:/Krishna/projects/wildfire_from_lfmc/data/traits/yanlan_HESS/g1_50.tif",lons,lats)
    array[14]= get_value("D:/Krishna/projects/grid_fire/data/nlcd/nlcd_2016_4km.tif",lons,lats)
    
    array[15] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/Elevation/usa_dem.tif",lons,lats)
    array[16] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/ASPECT/usa_aspect_wgs1984_clip.tif",lons,lats)
    array[17] = get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/Elevation/usa_slope_project.tif",lons,lats)
    array[18]= get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/twi/twi.tif",lons,lats)
    
    band = 1
    ds = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/fireSeasonLength.tif")
    array[19]= ds.GetRasterBand(band).ReadAsArray()
    ds =  gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/ndvi_mean.tif")
    array[20] =ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/vpd_mean.tif")
    array[21] = ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/vpdStd.tif")
    array[22]= ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open("D:/Krishna/projects/pws_drivers/data/distance_to_water_bodies.tif")
    array[23]= ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open("D:/Krishna/projects/pws_drivers/data/agb_2020.tif")
    array[24]= ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/pptMean.tif")
    array[25]= ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/pptStd.tif")
    array[26]= ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open("D:/Krishna/projects/wildfire_from_lfmc/data/mean/landcover.tif")
    array[27]= ds.GetRasterBand(band).ReadAsArray()
    array[28]= get_value("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/soil/Hydraul_Param_SoilGrids_Schaap_0/n_30cm.tif",lons,lats)

    ds = None
    
    df = create_df(array,keys)
    df.dropna(subset = ["pws"], inplace = True)
    
    # df.loc[df['elevation']<-1e3] = np.nan
    # df.loc[df['slope']<-1e3] = np.nan
    # df.loc[df['aspect']>1e3] = np.nan
    # df.loc[df['twi']>1e4] = np.nan
    
    
    df.describe()
    df.loc[df['silt']<-1] = np.nan
    df.loc[df['sand']<-1] = np.nan
    df.loc[df['clay']<-1] = np.nan
    df.loc[df['ks']<-1] = np.nan
    df.loc[df['thetas']<-1] = np.nan
    df.loc[df['pft']<50] = np.nan
    
    
    df.loc[df['isohydricity']>1e3] = np.nan
    df.loc[df['root_depth']<-1] = np.nan
    df.loc[df['hft']<-1e3] = np.nan
    df.loc[df['p50']<-1e3] = np.nan
    df.loc[df['gpmax']<-1e3] = np.nan
    df.loc[df['c']<-1e3] = np.nan
    df.loc[df['g1']<-1e3] = np.nan
    df.loc[df['slope']<-1e3] = np.nan
    df.loc[df['aspect']>2e3] = np.nan
    df.loc[df['twi']>2e3] = np.nan
    

    store = pd.HDFStore('D:/Krishna/projects/pws_drivers/data/store_plant_soil_2_dec_2021.h5')
    store['df'] = df
    store.close()
    
    return df

#%%load dataset

def plot_heatmap(df):
    """
    Make a heatmap of correlations between PWS and features.
    """
    sample = df.sample(1e4)
    sns.pairplot(sample)
    
    
    sns.heatmap(df.corr(),vmin = -0.2, vmax = 0.2, cmap = sns.diverging_palette(240, 10, n=8))
    
    fig, axs= plt.subplots(5,3,figsize = (8,8),sharey=True)
    axs = axs.flatten()
    ctr=0
    for col in df.columns:
        if col=="pws":
            continue
        # df[col].hist()
        # ax.set_xlabel(col)
        axs[ctr].set_ylim(-0.1,2)
        if col=="hft":
            sns.boxplot(x=col, y="pws", data=df.sample(int(1e4)),ax=axs[ctr],color = "grey")
        else:
            # sns.regplot(x=col, y="pws", data=df.sample(int(1e4)),ci = None,
                          # scatter_kws={"s": 0.1,'alpha':0.5},line_kws = {"linewidth":0},ax=axs[ctr],color = "grey")
            sns.kdeplot(x=col, y="pws", data=df.sample(int(1e4)),fill=True,
                          cmap = "Greys",ax=axs[ctr])
            axs[ctr].set_xlim(df[col].quantile(0.05),df[col].quantile(0.95))
        axs[ctr].set_xlabel("")
        axs[ctr].annotate("%s"%str(col), xy=(0.5, 0.9), xytext=(0.5, 0.95), ha = "center", va = "top", textcoords="axes fraction",fontsize = 10)
        ctr+=1
    plt.show()
    return axs

def main():
    #%% make and save dataframe:
    # This can be run in Krishna's computer only because there are many tif files required
    # create_h5()
    
    #%% Load h5
    # make sure dir_data in dirs.py points to location of store_plant_soil_topo_climate.h5
    # This is typically location of repo/data
    store = pd.HDFStore(os.path.join(dirs.dir_data, 'store_plant_soil_topo_climate.h5'))
    df =  store['df']
    store.close()
    df.columns = df.columns.astype(str)    
    
    #%% Plot heatmap
    # plot_heatmap(df)
        
if __name__ == "__main__":
    main()