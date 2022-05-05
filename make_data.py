# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:10:13 2021

@author: kkrao

Create dataframe of with input (features) and output (pws)
Shape of dataframe = (# of pixels, # of features + 1)
"""

import os

import pandas as pd
from osgeo import gdal
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

def create_h5(store_path):
    """
    Bring together all features and labels (pws). drop nans. Store it as a h5
    file. And return it as a dataframe of shape (# of examples, # of features + 1)
    Returns
    -------
    df : pandas dataframe

    """
    data = dict()
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","PWS_through2021.tif"))
    gt = ds.GetGeoTransform()
    data['pws'] = np.array(ds.GetRasterBand(1).ReadAsArray())
    
    lats, lons = get_lats_lons(data['pws'], gt)
    
    keys = ['pws','silt','sand','clay', 'ks','thetas','isohydricity',\
        'root_depth','canopy_height','hft','p50','gpmax', 'c','g1','pft',
        "elevation","aspect","slope","twi","dry_season_length","ndvi",\
            "vpd_mean","vpd_std", "dist_to_water","agb","ppt_mean","ppt_std","lc",\
                "t_mean","t_std","ppt_lte_100", "lon","lat"]
    
    array = np.zeros((len(keys), data['pws'].shape[0],data['pws'].shape[1])).astype('float')
    array[0] = data['pws']
    
    

    array[1]= get_value(os.path.join(dirs.dir_data, "pws_features","Unified_NA_Soil_Map_Subsoil_Silt_Fraction.tif"),lons,lats)
    array[2]= get_value(os.path.join(dirs.dir_data, "pws_features","Unified_NA_Soil_Map_Subsoil_Sand_Fraction.tif"),lons,lats)
    array[3]= get_value(os.path.join(dirs.dir_data, "pws_features","Unified_NA_Soil_Map_Subsoil_Clay_Fraction.tif"),lons,lats)
    array[4]= get_value(os.path.join(dirs.dir_data, "pws_features","Ks_30cm.tif"),lons,lats)
    array[5]= get_value(os.path.join(dirs.dir_data, "pws_features","thetas_30cm.tif"),lons,lats)
    

    
    array[6]= get_value(os.path.join(dirs.dir_data, "pws_features","isohydricity.tif"),lons,lats)
    array[7] = get_value(os.path.join(dirs.dir_data, "pws_features","root_depth.tif"),lons,lats)
    array[8] = get_value(os.path.join(dirs.dir_data, "pws_features","canopy_height.tif"),lons,lats)
    array[9]= get_value(os.path.join(dirs.dir_data, "pws_features","HFT.tif"),lons,lats)
    array[10]= get_value(os.path.join(dirs.dir_data, "pws_features","P50_liu.tif"),lons,lats)
    array[11]= get_value(os.path.join(dirs.dir_data, "pws_features","gpmax_50.tif"),lons,lats)
    array[12]= get_value(os.path.join(dirs.dir_data, "pws_features","C_50.tif"),lons,lats)
    array[13]= get_value(os.path.join(dirs.dir_data, "pws_features","g1_50.tif"),lons,lats)
    array[14]= get_value(os.path.join(dirs.dir_data, "pws_features","nlcd_2016_4km.tif"),lons,lats)
    
    array[15] = get_value(os.path.join(dirs.dir_data, "pws_features","usa_dem.tif"),lons,lats)
    array[16] = get_value(os.path.join(dirs.dir_data, "pws_features","usa_aspect_wgs1984_clip.tif"),lons,lats)
    array[17] = get_value(os.path.join(dirs.dir_data, "pws_features","usa_slope_project.tif"),lons,lats)
    array[18]= get_value(os.path.join(dirs.dir_data, "pws_features","twi.tif"),lons,lats)
    
    band = 1
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","fireSeasonLength.tif"))
    array[19]= ds.GetRasterBand(band).ReadAsArray()
    ds =  gdal.Open(os.path.join(dirs.dir_data, "pws_features","ndvi_mean.tif"))
    array[20] =ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","vpd_mean.tif"))
    array[21] = ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","vpdStd.tif"))
    array[22]= ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","distance_to_water_bodies.tif"))
    array[23]= ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","agb_2020.tif"))
    array[24]= ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","pptMean.tif"))
    array[25]= ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","pptStd.tif"))
    array[26]= ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","landcover.tif"))
    array[27]= ds.GetRasterBand(band).ReadAsArray()
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","tMean.tif"))
    array[28]= ds.GetRasterBand(band).ReadAsArray() 
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","tStd.tif"))
    array[29]= ds.GetRasterBand(band).ReadAsArray() 
    ds = gdal.Open(os.path.join(dirs.dir_data, "pws_features","ppt_lte_100.tif"))
    array[30]= ds.GetRasterBand(band).ReadAsArray() 
    array[31]= lons
    array[32]= lats
    
    ds = None
    
    df = create_df(array,keys)
    df.dropna(subset = ["pws"], inplace = True)
    
    df.describe()
    df.loc[df['silt']<-1] = np.nan
    df.loc[df['sand']<-1] = np.nan
    df.loc[df['clay']<-1] = np.nan
    df.loc[df['ks']<-1] = np.nan
    df.loc[df['thetas']<-1] = np.nan
    df.loc[df['pft']<41] = np.nan
    df.loc[df['pft']>81] = np.nan
    df.loc[df['elevation']<-1e3] = np.nan
    df.loc[df['slope']<-1e3] = np.nan
    df.loc[df['aspect']>2e3] = np.nan
    df.loc[df['twi']>2e3] = np.nan
    
    #plot map of where there is data
    df2 = df.copy()
    droppedFeats = ["pws", 'sand', 'silt', 'clay', 'ks', 'thetas', 'pft', 
                   'elevation', 'slope', 'aspect', 'twi']
    df2.dropna(subset = droppedFeats, inplace = True)
    latMap = np.empty( np.shape(pws) ) * np.nan
    latInd = np.round( (df2['lat'].to_numpy() - geotransform[3])/geotransform[5] ).astype(int)
    lonInd = np.round( (df2['lon'].to_numpy() - geotransform[0])/geotransform[1] ).astype(int)
    latMap[latInd, lonInd] = 1
    fig, ax1 = plt.subplots()
    im = ax1.imshow(latMap, interpolation='none')
    plt.title('lats with PWS, soil, elev, and PFT, NaNs')
    print('removing PWS, soil, elev, and PFT NaNs has length: ' + str(len(df3)))
    
    df.loc[df['isohydricity']>1e3] = np.nan
    df.loc[df['root_depth']<-1] = np.nan
    df.loc[df['hft']<-1e3] = np.nan
    df.loc[df['p50']<-1e3] = np.nan
    df.loc[df['gpmax']<-1e3] = np.nan
    df.loc[df['c']<-1e3] = np.nan
    df.loc[df['g1']<-1e3] = np.nan

    #plot map after traits
    df3 = df.copy()
    df3.dropna(inplace = True)
    latMap = np.empty( np.shape(pws) ) * np.nan
    latInd = np.round( (df3['lat'].to_numpy() - geotransform[3])/geotransform[5] ).astype(int)
    lonInd = np.round( (df3['lon'].to_numpy() - geotransform[0])/geotransform[1] ).astype(int)
    latMap[latInd, lonInd] = 1
    fig, ax1 = plt.subplots()
    im = ax1.imshow(latMap, interpolation='none')
    plt.title('lats with PWS, soil, elev, PFT, and trait, NaNs')
    print('removing all NaNs has length: ' + str(len(df4)))
    

    store = pd.HDFStore(store_path)
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
    store_path = os.path.join(dirs.dir_data, 'store_plant_soil_topo_climate_PWSthrough2021v2.h5')
    # This can be run in Krishna's computer only because there are many tif files required
    create_h5(store_path)
    
    #%% Load h5
    # make sure dirs.dir_data in dirs.py points to location of store_plant_soil_topo_climate.h5
    # This is typically location of repo/data
    store = pd.HDFStore(store_path)
    df =  store['df']
    store.close()
    df.columns = df.columns.astype(str)    
    
    #%% Plot heatmap
    # plot_heatmap(df)
        
if __name__ == "__main__":
    main()