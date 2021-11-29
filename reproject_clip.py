# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:40:31 2021

@author: kkrao

Script used for spatial manipulation of rasters.
"""

import numpy as np
import fiona
import rasterio
import rasterio.mask
import rasterio.warp  

def reproject(dst_crs,rasterPath, writePath):
    """
    Fast Reprojection of raster. Uses nearest neighbour.

    Parameters
    ----------
    rasterPath : tif or any other format supported by rasterio
    writePath : Where to store the reprojected raster?

    Returns
    -------
    None.

    """
    with rasterio.open(rasterPath) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
    
        with rasterio.open(writePath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.nearest)
            

def clip(rasterPath,shapePath,writePath):
    """
    Clip raster using a shapefile. Raster extent will equal shapefile's extent.
    Values outside shapefile, but inside shapefile extent will be nans.

    Parameters
    ----------
    rasterPath : tif or any other format supported by rasterio
    shapePath : .shp file
    writePath : Where to store the clipped raster?

    Returns
    -------
    None.

    """
    with fiona.open(shapePath, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        
    with rasterio.open(rasterPath) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
        
    out_meta.update({"driver": "GTiff",
                      "height": out_image.shape[1],
                      "width": out_image.shape[2],
                      "transform": out_transform})
    
    with rasterio.open(writePath, "w", **out_meta) as dest:
        dest.write(out_image)
    
def main():
    #%% Reproject
    dst_crs = 'EPSG:4326'
    rasterPath = "D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/ASPECT/usa_aspect.tif"
    writePath = "D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/ASPECT/usa_aspect_wgs1984.tif"
    reproject(dst_crs, rasterPath,writePath)
    
    #%% Clip
    rasterPath = "D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/ASPECT/usa_aspect_wgs1984.tif"
    shapePath = "D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/west_usa_shapefile_lcc.shp"
    writePath = "D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/ASPECT/usa_aspect_wgs1984_clip.tif"
    clip(rasterPath,shapePath,writePath) 
    
    
if __name__ == "__main__":
    main()
    
     
