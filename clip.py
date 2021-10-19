# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:40:31 2021

@author: kkrao
"""
import fiona
import rasterio
import rasterio.mask

def clip(rasterPath,shapePath,writePath):
    
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
     
rasterPath = "D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/ASPECT/usa_aspect_wgs1984.tif"
shapePath = "D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/west_usa_shapefile_lcc.shp"
writePath = "D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/ASPECT/usa_aspect_wgs1984_clip.tif"
clip(rasterPath,shapePath,writePath)