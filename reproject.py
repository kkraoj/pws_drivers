# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:56:55 2021

@author: kkrao
"""


import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

dst_crs = 'EPSG:4326'

with rasterio.open('D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/ASPECT/usa_aspect.tif') as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Elevation/ASPECT/usa_aspect_wgs1984.tif', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)