import yaml
import time
import rasterio
import rasterio.warp 
import rasterio.features
import fiona
from datetime import datetime, timedelta
import numpy as np
import os
from pathlib import Path

def pjoin(*subs): return Path(os.path.join(*subs))
def parse_date(date_str) -> datetime: return datetime.strptime(date_str, '%Y%m%d')
def parse_gt_date(date_str) -> datetime: 
    # there are 2 gt date formats
    try: return datetime.strptime(date_str, "%Y-%m-%d")
    except: return datetime.strptime(date_str, "%Y/%m/%d")

def get_project_data(
    project_id,
    gt_dir,
    gt_data_bands,
    shapefile_paths,
    target_crs="EPSG:4326",
):
    # Get gt
    gt_path = pjoin(gt_dir, project_id+".tif")
    gt_file = rasterio.open(gt_path)
    gt = gt_file.read(gt_data_bands)
    bounds = gt_file.bounds
    # Load shapefiles
    project_shape_collections = [fiona.open(p) for p in shapefile_paths]
    # create the shape ("polygon") associated to the project 
    polygon, crs, gt_date = None, None, None
    for collection in project_shape_collections:
        try:
            polygon = [s['geometry'] for s in collection if s['properties']['kv_id'] == int(project_id)][0]
            crs = collection.crs
            gt_date = [s["properties"]["PUB_DATO"] for s in collection if s['properties']['kv_id'] == int(project_id)][0]
            gt_date = parse_gt_date(gt_date)
            break
        except IndexError: pass 
    if polygon is None: print("No polygon found")
    polygon = rasterio.warp.transform_geom(src_crs=crs, dst_crs=target_crs, geom=polygon)
    bbox = {"geometries": None, "type": "Polygon", "coordinates": [[(bounds.left, bounds.bottom), (bounds.right, bounds.top)]]}
    bbox = rasterio.warp.transform_geom(src_crs=gt_file.crs, dst_crs=target_crs, geom=bbox)
    # change order of coords
    bbox["coordinates"] = bbox["coordinates"][0][:2]
    return gt_path, gt_file, gt, gt_date, gt_file.crs, polygon["coordinates"], bbox["coordinates"]
