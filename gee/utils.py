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

def configure(project_id, root="/scratch/ewalt/pdm/rs-uncertainty/"):
    return {
        "project_id": project_id,
        "gt_dir": pjoin(root, "assets/data/preprocessed"),
        "gt_data_bands": [1, 2, 3, 4, 5],
        "shapefile_paths": [
            pjoin(root, p) for p in 
            ['assets/data/NHM_projectDekning_AOI_edit2015_V2.shp', 'assets/data/ALS_projects_Dz_all_norway.shp']
        ],
    }

def get_project_data(
    project_id,
    gt_dir,
    gt_data_bands,
    shapefile_paths,
    target_crs="EPSG:4326",
):
    # Get gt
    gt_file = rasterio.open(pjoin(gt_dir, project_id+".tif"))
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
    print("reprojecting polygon", crs, "->", target_crs)
    polygon = rasterio.warp.transform_geom(src_crs=crs, dst_crs=target_crs, geom=polygon)
    return gt_file, gt, gt_date, gt_file.crs, polygon["coordinates"], bounds