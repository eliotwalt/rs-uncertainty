import yaml
import time
import os 
import rasterio
import rasterio.warp 
import rasterio.features
import fiona
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import gdal
from pathlib import Path
import subprocess, shlex

def pjoin(*subs): return Path(os.path.join(*subs))
def parse_date(date_str) -> datetime: return datetime.strptime(date_str, '%Y%m%d')
def parse_gt_date(date_str) -> datetime: 
    # there are 2 gt date formats
    try: return datetime.strptime(date_str, "%Y-%m-%d")
    except: return datetime.strptime(date_str, "%Y/%m/%d")

def get_gt_project_data(
    project_id,
    gt_dir,
    shapefile_paths,
    target_crs="EPSG:4326",
):
    # Get gt
    gt_path = pjoin(gt_dir, project_id+".tif")
    gt_file = rasterio.open(gt_path)
    bounds = gt_file.bounds
    # Load shapefiles
    project_shape_collections = [fiona.open(p) for p in shapefile_paths]
    # create the shape ("polygon") associated to the project 
    gt_date = None
    for collection in project_shape_collections:
        try:
            polygon = [s['geometry'] for s in collection if s['properties']['kv_id'] == int(project_id)][0]
            gt_date = [s["properties"]["PUB_DATO"] for s in collection if s['properties']['kv_id'] == int(project_id)][0]
            gt_date = parse_gt_date(gt_date)
            break
        except IndexError: pass 
    if polygon is None: print("No polygon found")
    bbox = {"geometries": None, "type": "Polygon", "coordinates": [[(bounds.left, bounds.bottom), (bounds.right, bounds.top)]]}
    bbox = rasterio.warp.transform_geom(src_crs=gt_file.crs, dst_crs=target_crs, geom=bbox)
    # change order of coords
    bbox["coordinates"] = bbox["coordinates"][0][:2]
    return gt_path, gt_file, gt_date, gt_file.crs, bbox["coordinates"]

def get_s2_project_data(
    project_id,
    s2_reprojected_dir,
    target_crs="EPSG:4326",
    s2_date=None,
    dateformat="%Y%m%dT%H%M%S"
):
    # get latest gt date (latest so that we have more chance to have it in the image collection)
    s2_file = None
    if s2_date is None:
        for s2_path in pjoin(s2_reprojected_dir, project_id).glob("*.tif"):
            date = datetime.strptime(s2_path.stem.split('_')[3], dateformat)
            if s2_date is None or date > s2_date: 
                s2_file = rasterio.open(s2_path)
                s2_date = date
    else:
        all_dates = []
        if isinstance(s2_date, str): s2_date = datetime.strptime(s2_date, dateformat)
        for s2_path in pjoin(s2_reprojected_dir, project_id).glob("*.tif"):
            date = datetime.strptime(s2_path.stem.split('_')[3], dateformat)
            all_dates.append(s2_path.stem.split('_')[3])
            if s2_date==date:
                s2_file = rasterio.open(s2_path)
    if s2_file is None:
        raise ValueError(f"Could not find provided date in s2 dir: {date.strftime(dateformat)}. Possible values: {all_dates}")
    # bbox
    bounds = s2_file.bounds
    bbox = {"geometries": None, "type": "Polygon", "coordinates": [[(bounds.left, bounds.bottom), (bounds.right, bounds.top)]]}
    bbox = rasterio.warp.transform_geom(src_crs=s2_file.crs, dst_crs=target_crs, geom=bbox)
    # change order of coords
    bbox["coordinates"] = bbox["coordinates"][0][:2]
    return s2_path, s2_file, s2_date, s2_file.crs, bbox["coordinates"]

def warp(input_path, ref_path, out_path):
    # Paths
    input_path = str(input_path)
    ref_path = str(ref_path)
    out_path = str(out_path)
    # Load reference dataset
    ref_ds = gdal.Open(ref_path)
    # warp options
    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=ref_ds.GetProjectionRef(),
        xRes=10.0, yRes=-10.0
    )
    # reproject
    out_ds = gdal.Warp(out_path, input_path, options=warp_options)
    return ref_ds, out_ds

def filterInvalidCloudProba(paths, bands):
    """
    Invalid: 
        - all 0
        - all 100  
        => constant array
    """
    if "MSK_CLDPRB" in bands:
        updated_paths = []
        idx = bands.index("MSK_CLDPRB")
        for path in paths:
            with rasterio.open(path) as f:
                cp = f.read(idx)
                if np.all(cp==np.ravel(cp)[0]): continue # all values are the same => invalid
                updated_paths.append(path)
        return updated_paths
    else:
        return paths
    
# Warping methods
def computeGDalRastersOffsets(ds1, ds2, round_fn=round):
    assert ds1.GetProjectionRef()==ds2.GetProjectionRef()
    # Get transforms
    Ox1, pw1, b1, Oy1, d1, ph1 = ds1.GetGeoTransform()
    Ox2, pw2, b2, Oy2, d2, ph2 = ds2.GetGeoTransform()
    assert (pw1==pw2 and ph1==ph2)
    # get offsets
    di = round_fn((Ox2-Ox1)/ph1)
    dj = round_fn((Oy2-Oy1)/pw1)
    return di, dj

def validGDalRasterPosition(i, j, di, dj, other_h, other_w):
    return (di<=i<=other_h+di and dj<=j<=other_w+dj)

def warpGDal(input_path, ref_path, out_path):
    # Paths
    input_path = str(input_path)
    ref_path = str(ref_path)
    out_path = str(out_path)
    # Load reference dataset
    ref_ds = gdal.Open(ref_path)
    # warp options
    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=ref_ds.GetProjectionRef(),
        xRes=10.0, yRes=-10.0
    )
    # warp
    out_ds = gdal.Warp(out_path, input_path, options=warp_options)
    return ref_ds, out_ds

def alignGDalWarpedRaster(ref_path, warped_path, round_fn=round):
    # get offsets
    refDs, gdalwarpDs = gdal.Open(ref_path), gdal.Open(warped_path)
    di, dj = computeGDalRastersOffsets(refDs, gdalwarpDs, round_fn=round_fn)
    # load rasters with rasterio
    ref_file = rasterio.open(ref_path)
    refData = ref_file.read(ref_file.indexes)
    gdalwarp_file = rasterio.open(warped_path)
    gdalwarpData = gdalwarp_file.read(gdalwarp_file.indexes)
    # Align rasters
    aligned_gdalwarpData = np.zeros_like(refData)
    _, other_h, other_w = gdalwarpData.shape
    for i in range(refData.shape[1]):
        for j in range(refData.shape[2]):
            if validGDalRasterPosition(i, j, di, dj, other_h, other_w):
                try:
                    aligned_gdalwarpData[:,i,j] = gdalwarpData[:,i+di,j+dj]
                except IndexError: pass
    with rasterio.Env():
        profile = ref_file.profile
        with rasterio.open(warped_path, "w", **profile) as f:
            f.write(aligned_gdalwarpData)
        
def warpAlignedGDal(input_path, ref_path, out_path, round_fn=round):
    warp(input_path, ref_path, out_path)
    alignGDalWarpedRaster(ref_path, out_path, round_fn=round_fn)                                                            

def warpRio(input_path, ref_path, out_path):
    cmd = f"rio warp {input_path} {out_path} --like {ref_path}"
    print(f"Executing rio warp command: {cmd}")
    r = subprocess.run(cmd.split(), capture_output=True, text=True)
    if len(r.stdout.split())!=0: 
        raise RuntimeError(f"rio warp failed: {r.stdout}")

# stat funcs
def getDataPositions(*xs, nodata=0.):
    valid_pos = [[] for _ in range(len(xs))]
    for i in range(xs[0].shape[1]):
        for j in range(xs[0].shape[2]):
            for x, vp in zip(xs, valid_pos):
                if not (x[:,i,j]==nodata).all(): vp.append((i,j))
    return tuple(valid_pos)

def countNoData(refData, rioData, gdalData=None, nodata=0.):
    N = refData.shape[1]*refData.shape[2]
    if gdal is not None:
        assert refData.shape==rioData.shape==gdalData.shape
        ref_pos, rio_pos, gdal_pos = getDataPositions(refData, rioData, gdalData, nodata=nodata)
        return ("ref", N-len(ref_pos)), ("rio", N-len(rio_pos)), ("gdal", N-len(gdal_pos))
    else:
        assert refData.shape==rioData.shape
        ref_pos, rio_pos = getDataPositions(refData, rioData, nodata=nodata)
        return ("ref", N-len(ref_pos)), ("rio", N-len(rio_pos))

def getGDalRioStatDataFrame(refData, rioData, gdalData, stat_name, round_fn, nodata=0.):
    assert stat_name in ["min", "max", "mean", "std"]
    assert refData.shape==rioData.shape==gdalData.shape
    stat_fn = eval(f"np.{stat_name}")
    gdalKey = f"gdal_{round_fn.__name__}"
    gdalDeltaKey = f"gdal_{round_fn.__name__}_delta"
    df = {"band": [], "stat": [], "ref": [], "rio": [], gdalKey: [], "rio_delta": [], gdalDeltaKey: []}
    num_bands = refData.shape[0]
    if nodata is not None:
        ref_pos, rio_pos, gdal_pos = getDataPositions(refData, rioData, gdalData, nodata=nodata)
        ref_pos, rio_pos, gdal_pos = list(zip(*ref_pos)), list(zip(*rio_pos)), list(zip(*gdal_pos))
    for band in range(num_bands):
        _refData = refData[band]
        _rioData = rioData[band]
        _gdalData = gdalData[band]
        if nodata is not None:           
            _refData = _refData[ref_pos[0],ref_pos[1]]         
            _rioData = _rioData[rio_pos[0],rio_pos[1]]            
            _gdalData = _gdalData[gdal_pos[0],gdal_pos[1]]
        df["band"].append(band)
        df["stat"].append(stat_name)
        df["ref"].append(stat_fn(_refData))
        df["rio"].append(stat_fn(_rioData))
        df[gdalKey].append(stat_fn(_gdalData))
        df["rio_delta"].append(df["ref"][-1]-df["rio"][-1])
        df[gdalDeltaKey].append(df["ref"][-1]-df[gdalKey][-1])
    return pd.DataFrame(df)

def getStatDataFrame(refData, rioData, stat_name, nodata=0.):
    assert stat_name in ["min", "max", "mean", "std"]
    assert refData.shape==rioData.shape
    stat_fn = eval(f"np.{stat_name}")
    df = {"band": [], "stat": [], "ref": [], "rio": [], "rio_delta": []}
    num_bands = refData.shape[0]
    if nodata is not None:
        ref_pos, rio_pos = getDataPositions(refData, rioData, nodata=nodata)
        ref_pos, rio_pos = list(zip(*ref_pos)), list(zip(*rio_pos))
    for band in range(num_bands):
        _refData = refData[band]
        _rioData = rioData[band]
        if nodata is not None:
            _refData = _refData[ref_pos[0],ref_pos[1]]         
            _rioData = _rioData[rio_pos[0],rio_pos[1]]    
        df["band"].append(band)
        df["stat"].append(stat_name)
        df["ref"].append(stat_fn(_refData))
        df["rio"].append(stat_fn(_rioData))
        df["rio_delta"].append(df["ref"][-1]-df["rio"][-1])
    return pd.DataFrame(df)