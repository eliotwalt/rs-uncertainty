import rasterio
import rasterio.warp
import rasterio.features
import fiona
import yaml
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import entropy
import pandas as pd
import math
import os
from pathlib import Path
from tqdm import tqdm
import random
import argparse
import re

def pjoin(*subs): return Path(os.path.join(*subs))
def parse_date(date_str) -> datetime: return datetime.strptime(date_str, '%Y%m%d')
def parse_gt_date(date_str) -> datetime: 
    # there are 2 gt date formats
    try: return datetime.strptime(date_str, "%Y-%m-%d")
    except: return datetime.strptime(date_str, "%Y/%m/%d")

def get_rasterized_polygon(project_id, cfg, gt_file):
    project_shape_collections = [fiona.open(p) for p in cfg['project_shapefiles']]
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
    polygon = rasterio.warp.transform_geom(src_crs=crs, dst_crs=gt_file.crs, geom=polygon)
    rasterized_polygon = rasterized_polygon = rasterio.features.rasterize(
        [(polygon, 1)],
        out_shape=gt_file.shape,
        transform=gt_file.transform,
        fill=0,
        dtype='uint8'
    )
    return rasterized_polygon, gt_date

def load_project_data(project_id, cfg):
    # Create invalid mask (invalid=nan, valid=1)
    gtf = rasterio.open(pjoin(cfg["gt_dir"], project_id+".tif"))
    # Valid mask
    valid_mask = gtf.read_masks(1)//255
    # Load polygon
    polygon, _ = get_rasterized_polygon(project_id, cfg, gtf)

    # Load split masks
    with rasterio.open(pjoin(cfg["split_mask_dir"], project_id+".tif")) as fh:
        split_mask = fh.read(1).astype("float16")

    # Collect dates and cloud maps
    cloud_maps, dates = [], []
    s2_paths = list(pjoin(cfg["s2_reprojected_dir"], project_id).glob("*.tif"))
    for s2_path in s2_paths:
        with rasterio.open(s2_path) as fh:
            cloud_map = fh.read(13).astype("float16")
        date = parse_date(s2_path.stem.split('_')[3].split('T')[0])
        cloud_maps.append(cloud_map)
        dates.append(date)
    cloud_maps, dates = list(zip(*sorted(zip(cloud_maps, dates), key=lambda x: x[1])))
    
    # stack
    cloud_tensor = np.stack(cloud_maps, axis=0)

    return gtf, valid_mask, polygon, split_mask, cloud_tensor, s2_paths

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--cfg", help="config file")
    args = p.parse_args()
    # Load config
    with open(args.cfg, "r") as f: cfg = yaml.safe_load(f)
    # arguments
    patch_half = cfg["patch_size"]//2
    projects = cfg["projects"]
    out_dir = cfg["save_dir"]
    bins = cfg["cloud_probability_bins"]
    # loop
    for project_id in projects:
        counts = {k: 0 for k in range(1, len(bins)+1)}
        print(f"Processing {project_id}...")
        gt_file, valid_mask, rasterized_polygon, split_mask, cloud_tensor, s2_paths = load_project_data(project_id, cfg)
        shape = cloud_tensor.shape[1:]
        cp_masks = np.zeros(cloud_tensor.shape, dtype=np.uint8)
        for i in tqdm(range(patch_half, shape[0]-patch_half)):
            patches  = None # Debug
            for j in range(patch_half, shape[1]-patch_half):
                i_slice = slice(i - patch_half, i + patch_half + 1)
                j_slice = slice(j - patch_half, j + patch_half + 1)
                is_same_split = (split_mask[i_slice, j_slice] == split_mask[i_slice, j_slice][0, 0]).all()
                is_in_polygon = (rasterized_polygon[i_slice, j_slice] == 1).all()
                if is_same_split and is_in_polygon and valid_mask[i,j]:
                    # (i,j) is a valid patch center
                    patches = cloud_tensor[:,i_slice,j_slice]
                    date_means = patches.mean(axis=(1,2))
                    bin_ids = np.digitize(date_means, bins=bins)
                    # ignore patches that have no clouds across dates
                    if (patches==0).all(): continue
                    # ignore patches that fall in less than 2 bins
                    if np.unique(bin_ids).shape[0] < 2: continue
                    # fill raster
                    cp_masks[:,i,j]=bin_ids
                    # update counts
                    for bin_id in bin_ids:
                        if bins[-1]==100 and bin_id==len(bins): bin_id -= 1
                        counts[bin_id] += 1
        # save masks
        save_dir = pjoin(out_dir, project_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        for s2_path, cp_mask in zip(s2_paths, cp_masks):
            with rasterio.Env():
                profile = gt_file.profile
                profile.update(
                    driver='GTiff',
                    count=1,
                    compress='deflate',
                    dtype=rasterio.uint8,
                    nodata=None)
                with rasterio.open(pjoin(save_dir, s2_path.stem+".tif"), 'w', **profile) as f:
                    f.write(cp_mask.astype('uint8').reshape(1, cp_mask.shape[0], cp_mask.shape[1]))
                print(f"Saved ct mask at {pjoin(save_dir, s2_path.stem+'.tif')}")
        # print counts
        for bin_, count in counts.items():
            b0 = bins[bin_-1]
            b1 = str(bins[bin_])+")" if bin_ < len(bins) else "100]"
            print(f"bin_id={bin_}, bounds=[{b0}, {b1}, count={count}")
        # save counts
        with open(pjoin(save_dir, "stats.txt"), "w") as f:
            for bin_, count in counts.items():
                b0 = bins[bin_-1]
                b1 = str(bins[bin_])+")" if bin_ < len(bins) else "100]"
                f.write(f"bin_id={bin_}, bounds=[{b0}, {b1}, count={count}\n")
    # write all counts
    counts = {k: 0 for k in range(1, len(bins)+1)}
    for project_id in projects:
        with pjoin(out_dir, project_id, "stats.txt").open() as f:
            for line in f.readlines():
                bin_id = int(re.findall("bin_id=\d+", line)[0].split("=")[1])
                cnt = int(re.findall("count=\d+", line)[0].split("=")[1])
                counts[bin_id] += cnt
    with pjoin(out_dir, "stats.txt").open("w") as f:
        for bin_, count in counts.items():
            b0 = bins[bin_-1]
            b1 = str(bins[bin_])+")" if bin_ < len(bins) else "100]"
            f.write(f"bin_id={bin_}, bounds=[{b0}, {b1}, count={count}\n")

if __name__ == "__main__":
    main()
