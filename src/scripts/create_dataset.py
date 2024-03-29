import os, sys, shutil
import argparse
from copy import deepcopy
from typing import *
from pathlib import Path
from datetime import datetime, timedelta
from itertools import chain
from collections import defaultdict
import pickle
import yaml, json
import blowtorch
from blowtorch import Run
from tqdm import trange, tqdm
import numpy as np
import fiona
import os
import rasterio.warp
import rasterio.features
import matplotlib
import matplotlib.pyplot as plt
from time import time
import sys, os 
root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, root)
from utils import split_list, RunningStats, CombinedStats

SEPARATOR = 65535

def _path(x):
    try:
        return Path(x)
    except Exception as e: raise e

def pjoin(*subs): return Path(os.path.abspath(os.path.join(*subs)))

def parse_date(date_str) -> datetime: return datetime.strptime(date_str, '%Y%m%d')
def parse_gt_date(date_str) -> datetime: 
    # there are 2 gt date formats
    try: return datetime.strptime(date_str, "%Y-%m-%d")
    except: return datetime.strptime(date_str, "%Y/%m/%d")

def pjoin(*subs: List[Union[str,Path]]) -> Path: return Path(os.path.join(*subs))

def defaultdict2dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict2dict(v) for k, v in d.items()}
    return d

class ProjectsPreprocessor:
    """
    Process a list of projects
    """
    def __init__(
        self,
        run: blowtorch.run.Run,
        seed: int=12345,
        separator: int=SEPARATOR,
        split_map = ["train", "val", "test"], # if ["test"] then it will only generate the test set
        verbose: bool=True
    ) -> None:
        # attributes
        self.run = run
        self.seed = seed
        self.separator = separator
        self.split_map = split_map
        self.verbose = verbose
        self.patches_sampling_strategy_map = {
            "valid_center": self._get_valid_center_patches,
            "s2_offset_valid_center": self._get_s2_offset_valid_center_patches,
        }
        self.projects_stats = {}
        # verify run config
        self._validate_run_config()
        # set all random seeds
        self.run.seed_all(self.seed)
        # save directory
        self.save_dir = self.run["save_dir"]
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        # create dataset
        self.prange = trange

    def __call__(self):
        # collect arguments
        args_dicts = []
        projects = self.run["projects"].copy()
        # only loop on projects that are actually here
        gt_projects = set([d.stem for d in Path(self.run["gt_dir"]).glob("*.tif")])
        s2_projects = set([d.name for d in os.scandir(self.run["s2_reprojected_dir"])])
        projects = list(s2_projects.intersection(gt_projects).intersection(set(projects)))
        i, N = 0, len(projects)
        for gt_file_path in Path(self.run["gt_dir"]).glob("*.tif"):
            if gt_file_path.stem in projects and os.path.exists(pjoin(self.run["s2_reprojected_dir"], gt_file_path.stem)):
                i += 1
                projects.remove(gt_file_path.stem)
                #project_projects_stats = self._create_project_dataset(gt_file_path)
                #self.projects_stats[gt_file_path.stem] = project_projects_stats
                args_dicts.append({
                    "gt_file_path": gt_file_path,
                    "i": i, "N": N
                })
                if len(projects) == 0: break
        # create dataset
        start = time()
        if self.verbose: print("Processing projects")
        for args_dict in args_dicts:
            project_projects_stats, project_id = self._create_project_dataset(args_dict)
            self.projects_stats[project_id] = project_projects_stats
        if self.verbose: print("Done processing projects in {}".format(timedelta(seconds=time()-start)))
        # save stats and config
        start = time()
        if self.verbose: print("Writing stats")
        if "stats_path" in self.run.get_raw_config() and self.run["stats_path"] is not None:
            self._copy_global_stats(self.projects_stats, self.run["stats_path"])
        with pjoin(self.save_dir, "stats.yaml").open("w") as fh:
            yaml.dump(self.projects_stats, fh)
        if self.verbose: print("Done writing stats in {}".format(timedelta(seconds=time()-start)))
        start = time()
        if self.verbose: print("Writing config")
        with pjoin(self.save_dir, "data_config.yaml").open("w") as fh:
            yaml.dump(self.run.get_raw_config(), fh)
        if self.verbose: print("Done writing config in {}".format(timedelta(seconds=time()-start)))

    def _copy_global_stats(
        self,
        stats,
        stats_path
    ):
        with open(stats_path) as f:
            external_stats = yaml.safe_load(f)
        stats.update({
            "s2_stats": external_stats["s2_stats"],
            "s1_stats": external_stats["s1_stats"],
            "labels_stats": external_stats["labels_stats"],
        })
        return stats

    def _create_project_dataset(self, args_dict: dict) -> None:
        gt_file_path, i, N = args_dict["gt_file_path"], args_dict["i"], args_dict["N"]
        if self.verbose:
            start = time()
            print(f"[{i}/{N}] Processing id {gt_file_path.stem} ({datetime.now().strftime('%H:%M:%S')})")
        # get project id
        project_id = gt_file_path.stem
        # initialize stats dict
        project_projects_stats = {}
        project_projects_stats["num_train"] = 0
        project_projects_stats["num_val"] = 0
        project_projects_stats["num_test"] = 0
        # read gt
        gt_file, valid_mask, labels = self._read_gt_file(gt_file_path)
        project_projects_stats["shape"] = list(valid_mask.shape)
        # read split_mask
        smpath = None if self.run["split_mask_dir"] is None else pjoin(self.run["split_mask_dir"], project_id+".tif")
        split_mask = self._read_split_mask(smpath, gt_file.shape)
        # get rasterized polygon
        rasterized_polygon, gt_date = self._get_polygon_raster(
            self.run["project_shapefiles"],
            project_id, gt_file
        )
        project_projects_stats["gt_date"] = gt_date
        # read images
        images, image_ids, s2_images, s1_images_ascending, s1_images_descending = self._read_project_images(project_id)
        project_projects_stats["s2_dates"] = [d for _, d in s2_images]
        project_projects_stats["s1_ascending_dates"] = [d for _, d in s1_images_ascending]
        project_projects_stats["s1_descending_dates"] = [d for _, d in s1_images_descending]
        project_projects_stats["num_s2_images"] = len(s2_images)
        project_projects_stats["num_s1_images"] = len(s1_images_ascending)
        project_projects_stats["num_s1_images"] = len(s1_images_descending)
        # make patches
        patches_stats = self._make_project_patches(
            gt_file,
            project_id,
            split_mask,
            rasterized_polygon,
            valid_mask,
            s1_images_ascending,
            s1_images_descending,
            s2_images,
            images,
            image_ids,
            labels,
            gt_date,
        )
        # copy stats
        for k, v in patches_stats.items():
            project_projects_stats[k] = v
            if k in ["num_train", "num_test", "num_test"]: project_projects_stats[k] += v
        if self.verbose:
            print(f"[{i}/{N}] Done processing id {gt_file_path.stem} in {str(timedelta(seconds=time()-start))}")
        gt_file.close()
        return project_projects_stats, project_id

    def _validate_run_config(self) -> None:
        """
        Validate run configuration:
        1. `patch_size` must be odd
        2. `sampling_strategy` in 
        """
        assert self.run["patch_size"]%2==1, "Path size must be odd."
        assert self.run['sampling_strategy'] in self.patches_sampling_strategy_map.keys(), \
            f"Invalid sampling strategy: {self.run['sampling_strategy']}. Possible values are {list(self.patches_sampling_strategy_map.keys())}."

    def _read_gt_file(self, path):
        """
        DatasetCreateor._read_gt_file method: read and return gt_file
        
        Args:
        - path (Union[str, pathlib.Path]): path to gt file

        Returns:
        - gt_file (rasterio.io.DatasetReader): Rasterio opened gt file
        - valid_mask (np.ndarray[H, W]): Binary validity mask
        - labels (np.ndarray[num_label_bands, H, W]): Dense forest structure variables
        """
        gt_file = rasterio.open(path)
        valid_mask = gt_file.read_masks(1)
        labels = gt_file.read(self.run['data_bands'])
        # set invalid gt points to nan
        labels[:, valid_mask == 0] = np.nan
        return gt_file, valid_mask, labels
    
    def _read_split_mask(self, path, shape):
        """ 
        DatasetCreateor._read_split_mask method: read and return split mask
        
        Args:
        - path (Union[str, pathlib.Path]): path to split mask file

        Returns:
        - split_mask (np.ndarray[H, W]): Cross-validation split mask, `0->train`, `1->val`, `2->test`.
        """
        if path is None:
            split_mask = np.full(shape, 2, dtype="float16")
        else:
            with rasterio.open(path) as f: split_mask = f.read(1).astype('float16')
        assert split_mask.shape == shape, f"Invalid shape mask shape"
        return split_mask
    
    def _get_polygon_raster(self, files: List[Union[str, Path]], project_id: str, gt_file):
        """
        DatasetCreateor._get_polygon_raster method: creates polygon raster associated to a project
        
        Args:
        - project_id (str): project id
        - files (List[Union[str, Path]]): shape file paths list

        Returns:
        - rasterized_polygon (np.ndarray[H, W]): Polygon binary mask
        """
        # read shape files
        project_shape_collections = [fiona.open(p) for p in self.run['project_shapefiles']]
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
        rasterized_polygon = rasterio.features.rasterize(
            [(polygon, 1)],
            out_shape=gt_file.shape,
            transform=gt_file.transform,
            fill=0,
            dtype='uint8'
        )
        return rasterized_polygon, gt_date
    
    def _read_project_images(self, project_id: str):
        """ 
        DatasetPreprocessor._read_project_images method: collect images associated to a project id

        Args:
        - project_id (str): project id

        Returns:
        - images (List[np.ndarray]): list of all images ; Sentinel-2, then Sentinel-1 ascending, then Sentinel-1 descending
        - images (List[int]): list of image ids (useful later)
        - s2_images (List[Tuple[np.ndarray[13, H, W]]]): list of all Sentinel-2 images
        - s1_images_ascending (List[Tuple[np.ndarray[2, H, W]]]): list of all Sentinel-1 ascending images
        - s1_images_descending (List[Tuple[np.ndarray[2, H, W]]]): list of all Sentinel-1 descending images
        """
        s2_images = []
        s1_images_ascending = []
        s1_images_descending = []
        # read s2
        print(f"[debug:291] s2 reprojected dir: {os.path.abspath(os.path.join(self.run['s2_reprojected_dir'], str(project_id)))} (exists={os.path.exists(self.run['s2_reprojected_dir'])})")
        for img_path in pjoin(self.run['s2_reprojected_dir'], str(project_id)).glob('*.tif'):
            print(f"[debug:293] img_path: {os.path.abspath(img_path)} (exists={os.path.exists(img_path)})")
            with rasterio.open(os.path.abspath(img_path)) as fh:
                s2_images.append((fh.read(fh.indexes), parse_date(img_path.stem.split('_')[3].split('T')[0])))
        # read s1
        for img_path in pjoin(self.run['s1_reprojected_dir'], project_id).glob('*.tif'):
            with rasterio.open(os.path.abspath(img_path)) as fh:
                if img_path.stem.endswith('_A'):
                    s1_list = s1_images_ascending
                elif img_path.stem.endswith('_D'):
                    s1_list = s1_images_descending
                else:
                    raise ValueError(f'Could not extract orbit direction from filename: {img_path.name}')
                s1_list.append((fh.read(fh.indexes), parse_date(img_path.stem.split('_')[5].split('T')[0])))
        images = [img for img, _ in s2_images]+[img for img, _ in chain(s1_images_ascending, s1_images_descending)]
        image_ids = [id(img) for img in images]
        return images, image_ids, s2_images, s1_images_ascending, s1_images_descending
    
    def _make_project_patches(self, *args, **kwargs):
        """
        DatasetPreprocessor._make_project_patches method: selects and apply patch sampling strategy specified in configuration according 
            to `self.patches_sampling_strategy_map`.
        """
        return self.patches_sampling_strategy_map[self.run["sampling_strategy"]](*args, **kwargs)
    
    def _get_valid_center_patches(
        self,
        gt_file,
        project_id,
        split_mask,
        rasterized_polygon,
        valid_mask,
        s1_images_ascending,
        s1_images_descending,
        s2_images,
        images,
        image_ids,
        labels,
        gt_date,
    ): 
        """
        
        """
        # variables
        stats = {}
        locations = defaultdict(list)
        loc_to_images_map = defaultdict(list)
        offsets = defaultdict(list)
        shape = gt_file.shape
        num_images_per_pixel = np.zeros((1, *shape), dtype=np.uint8)
        patch_half = self.run['patch_size'] // 2
        for i in self.prange(patch_half, gt_file.shape[0] - patch_half):
        # for i in range(patch_half, gt_file.shape[0] - patch_half):
            # if i==10: break # DEBUG
            for j in range(patch_half, gt_file.shape[1] - patch_half):
                i_slice = slice(i - patch_half, i + patch_half + 1)
                j_slice = slice(j - patch_half, j + patch_half + 1)
                is_same_split = (split_mask[i_slice, j_slice] == split_mask[i_slice, j_slice][0, 0]).all()
                is_in_polygon = (rasterized_polygon[i_slice, j_slice] == 1).all()
                # Ignore pixels that span over multiple splits and don't lie completly in their polygon
                if is_same_split and is_in_polygon and valid_mask[i,j]:
                    dataset = self.split_map[int(split_mask[i,j])]
                    images_for_pixel: List[Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]] = []
                    s2_dates_used = set()
                    # filter out s1 images which contain a nodata pixel in the patch, i.e. images which do
                    # not fully cover the patch. We noticed that some s1 images have weird stripes with
                    # values close to (but not exactly) zero near the swath edge. Therefore we empirically
                    # set the threshold value to 8.
                    valid_ascending = [img for img in s1_images_ascending if (img[0][:, i_slice, j_slice] > 8.).all()]
                    valid_descending = [img for img in s1_images_descending if (img[0][:, i_slice, j_slice] > 8.).all()]
                    if len(valid_ascending) == 0 or len(valid_descending) == 0:
                        continue
                    for s2_image, s2_date in s2_images:
                        # do not add TEST images if the difference between s2_date and gt_date is greater than 
                        # the threshold `testset_max_days_delta` (if specified)
                        if dataset == "test" and self.run["testset_max_days_delta"] is not None and abs((gt_date-s2_date).days > self.run["testset_max_days_delta"]):
                            continue
                        # do not add image if an image of the same date has been added for this location before.
                        # this is the case e.g. for the overlap region between two adjacent S2 images, which is
                        # identical for both images and would result in duplicate data points.
                        if s2_date in s2_dates_used:
                            continue
                        # only add patch if there is no nodata pixel contained, where a nodata pixel is
                        # defined as having zeros across all channels.
                        if (s2_image[:, i_slice, j_slice] == 0.).all(0).any():
                            continue
                        # only add patch with less than 10% cloudy pixels, where a cloudy pixel is defined as
                        # having cloud probability > 10%.
                        if (s2_image[-1, i_slice, j_slice] > self.run['cloud_prob_threshold']).sum() \
                                / self.run['patch_size']**2 > self.run['cloudy_pixels_threshold']:
                            continue
                        # determine matching s1 image date(s). All S1 images within 15 days of the S2 image will be
                        # added (and sampled randomly from during training).
                        matching_ascending = [img for img, date in valid_ascending if
                                            abs((s2_date - date).days) <= 15]
                        matching_descending = [img for img, date in valid_descending if
                                            abs((s2_date - date).days) <= 15]
                        # add s2 and matching s1 images to list of available images for this location
                        if len(matching_ascending) and len(matching_descending):
                            images_for_pixel.append((s2_image, matching_ascending, matching_descending))
                            s2_dates_used.add(s2_date)
                    num_images_per_pixel[0, i, j] = len(images_for_pixel)
                    # a data point corresponds to one image coordinate, such that regions with higher number
                    # of available images are not oversampled during training. Only add if there's at least one image
                    # for that pixel.
                    if len(images_for_pixel):
                        data_point = (i, j, images_for_pixel, labels)
                        # transform `images_for_pixel` into contiguos numpy array where images are referenced based on
                        # their index in `images`
                        this_loc_to_images_map = []
                        for s2_image, s1_a_list, s1_d_list in images_for_pixel:
                            this_loc_to_images_map.extend(
                                [image_ids.index(id(s2_image)), self.separator]
                                + [image_ids.index(id(img)) for img in s1_a_list]
                                + [self.separator]
                                + [image_ids.index(id(img)) for img in s1_d_list]
                                + [self.separator]
                            )
                        # add sample stats to corresponding split
                        locations[dataset].append((i, j))
                        offsets[dataset].append(len(loc_to_images_map[dataset]))
                        loc_to_images_map[dataset].extend(this_loc_to_images_map)
        # save stats
        stats["num_images_per_pixel"] = {
            "mean": float(num_images_per_pixel.mean()), 
            "min": float(num_images_per_pixel.min()),
            "max": float(num_images_per_pixel.max()),
        }
        stats["num_train"] = len(locations["train"])
        stats["num_val"] = len(locations['val'])
        stats["num_test"] = len(locations['test'])
        # save pkl
        project_data, _ = self._save_project_patches(
            project_id,
            images,
            locations,
            loc_to_images_map,
            offsets,
            labels,
        )
        stats = self._compute_project_stats(
            project_id,
            stats,
            project_data,
        )
        self._save_image_density(gt_file, project_id, num_images_per_pixel)
        return stats

    def _get_s2_offset_valid_center_patches(
        self,
        gt_file,
        project_id,
        split_mask,
        rasterized_polygon,
        valid_mask,
        s1_images_ascending,
        s1_images_descending,
        s2_images,
        images,
        image_ids,
        labels,
        gt_date,
    ): 
        """
        
        """
        # variables
        stats = {}
        locations = defaultdict(list)
        loc_to_images_map = defaultdict(list)
        offsets = defaultdict(list)
        shape = gt_file.shape
        num_images_per_pixel = np.zeros((1, *shape), dtype=np.uint8)
        patch_half = self.run['patch_size'] // 2
        for i in self.prange(patch_half, gt_file.shape[0] - patch_half):
        # for i in range(patch_half, gt_file.shape[0] - patch_half):
            # if i==10: break # DEBUG
            for j in range(patch_half, gt_file.shape[1] - patch_half):
                i_slice = slice(i - patch_half, i + patch_half + 1)
                j_slice = slice(j - patch_half, j + patch_half + 1)
                # no is same split as only test!
                is_in_polygon = (rasterized_polygon[i_slice, j_slice] == 1).all()
                # Ignore pixels that span over multiple splits and lie completly in their polygon
                if is_in_polygon and valid_mask[i,j]:
                    images_for_pixel: List[Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]] = []
                    # filter out s1 images which contain a nodata pixel in the patch, i.e. images which do
                    # not fully cover the patch. We noticed that some s1 images have weird stripes with
                    # values close to (but not exactly) zero near the swath edge. Therefore we empirically
                    # set the threshold value to 8.
                    valid_ascending = [img for img in s1_images_ascending if (img[0][:, i_slice, j_slice] > 8.).all()]
                    valid_descending = [img for img in s1_images_descending if (img[0][:, i_slice, j_slice] > 8.).all()]
                    if len(valid_ascending) == 0 or len(valid_descending) == 0:
                        continue
                    for s2_image, s2_date in s2_images:
                        # only add patch if there is no nodata pixel contained, where a nodata pixel is
                        # defined as having zeros across all channels.
                        if (s2_image[:, i_slice, j_slice] == 0.).all(0).any():
                            continue
                        # Pick the closest s1 asc/desc images to form the input
                        matching_ascending = [list(sorted(valid_ascending, key=lambda x: abs(x[1]-s2_date)))[0][0]]
                        matching_descending = [list(sorted(valid_descending, key=lambda x: abs(x[1]-s2_date)))[0][0]]
                        # add s2 and matching s1 images to list of available images for this location
                        if len(matching_ascending) and len(matching_descending):
                            images_for_pixel.append((s2_image, matching_ascending, matching_descending))
                    num_images_per_pixel[0, i, j] = len(images_for_pixel)
                    # a data point corresponds to one image coordinate, such that regions with higher number
                    # of available images are not oversampled during training. Only add if there's at least one image
                    # for that pixel.
                    if len(images_for_pixel):
                        data_point = (i, j, images_for_pixel, labels)
                        # transform `images_for_pixel` into contiguos numpy array where images are referenced based on
                        # their index in `images`
                        this_loc_to_images_map = []
                        for s2_image, s1_a_list, s1_d_list in images_for_pixel:
                            this_loc_to_images_map.extend(
                                [image_ids.index(id(s2_image)), self.separator]
                                + [image_ids.index(id(img)) for img in s1_a_list]
                                + [self.separator]
                                + [image_ids.index(id(img)) for img in s1_d_list]
                                + [self.separator]
                            )
                        # add sample stats to corresponding split
                        locations["test"].append((i, j))
                        offsets["test"].append(len(loc_to_images_map["test"]))
                        loc_to_images_map["test"].extend(this_loc_to_images_map)
        # save stats
        stats["num_images_per_pixel"] = {
            "mean": float(num_images_per_pixel.mean()), 
            "min": float(num_images_per_pixel.min()),
            "max": float(num_images_per_pixel.max()),
        }
        stats["num_train"] = 0
        stats["num_val"] = 0
        stats["num_test"] = len(locations['test'])
        # save pkl
        project_data, _ = self._save_project_patches(
            project_id,
            images,
            locations,
            loc_to_images_map,
            offsets,
            labels,
        )
        stats = self._compute_project_stats(
            project_id,
            stats,
            project_data,
        )
        self._save_image_density(gt_file, project_id, num_images_per_pixel)
        return stats
    
    def _compute_project_stats(
        self,
        project_id,
        stats,
        data,
    ):
        patch_size = self.run["patch_size"]
        s2_stats = RunningStats((12,))
        s1_stats = RunningStats((2,))
        labels_stats = RunningStats((5,))
        print(f"Computing project {project_id} stats...")
        locations = data['train'][0]
        loc_to_images_map = data['train'][1]
        offsets = data['train'][2]
        images = data['images']
        labels = data['labels']
        for index in trange(len(locations)):
            i, j = locations[index]
            patch_half = patch_size // 2
            i_slice = slice(i - patch_half, i + patch_half + 1)
            j_slice = slice(j - patch_half, j + patch_half + 1)
            # extract the part from loc_to_images_map that contains the valid s2 and s1 indices for this location
            upper = None if index == len(locations) - 1 else offsets[index + 1]
            _map = loc_to_images_map[offsets[index]:upper]
            assert _map[0] != SEPARATOR and _map[-1] == SEPARATOR, f"{i}, {j}"
            _map = np.array(split_list(_map.tolist(), SEPARATOR), dtype='object')
            assert len(_map) % 3 == 0
            for s2_indices, s1_a_indices, s1_d_indices in np.array_split(_map, len(_map) // 3):
                assert len(s2_indices) == 1
                s2_index = s2_indices[0]
                s2_stats.add(images[s2_index][:12, i_slice, j_slice].reshape(-1, patch_size**2).transpose())
                for s1_index in chain(s1_a_indices, s1_d_indices):
                    s1_stats.add(images[s1_index][:, i_slice, j_slice].reshape(-1, patch_size**2).transpose())
            labels_patch = labels[:, i_slice, j_slice]
            labels_stats.add(labels_patch[:, ~np.isnan(labels_patch).any(0)].transpose())
        # update stats
        stats.update({
            "s2_stats": s2_stats.to_dict(),
            "s1_stats": s1_stats.to_dict(),
            "labels_stats": labels_stats.to_dict(),
        })
        return stats
    
    def _save_image_density(self, gt_file, project_id, num_images_per_pixel):
        with rasterio.Env():
            profile = gt_file.profile
            profile.update(
                driver='GTiff',
                count=1,
                compress='deflate',
                nodata=None,
                dtype='uint8'
            )
            with rasterio.open(pjoin(self.save_dir, f"num_images_per_pixel_{project_id}.tif"), "w", **profile) as f:
                f.write(num_images_per_pixel)
    
    def _make_dataset_info(
        self,
        project_id,
        gt_file,
        locations,
        split_mask,
        valid_mask,
        rasterized_polygon
    ):
        # Create valid center map
        shape = gt_file.shape
        valid_center_mask = np.zeros(shape, dtype=np.uint8)
        idx = []
        for split in locations.keys():
            idx.extend(locations[split])
        idx = np.array(idx)
        valid_center_mask[idx[:,0],idx[:,1]] = 1
        # create 2D info map
        dataset_info_map = split_mask.copy()
        dataset_info_map += 2
        dataset_info_map[valid_mask==0] = 1
        dataset_info_map[valid_center_mask==0] = 0
        dataset_info_map[rasterized_polygon==0] = -1 # np.nan
        # create tensor
        to3d = lambda x: x if len(x.shape)==3 else np.expand_dims(x, axis=0)
        raster = np.concatenate([to3d(a) for a in [
            dataset_info_map,
            valid_center_mask,
            split_mask,
            valid_mask
        ]], axis=0)
        # save geotiff
        with rasterio.Env():
            profile = gt_file.profile
            profile.update(
                driver='GTiff',
                count=raster.shape[0],
                compress='deflate',
                nodata=None,
                dtype='uint8'
            )
            with rasterio.open(pjoin(self.save_dir, f"info_map_{project_id}.tif"), "w", **profile) as f:
                f.write(raster)
                f.write_mask(rasterized_polygon.astype(bool)) # crop to rasterized_polygon boundaries
    
    def _save_project_patches(
            self,
            project_id,
            images,
            locations,
            loc_to_images_map,
            offsets,
            labels,
        ):
        path = pjoin(self.save_dir, f"{project_id}.pkl")
        print(f"Pickling project {project_id}: {str(path)}")
        data = {
            'images': images,
            'train': (
                np.array(locations['train'], dtype=np.uint16),
                np.array(loc_to_images_map['train'], dtype=np.uint16),
                np.array(offsets['train'], dtype=np.uint64)
            ),
            'val': (
                np.array(locations['val'], dtype=np.uint16),
                np.array(loc_to_images_map['val'], dtype=np.uint16),
                np.array(offsets['val'], dtype=np.uint64)
            ),
            'test': (
                np.array(locations['test'], dtype=np.uint16),
                np.array(loc_to_images_map['test'], dtype=np.uint16),
                np.array(offsets['test'], dtype=np.uint64)
            ),
            'labels': labels
        }
        with path.open("wb") as fh:
            pickle.dump(data, fh)
        print(f"Done pickling project {project_id}")
        return data, path
    
def configure(cfg_f, num_projects_per_job):
    """
    Given a blowtorch configuration and a number of jobs, configure each jobs (i.e generate job configurations
    and return them). Each job gets a list of projects to process.
    """
    # load main configuration
    with cfg_f.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # create save_dir
    save_dir = pjoin(cfg["save_dir"], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if cfg["name"] is not None:
        save_dir = Path(str(save_dir)+f"_{cfg['name']}")
    save_dir.mkdir(parents=True)
    cfg["save_dir"] = str(save_dir)
    # write config to save_dir
    with pjoin(save_dir, "data_config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False) 
    # confugre jobs
    projects = cfg["projects"]
    sub_cfg_paths = []
    sub_save_dirs = [0]
    for i in range(0, len(projects), num_projects_per_job):
        # select projects
        sub_projects = projects[i:i+num_projects_per_job]
        if len(sub_projects)==0: break
        cfg["projects"] = sub_projects
        # define subdirectory
        sub_save_dir = pjoin(save_dir, "-".join(sub_projects))
        sub_save_dir.mkdir(parents=False)
        cfg["save_dir"] = str(sub_save_dir)
        # write config
        sub_cfg_path = pjoin(sub_save_dir, cfg_f.name)
        with sub_cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False)
        sub_cfg_paths.append(sub_cfg_path)
    print(" ".join([str(p) for p in [pjoin(save_dir, "data_config.yaml")]+sub_cfg_paths]))

def preprocess(cfg_f):
    """
    Given a configuration file, preprocess a list of projects
    """
    run = blowtorch.run.Run(config_files=[cfg_f])
    ppp = ProjectsPreprocessor(run)
    ppp()
    
def aggregate(cfg_f):
    """
    Given a path to a directory of processed projects, compute aggreagated statistics
    """
    import sys, os 
    root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, root)
    from utils import split_list, RunningStats
    # load main configuration
    with cfg_f.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # load save_dir
    save_dir = Path(cfg["save_dir"])
    # iterate over sub_directories
    stats = {"num_train": 0, "num_val": 0, "num_test": 0}
    accum_keys = list(stats.keys()).copy()
    s2_cstats = CombinedStats((12,))
    s1_cstats = CombinedStats((2,))
    labels_cstats = CombinedStats((5,))
    print("Starting projects aggregation.")
    for subdir in tqdm(list(save_dir.iterdir())):
        if os.path.isdir(subdir):
            print(f"Aggregating: {subdir.name}")
            # 1. copy pkl files
            print("Copying pkl files ...")
            for pkl_file in subdir.glob("*.pkl"):
                dst = pjoin(save_dir, pkl_file.name)
                shutil.copyfile(pkl_file, dst)
            # 2. copy tif files
            print("Copying tif files ...")
            for tif_file in subdir.glob("*.tif"):
                dst = pjoin(save_dir, tif_file.name)
                shutil.copyfile(tif_file, dst)
            # 3. combine stats
            print("Combining stats ...")
            with pjoin(subdir, "stats.yaml").open("r") as f:
                sub_stats = yaml.safe_load(f)
            projects = list(sub_stats.keys()).copy()
            for p in projects:
                ## 3.1. copy project data
                stats[p] = sub_stats[p]
                ## 3.2. update accumulators
                for key in accum_keys:
                    stats[key] += sub_stats[p][key]
                ## 3.3. update moments
                s2_cstats.add(sub_stats[p]["s2_stats"])
                s1_cstats.add(sub_stats[p]["s1_stats"])
                labels_cstats.add(sub_stats[p]["labels_stats"])
            shutil.rmtree(subdir)
            print("Done.")
    stats.update({
        "s2_stats": s2_cstats.to_dict(),
        "s1_stats": s1_cstats.to_dict(),
        "labels_stats": labels_cstats.to_dict(),
    })
    print("Writing aggregated stats ...")
    with pjoin(save_dir, "stats.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(stats, f, sort_keys=False)
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # generic argument
    p.add_argument("--cfg", help="path to config file (yaml)", type=_path)
    # configuration args
    p.add_argument("--configure", help="set to configure mode", action="store_true")
    p.add_argument("--num_projects_per_job", help="number of jobs", type=int)
    # run args
    p.add_argument("--preprocess", help="set to compute mode", action="store_true")
    # aggregation args
    p.add_argument("--aggregate", help="set to aggregate mode", action="store_true")
    args = p.parse_args()
    if args.configure:
        assert args.num_projects_per_job, "--num_projects_per_job must be set in configuration mode"
        configure(args.cfg, args.num_projects_per_job)
    elif args.preprocess:
        preprocess(args.cfg)
    elif args.aggregate:
        aggregate(args.cfg)
    else: raise AttributeError("Must be set either to configuration (--configure) or aggregatation (--aggregate) mode.")
    sys.exit(0)