from typing import *
from pathlib import Path
from datetime import datetime
from itertools import chain
from collections import defaultdict
import pickle
import yaml
import blowtorch
from blowtorch import Run
from tqdm import trange
import numpy as np
import fiona
import os
import rasterio.warp
import rasterio.features
import matplotlib
import matplotlib.pyplot as plt

def parse_date(date_str) -> datetime: return datetime.strptime(date_str, '%Y%m%d')

def pjoin(*subs: List[Union[str,Path]]) -> Path: return Path(os.path.join(*subs))

def defaultdict2dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict2dict(v) for k, v in d.items()}
    return d

class DatasetCreator:
    def __init__(
        self,
        run: blowtorch.run.Run,
        seed: int=12345,
        separator: int=65535,
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
            "valid_center": self._get_valid_center_patches
        }
        self.dataset_stats = defaultdict()
        # verify run config
        self._validate_run_config()
        # set all random seeds
        self.run.seed_all(self.seed)
        self.dataset_stats["seed"] = self.seed
        # save directory
        self.save_dir = pjoin(self.run["save_dir"], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.save_dir.mkdir(parents=True)
        # create dataset
        projects = self.run["projects"].copy()
        for gt_file_path in Path(self.run["gt_dir"]).glob("*.tif"):
            if gt_file_path.stem in projects:
                projects.remove(gt_file_path.stem)
                project_dataset_stats = self._create_project_dataset(gt_file_path)
                self.dataset_stats[gt_file_path.stem] = project_dataset_stats
                if len(projects) == 0: break
        # save stats and config
        with pjoin(self.save_dir, "stats.yaml").open("w") as fh:
            yaml.dump(defaultdict2dict(self.dataset_stats), fh)
        with pjoin(self.save_dir, "data_config.yaml").open("w") as fh:
            yaml.dump(self.run.get_raw_config(), fh)

    def _create_project_dataset(self, gt_file_path) -> None:
        # get project id
        project_id = gt_file_path.stem
        if self.verbose: print(f"Processing project id {project_id}")
        # initialize stats dict
        project_dataset_stats = defaultdict()
        project_dataset_stats["num_train"] = 0
        project_dataset_stats["num_val"] = 0
        project_dataset_stats["num_test"] = 0
        # read gt
        gt_file, valid_mask, labels = self._read_gt_file(gt_file_path)
        project_dataset_stats["shape"] = list(valid_mask.shape)
        # read split_mask
        split_mask = self._read_split_mask(pjoin(self.run["split_mask_dir"], project_id+".tif"))
        # get rasterized polygon
        rasterized_polygon, gt_date = self._get_polygon_raster(
            self.run["project_shapefiles"],
            project_id, gt_file
        )
        project_dataset_stats["gt_date"] = gt_date
        # read images
        images, image_ids, s2_images, s1_images_ascending, s1_images_descending = self._read_project_images(project_id)
        project_dataset_stats["s2_dates"] = [d for _, d in s2_images]
        project_dataset_stats["s1_ascending_dates"] = [d for _, d in s1_images_ascending]
        project_dataset_stats["s1_descending_dates"] = [d for _, d in s1_images_descending]
        project_dataset_stats["num_s2_images"] = len(s2_images)
        project_dataset_stats["num_s1_images"] = len(s1_images_ascending)
        project_dataset_stats["num_s1_images"] = len(s1_images_descending)
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
        )
        # copy stats
        for k, v in patches_stats.items():
            project_dataset_stats[k] = v
        return project_dataset_stats

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
    
    def _read_split_mask(self, path):
        """ 
        DatasetCreateor._read_split_mask method: read and return split mask
        
        Args:
        - path (Union[str, pathlib.Path]): path to split mask file

        Returns:
        - split_mask (np.ndarray[H, W]): Cross-validation split mask, `0->train`, `1->val`, `2->test`.
        """
        with rasterio.open(path) as f: split_mask = f.read(1).astype('float16')
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
    
    def _read_project_images(self, project_id: str):
        """ 
        DatasetCreator._read_project_images method: collect images associated to a project id

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
        for img_path in pjoin(self.run['s2_reprojected_dir'], project_id).glob('*.tif'):
            with rasterio.open(img_path) as fh:
                s2_images.append((fh.read(fh.indexes), parse_date(img_path.stem.split('_')[3].split('T')[0])))
        # read s1
        for img_path in pjoin(self.run['s1_reprojected_dir'], project_id).glob('*.tif'):
            with rasterio.open(img_path) as fh:
                if img_path.stem.endswith('_A'):
                    s1_list = s1_images_ascending
                elif img_path.stem.endswith('_D'):
                    s1_list = s1_images_descending
                else:
                    raise ValueError(f'Could not extract orbit direction from filename: {img_path.name}')
                s1_list.append((fh.read(fh.indexes), parse_date(img_path.stem.split('_')[5].split('T')[0])))
        images = [img for img, _ in chain(s2_images, s1_images_ascending, s1_images_descending)]
        image_ids = [id(img) for img in images]
        return images, image_ids, s2_images, s1_images_ascending, s1_images_descending
    
    def _make_project_patches(self, *args, **kwargs):
        """
        DatasetCreator._make_project_patches method: selects and apply patch sampling strategy specified in configuration according 
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
    ): 
        """
        
        """
        # variables
        stats = defaultdict()
        locations = defaultdict(list)
        loc_to_images_map = defaultdict(list)
        offsets = defaultdict(list)
        shape = gt_file.shape
        num_images_per_pixel = np.zeros((1, *shape), dtype=np.uint8)
        patch_half = self.run['patch_size'] // 2
        for i in trange(patch_half, gt_file.shape[0] - patch_half):
            # if i==10: break # DEBUG
            for j in range(patch_half, gt_file.shape[1] - patch_half):
                i_slice = slice(i - patch_half, i + patch_half + 1)
                j_slice = slice(j - patch_half, j + patch_half + 1)
                is_same_split = (split_mask[i_slice, j_slice] == split_mask[i_slice, j_slice][0, 0]).all()
                is_in_polygon = (rasterized_polygon[i_slice, j_slice] == 1).all()
                # Ignore pixels that span over multiple splits and lie completly in their polygon
                if is_same_split and is_in_polygon and valid_mask[i,j]:
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
                        dataset = self.split_map[int(split_mask[i,j])]
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
        _ = self._save_project_patches(
            project_id,
            images,
            locations,
            loc_to_images_map,
            offsets,
            labels,
        )
        self._make_dataset_info(
            project_id,
            gt_file,
            locations,
            split_mask,
            valid_mask,
            rasterized_polygon
        )
        return stats
    
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
        for split in locations.keys():
            idx = np.array(locations[split])
            valid_center_mask[idx[:,0],idx[:,1]] = 1                
        # create 2D info map
        dataset_info_map = split_mask.copy()
        dataset_info_map += 1
        dataset_info_map[valid_mask==0] = 1
        dataset_info_map[valid_center_mask==0] = 1
        dataset_info_map[rasterized_polygon==0] = np.nan 
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
                dtype='float'
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
        with path.open("wb") as fh:
            pickle.dump({
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
            }, fh)
        return path

    def plot(self, project_id, ax=None):
        rasterfile = rasterio.open(pjoin(self.save_dir, f"info_map_{project_id}.tif"))
        imap = rasterfile.read(1).astype(np.float16)
        if ax: ax.imshow(imap, cmap=matplotlib.colors.ListedColormap(["black", "yellow", "blue", "red"])); return ax
        else: plt.imshow(imap, cmap=matplotlib.colors.ListedColormap(["black", "yellow", "blue", "red"]))

if __name__=="__main__":
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    cfg_file = sys.argv[1] if len(sys.argv)>2 and sys.argv[1]!="" else os.path.join(root, "config", "create_dataset-dev.yaml")
    run = blowtorch.run.Run(config_files=[cfg_file])
    dataset = DatasetCreator(run)
    dataset.plot("346")
