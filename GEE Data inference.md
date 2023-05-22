# GEE Data inference

This document presents the strategy to infer forest structure on unseen Sentinel-2 data.

## Setting

The GEE Sentinel-2 data being warped from the GT rasters, we get two coordinate systems:

1. $\mathcal{R}:\;(i,j)\in[0,H]\times[0,W]$ applied to all original data, i.e ground truth, valid mask, split_mask, Sentinel-1
2. $\mathcal{R'}:\;(i,j)\in[0,H']\times[0,W']$ applied to unseen GEE Sentinel-2 data

There are 2 ways to deal with this:

1. Crop all data and save to disk: 
   - Pros: No change in the dataset generation code if done properly
   - Cons: Disk usage
2. Define additional valid pixel conditions based on the offset between $\mathcal{R}$ and $\mathcal{R}'$
   - Pros: Reduced disk usage, adaptability
   - Cons: Changes to dataset generation code necessary

It seems that option 2 is more suitable. The required changes in the code boil down to an additional valid pixel condition to ensure that the translated positions fall in the raster bounds.

## Data 

The data sources are:

- `gt_dir`: directory holding ground truth and valid mask rasters (`assets/data/preprocessed`)
- `split_mask_dir`: defines the set split (train, validation, test). This is still necessary as the Sentinel-1 rasters have been seen before (`assets/data/split_masks`)
- `s1_repojected_dir`: directory containing the reprojected Sentinel-1 rasters (`assets/data/sentinel_data/s1_reprojected`)
- `s2_gee_reprojected_dir`:  directory containing the REPROJECTED GEE Sentinel-2 rasters (`gee_s2_reprojected`). Produced by `gee/download_timeseries.py`

## GEE Download

Specs:

- Must create a directory `${dirname}_reprojected/${project_id}/${filename}.tif` containing ONLY reprojected rasters
- filename must be in format: `$PID_GEE_COPERNICUS-S2-SR-HARMONIZED_$IMAGEDT_$NOWDT.tif` with date time format `%Y%m%dT%H%M%S`so that we can parse the dates the same way as original data.

## Patch extraction strategy

We propose to add a sampling strategy to `src/scripts/create_dataset.py` for the special case of GEE Sentinel-2 data with the following specifications:

```
for i, j in R: # loop on reference coordinate system
	compute is_same_split # as valid_center
	compute is_in_polygon # as valid_center
	if is_same_split and is_in_polygon and valid_mask[i,j]:
		get valid ascending and descending S1 images # as valid_center
		for s2_image, s2_date in s2_images: # loop on GEE s2 images
			discard no_data_pixels # as valid_center
			compute offsets di, dj
			compute is_valid_offset # max(0,di)<=i<=min(H,H+di) and max(0,dj)<=j<=min(W,W+dj)
			if is_valid_offset:
				ascending = [closest s1_asc date] # as list to not have to change anything	
				descending = [closest s1_desc date]
```

Note:

1. Rasterio to GDal datasets

   ```python
   rio_file # rasterio dataset
   gdal_file = gdal.Open(rio_file.name) # gdal dataset
   ```

