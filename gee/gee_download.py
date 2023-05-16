import ee, os
import geetools
from geetools.utils import makeName
from geetools import tools
from geetools.batch import utils
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
from osgeo import gdal
from gdrive_handler import GDriveV3Wrapper

def downloadGEETasks(drive, tasks, filenames, drive_folder, local_folder, verbose=False):
    paths = []
    while len(tasks)>0:
        time.sleep(2)
        if verbose: print(f"Remaining tasks: {len(tasks)}, Remaining files: {len(filenames)}, Downloaded tasks: {len(paths)}")
        try: driveFiles = drive.listFile(folderName=drive_folder)
        except Exception as e: 
            print(e)
            continue
        # print(f"Files on drive: {[f.get('id') for f in driveFiles]}")
        for file in driveFiles:
            fileId, filename = file.get("id"), file.get("name")
            filestem = Path(filename).stem
            # print(f"current file: stem={filestem}, name={filename}")
            # print(f"taget stems: {filenames}")
            # print("should match:", filestem in filenames)
            if filestem in filenames:
                index = filenames.index(filestem)
                task = tasks[index]
                # print(f"[d:23] found match: filename={filename}, fileId={fileId}, taskId={task.id}")
                try:
                    # print(f"Downloading and deleting {drive_folder}/{filename} from drive")
                    path = drive.downloadFile(fileId=fileId, localdir=local_folder, fileName=filename)
                    try:
                        drive.deleteFile(fileId)
                        tasks.remove(task)
                        filenames.remove(filestem)
                        paths.append(path)
                    except Exception as e:
                        print(f"Could not delete file: fileId={fileId}, filename={filename}")
                        print(e)
                except Exception as e:
                    print(f"Could not download task: taskId={task.id}, filename={filename}")
                    print(e)
    return paths

def exportImageCollectionToDrive(
    collection, 
    fn_prefix, 
    folder, 
    scale, 
    datatype, 
    region, 
    crs, 
    verbose=False, 
    **kwargs
):
    """
    Slight modification of geetools.batch.Export.imagecollection.toDrive to use a filename prefix 
    to define the name property when exporting images
    """
    # compat
    namePattern='{id}'
    datePattern=None
    extra=None
    dataType=datatype
    # empty tasks list
    tasklist, filenames = [], []
    # get region
    if region:
        region = tools.geometry.getRegion(region)
    # Make a list of images
    img_list = collection.toList(collection.size())
    n = 0
    while True:
        try:
            img = ee.Image(img_list.get(n))
            name = fn_prefix+"{}-{}".format(
                img.id().getInfo().split("_")[0], 
                datetime.now().strftime("%Y%d%mT%H%M%S"))
            description = utils.matchDescription(makeName(img, namePattern, datePattern, extra).getInfo())
            # convert data type
            img = utils.convertDataType(dataType)(img)
            if region is None:
                region = tools.geometry.getRegion(img)
            task = ee.batch.Export.image.toDrive(image=img,
                                                 description=description,
                                                 folder=folder,
                                                 fileNamePrefix=name,
                                                 region=region,
                                                 scale=scale,
                                                 crs=crs,
                                                 **kwargs)
            task.start()
            if verbose: print(f"Submitted new task: taskId={task.id}, name={name}, description={description}")
            tasklist.append(task)
            filenames.append(name)
            n += 1
        except Exception as e:
            error = str(e).split(':')
            if error[0] == 'List.get': 
                if verbose: print(f"Reached end of image list at index: {n}")
                break
            else: raise e
    return tasklist, filenames

class GEELocalDownloader:
    def __init__(
        self,
        # google drive parameters
        token_file: str,
        credentials_file: str,
        gdrive_scopes: list,
        # GEE parameters
        gee_project: str,
        crs: str="EPSG:4326", # reference crs
        # other
        verbose: bool=False
    ):
        ee.Initialize(project=gee_project)
        self.drive = GDriveV3Wrapper(token_file, credentials_file, gdrive_scopes, verbose)
        self.crs = crs
        self.gee_project = gee_project
        self.verbose = verbose

    def verbose_print(self, msg):
        if self.verbose: print(msg)

    def merge_image_collection_by_date(self, imgCol):
        '''
        function that merges images together that have the same date. 
        adapted from: https://gis.stackexchange.com/questions/372744/how-to-create-a-loop-which-creates-single-mosaic-images-for-each-day-from-a-sel
        '''    
        self.verbose_print(f"Creating mosaic...")
        #Convert the image collection to a list.
        imgList = imgCol.toList(imgCol.size())        
        # Driver function for mapping the unique dates
        def uniqueDriver(image):
            return ee.Image(image).date().format("YYYY-MM-dd")        
        uniqueDates = imgList.map(uniqueDriver).distinct()
        # Driver function for mapping the moasiacs
        def mosaicDriver(date):
            date = ee.Date(date)            
            image = (imgCol
                .filterDate(date, date.advance(1, "day")) #or (date.advance(-1, "day"), date.advance(1, "day"))?
                .mosaic())            
            return image.set("system:time_start", date.millis(), 
                            "system:date", date.format("YYYY-MM-dd"),
                            "system:id", date.format("YYYY-MM-dd"))        
        mosaicImgList = uniqueDates.map(mosaicDriver)        
        return ee.ImageCollection(mosaicImgList)

    def aggregate_image_collection(self, icol, agg):
        self.verbose_print(f"Aggregating image collection")
        if "cloudless" in agg:
            # TODO: cloud masking
            raise NotImplementedError()
        agg_img = eval(f"icol.{agg}()")
        return agg_img
    
    # def copy_drive(self, tasks, filenames, drivefolder, localdir):
    #     def get_downloadable_tasks(tasks):
    #         tasks_ids = [t.id for t in tasks]
    #         ready = set()
    #         tasklist = ee.data.getTaskList()
    #         for task in tasklist:
    #             if task.id in tasks_ids: ready.add(tasks_ids.index(task.id))
    #         return list(ready)
    #     def copy_and_delete(drive, drivefolder, localdir, filenames):
    #         successes = []
    #         for file in drive.listFile(folderName=drivefolder):
    #             fileId, fileName = file.get("id"), file.get("name")
    #             imageName = fileName.split(".")[0]
    #             if names_list is not None and imageName not in names_list: continue
    #             self.verbose_print(f"Found match on drive: fileName={fileName}, fileId={fileId}")
    #             try:
    #                 self.drive.downloadFile(localdir=localdir, fileId=fileId, fileName=fileName)
    #                 try:
    #                     self.drive.deleteFile(fileId=fileId)
    #                     successes.append(imageName)
    #                 except Exception as e:
    #                     print(f"Could not delete file with id {fileId}.")
    #                     print(e)
    #             except Exception as e:
    #                 print(f"Download of file with id {fileId} failed.")
    #                 print(e)
    #         return successes
    #     paths = []
    #     while len(tasks)>0:
    #         time.sleep(5)
    #         downloadable_ids = get_downloadable_tasks(tasks)
    #         downloadable_filenames = [filenames[i] for i in downloadable_ids]
    #         self.verbose_print(f'Remaining tasks: {len(tasks)}, Downloadable tasks: {len(downloadable_ids)}')
    #         success = copy_and_delete(self.drive, drivefolder, localdir, downloadable_filenames)
    #         tasks = [task for task in tasks if task["config"]["description"] not in success]
    #         paths.extend(list(set([os.path.join(localdir, s+".tif") for s in success])))
    #     return paths

    # def download_image(self, image, fn_prefix, drivefolder, geometry, dtype, scale):
    #     projection = ee.Projection(self.crs)
    #     image = self.cast(ee.Image(image), dtype)#.clip(geometry)
    #     fn = fn_prefix+"{}-{}".format(image.id().getInfo().split("_")[0], datetime.now().strftime("%Y%d%mT%H%M%S"))# +"_withClip" # DEBUG
    #     task = ee.batch.Export.image.toDrive(
    #         image=image,
    #         description=fn,
    #         fileNamePrefix=fn,
    ##         region=geometry,
    #         crs=projection.getInfo()["crs"],
    #         crs_transform=projection.getInfo()["transform"],
    #         folder=drivefolder,
    #         fileFormat="GeoTIFF",
    #         scale=scale
    #     )
    #     task.start()
    #     self.verbose_print(f"Submitting image {fn}, taskId={task.id}")
    #     return vars(task)

    def reproject(self, paths, gt_path):
        def _reproject(input_path, gt_path):
            input_path = Path(input_path)
            output_path = Path(str(input_path).replace(input_path.stem, input_path.stem+"_reprojected"))
            with gdal.Open(gt_path) as gt_file:
                proj_ = gt_file.GetProjectionRef()
                warp_opts = gdal.WarpOptions(
                    format="GTiff",
                    dstSRS=proj_,
                    xRes=10.0, yRes=-10.0)
                x_ds = gdal.Warp(output_path, input_path, options=warp_opts)
                x_ds = None 
        for path in paths: 
            self.verbose_print(f"Reprojecting {path}")
            _reproject(path, gt_path)

    def get_image_collection(
        self,
        bbox: list,
        gt_date: datetime,
        date_offset_amount: int, 
        date_offset_unit: str="day",
        date_offset_policy: str="both", # (before, after, both)
        collection_name: str="COPERNICUS/S2_SR_HARMONIZED",
        s2_bands: list=['B1','B2','B3','B4','B5','B6','B7','B8','B8A', 'B9', 'B11','B12','MSK_CLDPRB'],
        mosaic: bool=False
    ):
        # 1. Define Polygon
        self.verbose_print(f"Creating rectangle from {bbox}...")
        aoi = ee.Geometry.Rectangle(bbox, proj=ee.Projection(self.crs))
        # 2. Define dates
        gt_date = ee.Date(gt_date)
        if date_offset_policy == "before":
            start_date = gt_date.advance(-date_offset_amount, date_offset_unit)
            end_date = gt_date
        elif date_offset_policy == "after":
            start_date = gt_date
            end_date = gt_date.advance(date_offset_amount, date_offset_unit)
        else:
            start_date = gt_date.advance(-date_offset_amount, date_offset_unit)
            end_date = gt_date.advance(date_offset_amount, date_offset_unit)
        self.verbose_print(f"Date filter: {start_date.format('dd.MM.yyyy').getInfo()} - {end_date.format('dd-MM-yyyy').getInfo()}")
        # 3. Get image collection
        icol = (ee.ImageCollection(collection_name)
             .filterBounds(aoi)
             .filterDate(start_date, end_date)
             .select(s2_bands))
        if mosaic:
            icol = self.merge_image_collection_by_date(icol)
        self.verbose_print(f"Found {icol.size().getInfo()} images")
        return icol, aoi

    def download_timeserie(
        self, 
        localdir: str,
        project_id: str,
        bbox: list,
        gt_date: datetime,
        gt_path: str,
        reproject_to_gt_crs: bool,
        date_offset_amount: int, 
        date_offset_unit: str="day",
        date_offset_policy: str="both", # (before, after, both)
        collection_name: str="COPERNICUS/S2_SR_HARMONIZED",
        s2_bands: list=['B1','B2','B3','B4','B5','B6','B7','B8','B8A', 'B9', 'B11','B12','MSK_CLDPRB'],
        drivefolder: str="geeExports",
        batch: bool=False, # if false, sequential dl o/w parallel
        agg: str=None, # if None, raw images, o/w compute agg
        mosaic: bool=False, # if True mosaic same dates
        dtype: str="uint16",
        scale: int=10,
    ):
        assert date_offset_policy in ["before", "after", "both"], f'Invalid date_offset_policy: {date_offset_policy}. Value must be in {["before", "after", "both"]}.'
        assert date_offset_amount >= 0, f'date_offset_amount must be positive'
        assert agg in [None, "mean", "median", "cloudless_mean", "cloudless_median"], f'Invalid agg: {agg}. Value must be in {[None, "mean", "median"]}.'
        assert dtype in ['float', 'byte', 'int', 'double', 'long', 'short', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32'], \
            f"Invalid dtype: {dtype}. Value must be in {['float', 'byte', 'int', 'double', 'long', 'short', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32']}."

        self.verbose_print(f"Downloading images for project {project_id}")

        # args
        fn_prefix = f'{project_id}-GEE_{collection_name.replace("/", "_")}-'
        print(fn_prefix)
        # Create ImageCollection
        icol, geometry = self.get_image_collection(
            bbox,
            gt_date,
            date_offset_amount,
            date_offset_unit,
            date_offset_policy,
            collection_name,
            s2_bands,
            mosaic
        )
        # Apply aggregation
        if agg is not None:
            img = self.aggregate_image_collection(icol, agg)
            task = self.download_image(img, fn_prefix, drivefolder, geometry, dtype, scale)
            paths = self.copy_drive([task], drivefolder, localdir)
        
        # Download image collection
        else:
            fn_prefix = f'{project_id}-GEE_{collection_name.replace("/", "_")}-'
            tasks, filenames = exportImageCollectionToDrive(
                collection=icol,
                region=geometry,
                folder=drivefolder,
                scale=scale,
                fn_prefix=fn_prefix,
                datatype=dtype,
                crs=self.crs,
                verbose=self.verbose
            )
            paths = downloadGEETasks(self.drive, tasks, filenames, drivefolder, localdir, verbose=self.verbose)

        # Reproject
        if reproject_to_gt_crs: self.reproject(paths, gt_path)