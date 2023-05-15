import ee
from datetime import datetime
from tqdm import tqdm
import time
from gdrive_handler import GDriveV3Wrapper

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

    def get_image_collection(
        self,
        polygon: list,
        gt_date: datetime,
        date_offset_amount: int, 
        date_offset_unit: str="day",
        date_offset_policy: str="both", # (before, after, both)
        collection_name: str="COPERNICUS/S2_SR_HARMONIZED",
        s2_bands: list=['B1','B2','B3','B4','B5','B6','B7','B8','B8A', 'B9', 'B11','B12','MSK_CLDPRB'],
        mosaic: bool=False
    ):
        # 1. Define Polygon
        self.verbose_print(f"Creating polygon from {len(polygon[0])} edges...")
        aoi = ee.Geometry.Polygon(polygon, proj=ee.Projection(self.crs))
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
        return icol

    def merge_image_collection_by_date(self, imgCol):
        '''
        function that merges images together that have the same date. 
        adapted from: https://gis.stackexchange.com/questions/372744/how-to-create-a-loop-which-creates-single-mosaic-images-for-each-day-from-a-sel
        '''    
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
    
    def copy_drive(self, tasks, drivefolder, localdir):
        def get_downloadable_tasks(tasks):
            ready = set()
            tasklist = ee.data.getTaskList()
            for task in tasklist:
                # filter tasks
                if not task["description"] in [t["config"]["description"] for t in tasks]: continue
                # ready/not ready
                print(f'[d:104] task state: {task["state"]}')
                if task["state"] == "COMPLETED": ready.add(task["description"])
            return list(ready)
        def copy_and_delete(drive, drivefolder, localdir, names_list=None):
            successes = []
            for file in drive.listFile(folderName=drivefolder):
                fileId, fileName = file.get("id"), file.get("name")
                imageName = fileName.split(".")[0]
                if names_list is not None and imageName not in names_list: continue
                self.verbose_print(f"Found match on drive: fileName={fileName}, fileId={fileId}")
                try:
                    self.drive.downloadFile(localdir=localdir, fileId=fileId, fileName=fileName)
                    try:
                        self.drive.deleteFile(fileId=fileId)
                        successes.append(imageName)
                    except Exception as e:
                        print(f"Could not delete file with id {fileId}.")
                        raise e
                except Exception as e:
                    print(f"Download of file with id {fileId} failed.")
                    print(e)
            return successes
        while len(tasks)>0:
            time.sleep(5)
            to_download = get_downloadable_tasks(tasks)
            self.verbose_print(f'Remaining tasks: {[task["config"]["description"] for task in tasks]}, Downlodable tasks: {to_download}')
            success = copy_and_delete(self.drive, drivefolder, localdir, to_download)
            self.verbose_print(f"Successfully downloaded: {success}")
            tasks = [task for task in tasks if task["config"]["description"] not in to_download+success]

    def cast(self, image, dtype):    
        return {
            "float":image.toFloat(), 
            "byte":image.toByte(), 
            "int":image.toInt(),
            "double":image.toDouble(),
            "long": image.toLong(),
            "short": image.toShort(),
            "int8": image.toInt8(),
            "int16": image.toInt16(),
            "int32": image.toInt32(),
            "int64": image.toInt64(),
            "uint8": image.toUint8(),
            "uint16": image.toUint16(),
            "uint32": image.toUint32()
        }[dtype]

    def download_image(self, image, fn_prefix, drivefolder, polygon, dtype, scale):
        projection = ee.Projection(self.crs)
        image = self.cast(ee.Image(image), dtype)
        fn = fn_prefix+"{}".format(image.id().getInfo().split("_")[0])
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=fn,
            fileNamePrefix=fn,
            region=polygon,
            crs=projection.getInfo()["crs"],
            crs_transform=projection.getInfo()["transform"],
            folder=drivefolder,
            fileFormat="GeoTIFF",
            scale=scale
        )
        task.start()
        self.verbose_print(f"Submitting image {fn}, taskId={task.id}...")
        return vars(task)

    def download_image_collection(self, icol, fn_prefix, drivefolder, batch, polygon, dtype, scale, localdir): 
        # Transform to list
        n = icol.size().getInfo()
        ilist = icol.toList(n)
        # Task start loop
        tasks = []
        for i in range(n):
            task = self.download_image(ilist.get(i), fn_prefix, drivefolder, polygon, dtype, scale)
            tasks.append(task)
            # sequential download
            if not batch:
                self.copy_drive([task], drivefolder, localdir)
            break
        # parallel download and/or make sure all downloaded
        self.copy_drive(tasks, drivefolder, localdir)

    def download_timeserie(
        self, 
        localdir: str,
        project_id: str,
        polygon: list,
        gt_date: datetime,
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
        verbose: bool=False,
    ):
        assert date_offset_policy in ["before", "after", "both"], f'Invalid date_offset_policy: {date_offset_policy}. Value must be in {["before", "after", "both"]}.'
        assert date_offset_amount >= 0, f'date_offset_amount must be positive'
        assert agg in [None, "mean", "median", "cloudless_mean", "cloudless_median"], f'Invalid agg: {agg}. Value must be in {[None, "mean", "median"]}.'
        assert dtype in ['float', 'byte', 'int', 'double', 'long', 'short', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32'], \
            f"Invalid dtype: {dtype}. Value must be in {['float', 'byte', 'int', 'double', 'long', 'short', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32']}."

        self.verbose_print(f"Downloading images for project {project_id}")

        # args
        fn_prefix = f'{project_id}-GEE_{collection_name.replace("/", "_")}-'
        
        # Create ImageCollection
        icol = self.get_image_collection(
            polygon,
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
            self.download_image(img, fn_prefix, drivefolder, polygon, dtype, scale)
        
        # Download image collection
        else:
            self.download_image_collection(icol, fn_prefix, drivefolder, batch, polygon, dtype, scale, localdir)