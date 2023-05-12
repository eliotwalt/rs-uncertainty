import ee
from datetime import datetime
from tqdm import tqdm
import time
from gdrive_handler import GoogleDriveHandler

class GoogleEarthEngineLocalDownloader:
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
        self.drive = GoogleDriveHandler(token_file, credentials_file, gdrive_scopes, verbose)
        self.crs = crs
        self.gee_project = gee_project
        self.verbose = verbose 

    def verbose_print(self, msg):
        if self.verbose: print(msg)

    def get_image_collection(
        self,
        polygon: list,
        crs: str, # polygon crs
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
        proj = ee.Projection(crs)
        aoi = (ee.Geometry.Polygon(polygon, proj=proj)
                .transform(ee.Projection(self.crs)))
        self.verbose_print(f"CRS: original={crs}, reprojected={self.crs}")
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
            icol = mergeByDate(icol)
        # self.verbose_print(f"Found {icol.size().getInfo()} images")
        return icol

    def merge_image_collection_by_date(self, icol):
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
    
    def copy_drive(self, drivefolder, localdir, start):
        def still_going(start):
            tasklist = ee.data.getTaskList()
            no_of_tasks = 0 #number of tasks related to the current identifier
            for task in tasklist: 
                taskdate = datetime.fromtimestamp(task['creation_timestamp_ms']/1000.0)
                if taskdate >= now: 
                    no_of_tasks += 1
            for task in range(no_of_tasks): 
                if tasklist[task]['state'].__eq__('UNSUBMITTED') or tasklist[task]['state'].__eq__('READY') or tasklist[task]['state'].__eq__('RUNNING'):
                    return True
                else: 
                    continue
        def copy_all(drive, drivefolder, localdir): # downloadasgo
            for file in drive.ListFile(folderName=drivefolder):
                fileId = file.get("id")
                try:
                    self.drive.download_file(localdir, fileId=fileId)
                    try:
                        self.delete_file(fileId=fileId)
                    except Exception as e:
                        print(f"Could not delete file with id {fileId}.")
                        print(e)
                except Exception as e:
                    print(f"Download of file with id {fileId} failed.")
                    print(e)
        while still_going(start):
            time.sleep(1)
            copy_all(self.drive, drivefolder, localdir)
        copy_all(self.drive, drivefolder, localdir)

    def download_image(self, image, fn_prefix, drivefolder, proj, polygon, dtype, scale):
        # Make filename
        image_date = image.date().format("yyyyMMdd").getInfo()
        image_id = image.id().getInfo()
        filename = f"{fn_prefix}_{image_id}_{image_date}"
        # Cast dtype
        image = {
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
        # define task
        proj_info = proj.getInfo()
        geo_info = geometry.getInfo()
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=filename,
            fileNamePrefix=filename,
            region=geo_info,
            crs=proj_info.crs,
            crs_transform=proj_info.transform,
            folder=folder_name,
            fileFormat="GeoTIFF",
            scale=scale
        )
        # start task
        task.start()

    def download_image_collection(self, icol, fn_prefix, drivefolder, batch, proj, polygon, dtype, scale, localdir): 
        now = datetime.now()
        icol_list = icol.toList(icol.size())
        num_tasks = len(icol_list)#.getInfo()
        # Parallel download
        if batch:
            self.verbose_print(f"Running download tasks in parallel")
            for image in tqdm(icol_list): 
                self.download_image(image, fn_prefix, drivefolder, proj, polygon, dtype, scale)
            self.copy_drive(drivefolder, localdir, datetime.now(), verbose=verbose)
        # Sequential download
        else:
            self.verbose_print(f"Running download tasks sequentially")
            for image in tqdm(icol_list): 
                download_image(image, project_id, collection_name, drivefolder, proj, polygon, dtype, scale)
                self.copy_drive(drivefolder, localdir, datetime.now(), verbose=verbose)

    def download_timeserie(
        self, 
        localdir: str,
        project_id: str,
        polygon: list,
        crs: str, # polygon crs
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

        # args
        fn_prefix = f'{project_id}-GEE_{collection_name.replace("/", "_")}-'
        proj = ee.Projection(crs)
        
        # Create ImageCollection
        icol = self.get_image_collection(
            polygon,
            crs,
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
            self.download_image(img, fn_prefix, drivefolder, proj, polygon, dtype, scale)
        
        # Download image collection
        else:
            self.download_image_collection(icol, fn_prefix, drivefolder, batch, proj, polygon, dtype, scale, localdir)