import os.path
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import ee
ee.Initialize(project="ee-walteliot") #if authentication needs to be refresehd: run earthengine authenticate --> if accessing from remote: copy output to your pc and follow instructions (you need gcloud instaalled on your PC for this)
import geetools 

import time
from dateutil.relativedelta import relativedelta
from datetime import datetime
from tqdm import tqdm

#PyDrive requires the client_secrets.json file plus (if you want to save the authentication) a settings.yaml file and creds.json
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
gauth = GoogleAuth()
gauth.CommandLineAuth() #solvable from remote
drive = GoogleDrive(gauth)
# if you get an error that the token expired --> delete what is contained in the creds.json file and import this script again
# maybe this could be automated more smoothly in the future by automating the refreshing of the token

def download_local(drivefolderid, localdir, start, verbose=False):
    while still_going(start):
        time.sleep(1)
        downloadasgo(drivefolderid, localdir, verbose=verbose)
    downloadasgo(drivefolderid, localdir, verbose=verbose)

def download_image(
    image: ee.Image,
    project_id: str,
    collection_name: str,
    folder_name: str, 
    proj: ee.Projection,
    geometry: ee.Geometry,
    dtype: str,
    scale: int=10,
):
    # Make filename
    image_date = image.date().format("yyyyMMdd").getInfo()
    image_id = image.id().getInfo()
    filename = f"{project_id}_GEE_{collection_name.split('/')[1]}_{image_id}_{image_date}"
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

def download_project(
    localdir: str,
    project_id: str,
    polygon: list,
    crs: str,
    gt_date: datetime,
    date_offset_amount: int, 
    date_offset_unit: str="day",
    date_offset_policy: str="both", # (before, after, both)
    collection_name: str="COPERNICUS/S2_SR_HARMONIZED",
    s2_bands: list=['B1','B2','B3','B4','B5','B6','B7','B8','B8A', 'B9', 'B11','B12','MSK_CLDPRB'],
    drivefolder: str="geeExports",
    drivefolderid='',
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
    # Connect to GDrive
    if verbose: print("Connecting to gdrive...")
    if not drivefolderid and len(drive.ListFile({'q': "title='{id}'".format(id=drivefolder)}).GetList())==0: 
        file_metadata = {'title': drivefolder,
                            'parents': ['root'],
                            'mimeType': 'application/vnd.google-apps.folder'}
        folder = drive.CreateFile(file_metadata)
        folder.Upload()
        drivefolderid = folder['id']
    elif not drivefolderid: 
        drivefolderid = drive.ListFile({'q': "title='{id}'".format(id=drivefolder)}).GetList()[0]['id']  #version v2 of Google Drive API
    
    # polygon (projected to lat/lon)
    if verbose: print("Creating polygon")
    proj = ee.Projection(crs)
    aoi = (ee.Geometry.Polygon(polygon, proj=proj)
            .transform(ee.Projection("EPSG:4326"))) 
    if verbose:
        print(f"Original CRS: {crs}")
        print(f"Reprojected CRS: EPSG:4326")
    
    # dates
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
    if verbose: print(f"Date filter: {start_date.strftime('dd.MM.yyyy')}-{{end_date.strftime('dd-MM-yyyy')}}")
    
    # image collection
    icol = (ee.ImageCollection(collection_name)
             .filterBounds(aoi)
             .filterDate(start_date, end_date)
             .select(s2_bands))
    if mosaic:
        icol = mergeByDate(icol)
    if verbose: print(f"Found {icol.size().getInfo()} images")
    if agg is not None:
        if verbose: print(f"Downloading aggregated image")
        if "cloudless" in agg:
            # TODO: cloud masking
            raise NotImplementedError()
        agg_img = eval(f"icol.{agg}()")
        download_image(agg_img, project_id, collection_name, drivefolder, proj, polygon, dtype, scale)
        download_local(drivefolderid, localdir, datetime.now(), verbose=verbose)
    
    # Download
    else:
        now = datetime.now()
        num_tasks = icol.size().getInfo()
        icol_list = icol.toList(num_tasks)
        # Parallel download
        if batch:
            if verbose: print("Creating download tasks")
            for image in tqdm(icol_list): 
                download_image(image, project_id, collection_name, drivefolder, proj, polygon, dtype, scale)
            if verbose: print("Submitting batch download")
            download_local(drivefolderid, localdir, datetime.now(), verbose=verbose)
        # Sequential download
        else:
            if verbose: print("Runnind download tasks sequentially")
            for image in tqdm(icol_list): 
                download_image(image, project_id, collection_name, drivefolder, proj, polygon, dtype, scale)
                download_local(drivefolderid, localdir, datetime.now(), verbose=verbose)

def ee_download(localdir, aoi_df, idcol = 'identifier', geocol = 'geegeo', datecol = 'dates', s2_bands = ['B4', 'B3', 'B2', 'B8', 'QA60'], advance=(1, 'week'), imagecollection="1C", drivefolder='RMACdownload', drivefolderid='',  batch=True, median=True, verbose=True): 
    '''
    function to export all the sentinel-2 images falling into the times and places specified in the pandas df from the GEE to Drive, 
    and subsequently download those to your local disc and delete them from Drive

    PARAMETERS
    localdir: path to the local directory you want the images to be stored at
    aoi_df: pandas dataframe, currently organized in a way that it requires the following columns:
        idcol: name/identifier of the area of interest i.e. the area you want to download images for
        geocol: polygons in the form of a nested list, defining the area of interest
        datecol: list of dates ("yyyy-MM-dd") before or after you want to get images for
    s2_bands: list, bands to select
    advance: tuple - 1st element: int referring to delta (negative if you want to go before the dates, positive if you want to go post the dates); default: 1
                     2nd element: String referring to the unit: One of 'year', 'month' 'week', 'day', 'hour', 'minute', or 'second'; default: 'week'
    imagecollection: String referring to the ImageCollection you want to access, i.e. either "1C" referring to Sentinel-2 Level 1C or "2A" referring to Sentinel-2 Level 2A
    drivefolder: name of drive folder temporarily used for file download, not deleted to enable manual checking of download process. Folder should be empty and is created with 'root' parent if does not exist. Default 'RMACdownload'
    drivefolderid: manual hack to circumvent drivefolder creation and id retrieval. Default '', i.e. rewritten 
    batch: Boolean, indicating whether you want to download all the images in the time span (batch=True) or only one (batch=False)
    median: Boolean, indicating (when you only download one image) whether you want to download the median (True) or the image with the least clouds (False)
    '''
    flag = False

    if not drivefolderid and len(drive.ListFile({'q': "title='{id}'".format(id=drivefolder)}).GetList())==0: 
        file_metadata = {'title': drivefolder,
                            'parents': ['root'],
                            'mimeType': 'application/vnd.google-apps.folder'}

        folder = drive.CreateFile(file_metadata)
        folder.Upload()
        drivefolderid = folder['id']
    elif not drivefolderid: drivefolderid = drive.ListFile({'q': "title='{id}'".format(id=drivefolder)}).GetList()[0]['id']  #version v2 of Google Drive API
    
    now = datetime.now()

    for i in range(len(aoi_df)): 
        aoi_name = aoi_df[idcol][i]
        aoi = ee.Geometry.Polygon(aoi_df[geocol][i])



        if imagecollection == '1C':
            s2 = ee.ImageCollection("COPERNICUS/S2")       #Level-1C: orthorectified top-of-atmosphere reflectance.                 Dataset availability: 2015-06-23 – Present
        elif imagecollection == '2A':
            s2 = ee.ImageCollection("COPERNICUS/S2_SR") #Level-2A: orthorectified atmospherically corrected surface reflectance. Dataset availability: 2017-03-28 – Present (earlier L2 coverage is not global)
        
        cropped = s2.filterBounds(aoi)
        selected = cropped.select(s2_bands)

        for j in range(len(aoi_df[datecol][i])):
            date1 = aoi_df[datecol][i][j]
            date1_ee = ee.Date(date1)
            date2_ee = date1_ee.advance(advance[0], advance[1])
            try: date2 = date2_ee.format("yyyy-MM-dd").getInfo()
            except: 
                print(date1, " is not an appropriate date format - continuing with next date.")
                continue  
            if advance[0] > 0:           
                filtered = selected.filterDate(date1_ee, date2_ee)
            else: 
                filtered = selected.filterDate(date2_ee, date1_ee) 

            # mosaic images of the same date together: 
            filtered = mergeByDate(filtered)

            # export to Drive:
            if batch == True: #export all images of the collection:
                try: 
                    geetools.batch.Export.imagecollection.toDrive( 
                        collection=filtered,
                        region=aoi,
                        folder=drivefolder,
                        namePattern=aoi_name + '_{system_date}_{id}', #only aoi + date not good enough -> overwrites files
                        scale=10,
                        dataType='float',
                        crs='EPSG:4326')
                    
                    flag = True
                except Exception as e:
                    print("There is no image in the specified polygon '{}' inbetween {} and {}.".format(aoi_name,date1,date2)) 
                    print(e)
            else:
                if median == False: #if you want to export the image with the least clouds:
                    singleimg = filtered.sort('CLOUD_COVER', True).first()
                    imgdate = singleimg.date().format("yyyy-MM-dd").getInfo()
                    imgid = singleimg.id().getInfo()            
                else: #if you want the median image of that collection:
                    singleimg = filtered.median()
                    imgdate="median_inbetween"
                    imgid = date1 + "_" + date2

                try: 
                    task = ee.batch.Export.image.toDrive(image=singleimg,
                                                region=aoi,
                                                folder=drivefolder,
                                                description= "{}_{}_{}".format(aoi_name, imgdate, imgid),
                                                scale=10,
                                                crs='EPSG:4326')
                    task.start()
                    flag = True
                except Exception as e: 
                    print("There is no image in the specified polygon '{}' inbetween {} and {}.".format(aoi_name,date1,date2)) 
                    print(e)

            downloadasgo(drivefolderid, localdir, verbose=verbose) 

    if flag and verbose: print('\nLaunched all GEE tasks, waiting for GEE')

    while still_going(now):
        time.sleep(1) #arbitrary sleep time to avoid repeated call
        downloadasgo(drivefolderid, localdir, verbose=verbose)
    downloadasgo(drivefolderid, localdir, verbose=verbose) #Ensure no file left behind due to possible (unlikely) small time inconsistences between GEE & Drive   
        

def still_going(now): 
    '''
    function to check whether the tasks started at the beginning of the loop are still running on you Earth Engine account

    PARAMETERS
    now (datetime): datetime set at the start of each loop

    RETURNS
    True if any of the tasks is unsubmitted, ready or running
    '''
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

def downloadasgo(drivefolderid, localdir, verbose=False):
    fileList = drive.ListFile({'q': "'{id}' in parents and trashed=false".format(id=drivefolderid)}).GetList() 
    for file in fileList:
        # if verbose: print('Downloading: ', file['title'], file['id'])
        ref = drive.CreateFile({'id': file['id']}) # drive.CreateFile() does not create the file but a reference to it
        try: 
            ref.GetContentFile(localdir + file['title'])
            print("The download of {} worked.".format(file['title']))
            try: 
                ref.Delete()
                print("The deletion of file {} worked.".format(file['title']))
            except Exception as e: 
                print("The deletion of {} did not work!".format(file['title']))
                print(e)
        except Exception as e: 
            print("The download of {} did not work!".format(file['title']))
            print(e)
        
def mergeByDate(imgCol):
    '''
    function that merges images together that have the same date. 
    adapted from: https://gis.stackexchange.com/questions/372744/how-to-create-a-loop-which-creates-single-mosaic-images-for-each-day-from-a-sel

    PARAMETERS
    imgCol: [ee.ImageCollection] mandatory value that specifies the image collection to merge by dates with.

    RETURNS 
    ee.ImageCollection where images of the same day are mosaiced together
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