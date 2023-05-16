import os, io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# See, edit, create, delete all GDrive files
# see: https://developers.google.com/identity/protocols/oauth2/scopes
SCOPES = ["https://www.googleapis.com/auth/drive"]

TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"

class GDriveV3Wrapper():
    def __init__(
        self,
        token_file: str=TOKEN_FILE,
        credentials_file: str=CREDENTIALS_FILE,
        scopes: list=SCOPES,
        verbose: bool=False
    ):
        self.token_file = token_file
        self.credentials_file = credentials_file
        self.scopes = scopes
        self.verbose = verbose
        # connect
        self.service = self.getService()
        self.folderIdsMap = {}

    def verbose_print(self, msg):
        if self.verbose: print(msg)
    
    def getService(self):
        self.verbose_print("Connecting to google drive...")
        # Check for tokens
        creds = None
        if os.path.exists(self.token_file):
            self.verbose_print(f"Trying token in {self.token_file}...")
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
            if creds.valid: self.verbose_print("Token authorized.")
            else: self.verbose_print("Token could not be authorized.")
        # Get token through login
        if not creds or not creds.valid:
            self.verbose_print(f"Authenticating...")
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        # builg drive service
        self.verbose_print("Building service...")
        service = build('drive', 'v3', credentials=creds)
        return service

    def getFolderId(self, folderName):
        try:
            return (
                self.service
                .files()
                .list(
                    q=f"mimeType = 'application/vnd.google-apps.folder' and name = '{folderName}'",
                    pageSize=10, 
                    fields="nextPageToken, files(id, name)"
                ).execute()
                .get("files", [])[0]
                .get("id")
            )
        except Exception as e:
            print(f"An error occured when requesting drive folder id for name {folderName}")
            raise e

    # To fix: returns empty list for some reason
    def getFileId(self, fileName, folderId):
        self.verbose_print(f"Requesting id for file name {fileName} and folder id {folderId}")
        try:
            return (
                self.service
                .files()
                .list(
                    q=f"mimeType = 'application/vnd.google-apps.file' and name = '{fileName}' and '{folderId}' in parents",
                    pageSize=100, 
                    fields="nextPageToken, files(id, name)"
                ).execute()
                .get("files", [])[0]
                .get("id")
            )
        except Exception as e:
            print(f"An error occured when requesting fileId for name {fileName}")
            raise e

    def listFile(self, folderName):
        if not folderName in self.folderIdsMap.keys():
            self.folderIdsMap[folderName] = self.getFolderId(folderName)
        folderId = self.folderIdsMap[folderName]
        try:
            return (
                self.service
                .files()
                .list(
                    q=f"'{folderId}' in parents", 
                    pageSize=10, 
                    fields="nextPageToken, files(id, name)"
                ).execute()
                .get('files', [])
            )
        except Exception as e:
            print(f"An error occured when listing drive files in folder with name {folderName} and id {folderId}")
            raise e

    def downloadFile(self, localdir, fileId, fileName):
        dest = os.path.join(localdir, fileName)
        self.verbose_print(f"Downloading file with id {fileId} to {dest}")
        try:
            request = self.service.files().get_media(fileId=fileId)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            with open(dest, "wb") as f:
                f.write(file.getvalue())
            self.verbose_print(f"Done downloading file with id {fileId} to {dest}")
        except Exception as  e:
            print(F'An error occurred during download of file with id {fileId}')
            raise e
        return dest

    def deleteFile(self, fileId):
        self.verbose_print(f"Deleting file with id {fileId}")
        try:
            self.service.files().delete(fileId=fileId).execute()
        except Exception as e:
            print(f"An error occured when deleting file with fileId {fileId}")
            raise e
        
if __name__ == "__main__":
    drive = GDriveV3Wrapper()
    folderId = drive.getFolderId("geeExports")
    filename = "764-GEE_COPERNICUS_S2_SR_HARMONIZED-20170821T104021-20231605T182845.tif"
    driveFiles = drive.listFile("geeExports")
    fileId = [f.get("id") for f in driveFiles][[f.get("name") for f in driveFiles].index(filename)]
    path = drive.downloadFile(fileId=fileId, localdir="gee_data", fileName="dlmain-"+filename)
    print(path)
    drive.deleteFile(fileId)
    driveFiles = drive.listFile("geeExports")
    print(filename in [f.get("name") for f in driveFiles])