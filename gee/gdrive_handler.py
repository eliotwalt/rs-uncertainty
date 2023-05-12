import os, io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# See, edit, create, delete all GDrive files
# see: https://developers.google.com/identity/protocols/oauth2/scopes
SCOPES = ["https://www.googleapis.com/auth/drive"]

TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"

class GoogleDriveHandler():
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

    def _get_folder_id(self, folderName):
        try:
            self.verbose_print(f"Requesting id for folder name {folderName}")
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

    def _get_file_id(self, fileName, folderId):
        self.verbose_print(f"Requesting id for file name {folderName} and folder id {folderId}")
        try:
            return (
                self.service
                .files()
                .list(
                    q=f"mimeType = 'application/vnd.google-apps.file' and name = '{folderName}' and '{folderId}' in parents",
                    pageSize=10, 
                    fields="nextPageToken, files(id, name)"
                ).execute()
                .get("files", [])[0]
                .get("id")
            )
        except Exception as e:
            print(f"An error occured when requesting drive folder id for name {folderName}")
            raise e

    def listFile(self, folderName):
        self.verbose_print(f"listing file in {folderName}")
        if not folderName in self.folderIdsMap.keys():
            self.folderIdsMap[folderName] = self._get_folder_id(folderName)
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

    def download_file(self, localdir, fileName=None, fileId=None, folderId=""):
        assert (fileName is not None and folderId is not None) or fileId is not None, f"must provide either folderId and folderName or fileId"
        if fileId is None:
            fileId = self._get_file_id(fileName, folderId)
        if fileName is None:
            fileName = self.service.files().get(fileId=fileId).exectue()["name"]
            dest = os.path.join(localdir, fileName)
        self.verbose_print(f"Deleting file with id {fileId} to {dest}")
        try:
            request = service.files().get_media(fileId=file_id)
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

    def deleteFile(self, fileName=None, fileId=None, folderId=""):
        assert (fileName is not None and folderId is not None) or fileId is not None, f"must provide either folderId and folderName or fileId"
        if fileId is None:
            fileId = self._get_file_id(fileName, folderId)
        self.verbose_print(f"Deleting file with id {fileId}")
        try:
            self.service.file().delete(fileId)
        except Exception as e:
            print(f"An error occured when deleting file with fileId {fileId}")
            raise e

def connect_drive():
    creds = None
    # Check for tokens
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # Get token through login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    # builg drive service
    try:
        service = build('drive', 'v3', credentials=creds)
    except HttpError as error:
        print(f'An error occurred: {error}')
    return service

def get_drivefolderid(service, folderName):
    try:
        return (service.files()
            .list(q = "mimeType = 'application/vnd.google-apps.folder' and name = 'geeExports'", pageSize=10, fields="nextPageToken, files(id, name)").execute())
        folderIdResult = folderId.get("files", [])
        folderId = folderIdResult[0].get("id")
    except Exception as e:
        print(f"An error occured when requesting drive folder id for name {folderName}")
        raise e
    return folderId


if __name__ == "__main__":
    service = connect_drive()
    with open("drive.log", "w") as f:
        f.write("Service object type: {}".format(type(service)))
        f.write("Service object class: {}".format(service.__class__.__name__))
        f.write("Service object members: {}".format(dir(service)))
        f.write("Service object attributes: {}".format(vars(service)))
        f.write("has ListFile: {}".format(hasattr(service, "ListFile")))
        f.write("has CreateFile: {}".format(hasattr(service, "CreateFile")))
    print(service.files())
    folderId = service.files().list(q = "mimeType = 'application/vnd.google-apps.folder' and name = 'geeExports'", pageSize=10, fields="nextPageToken, files(id, name)").execute()
    print("FolderId", folderId)
    folderIdResult = folderId.get("files", [])
    print("FolderIdResult", folderIdResult)
    folderId = folderIdResult[0].get("id")
    print("FolderId", folderId)
    fileIds 
