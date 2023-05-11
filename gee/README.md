###  Download of Sentinel-2 images with *ee_download_workflow.py* 

#### 1. Set up an environment with a Python version lower than 3.10

#### 2. Install the required packages:

*for GEE:*
- earthengine-api
- google-cloud-sdk (after this you need to run: `earthengine authenticate` from your commandline --> need to verify this in the browser but it should work from remote, just follow the instructions)
- geetools ([Tutorial](https://github.com/gee-community/gee_tools) for batch downloads)

*for working with the Google Drive: you need to set up a **Google Cloud** account and enable the **Google Drive** API:*

&rarr; for a Tutorial see for example [this](https://www.geeksforgeeks.org/get-list-of-files-and-folders-in-google-drive-storage-using-python/) (only do the setup - not the Python script)
- google-api-python-client 
- google-auth-httplib2
- google-auth-oauthlib

*for downloading and deleting files from drive*

- pydrive (see [Tutorial](https://medium.com/analytics-vidhya/pydrive-to-download-from-google-drive-to-a-remote-machine-14c2d086e84e))
 



- time


#### 3. Required files 
(but it explains where you get them etc. in the tutorials): 
  - *client_secrets.json* (needed for pydrive)
  - *settings.yaml* (if you do not want to authenticate everytime you run the script - works for PyDrive)
  - *creds.json* (this gets created when you have a settings.yaml file and you run the script for the first time)


The *settings.yaml* file should contain the following:

```
client_config_backend: 'settings'
client_config:
   client_id: "<enter own client_id here>" #add your own client_id here
   client_secret: "<enter own client_secret here>" #add your own client_secret here
save_credentials: True
save_credentials_backend: 'file'
save_credentials_file: 'creds.json'
get_refresh_token: True
oauth_scope:
- "https://www.googleapis.com/auth/drive"
```