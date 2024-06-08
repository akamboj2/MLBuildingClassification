from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
from oauth2client import file, client, tools
from httplib2 import Http
import os
import shutil

SCOPES = 'https://www.googleapis.com/auth/drive'


def save(file_path):
    """
    Saves the parameter file and returns the key.
    :param file_path: path to the file
    :return: file id
    """
    store = file.Storage('token.json')
    creds = store.get()

    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        creds = tools.run_flow(flow, store)

    drive = build('drive', 'v3', http=creds.authorize(Http()))

    file_metadata = {'classifier_param': 'params.txt'}
    media = MediaFileUpload(file_path,
                            mimetype='classifier_param/txt')

    out = drive.files().create(body=file_metadata,
                               media_body=media,
                               fields='id').execute()
    return out.get('id')


def load(file_id, file_name, file_dir):
    """
    Loads the file id as the file name in the file directory.
    :param file_id: id
    :param file_name: file name
    :param file_dir: file directory
    :return:
    """
    store = file.Storage('token.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        creds = tools.run_flow(flow, store)

    drive = build('drive', 'v3', http=creds.authorize(Http()))
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Downloading google drive file... %d%%." % int(status.progress() * 100))

    path = file_dir + '/' + file_name
    if os.path.isfile(path):
        os.remove(path)

    fh.seek(0)
    with open(path, 'wb') as f:
        shutil.copyfileobj(fh, f, length=2**30)
