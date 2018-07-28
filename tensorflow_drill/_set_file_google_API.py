"""
# Google Colaboratory Google API
#   - How to LOAD a file using 'PyDrive'
#   - Way-01 : Read Token from Dir getList()
#   - sometimes, getList() occurs error then use Way-02
#"""
print(__doc__)
# !pip3 install - U - q PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


def set_file_google_API(file_name):
    # 1. Authenticate and create the PyDrive client.
    auth.authenticate_user()
    ga = GoogleAuth()
    ga.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(ga)

    # PyDrive reference:
    # https://googledrive.github.io/PyDrive/docs/build/html/index.html

    # Auto-iterate through all files that matches this query
    _root = drive.ListFile({'q': "'root' in parents and trashed=false"})

    def get_drive_from_dirname(drive_file, dir_name):
        for _file in drive_file.GetList():
            if _file['title'] == dir_name:
                print('name: %-37s -id: %s' % (_file['title'], _file['id']))
                return drive.ListFile({'q': "'%s' in parents" % (_file['id'])})

    def get_file_from_drive_file(file_name, drive_file):
        for _file in drive_file.GetList():
            if _file['title'] == file_name:
                print('file_name: %-32s -id: %s' %(_file['title'], _file['id']))
                print("-" * 90 + "\n"
                      "file found : %s \n" % True if _file['id'] else False)
                return _file

    d1 = get_drive_from_dirname(_root, 'StudyGroup')    # /root
    d2 = get_drive_from_dirname(d1, 'KKKSSPPP')         # /folder_01
    d3 = get_drive_from_dirname(d2, '_static')          # /folder_11

    # ff = file_found : target object                   # 'file_name'
    ff = get_file_from_drive_file(file_name, d3)

    # 3. Load a file by ID and print its contents.
    downloaded = drive.CreateFile({'id': ff['id']})

    #downloaded.GetContentFile("MNIST_data/%s" % (filename))
    downloaded.GetContentFile(file_name)


set_file_google_API("t10k-images-idx3-ubyte.gz")
set_file_google_API("train-images-idx3-ubyte.gz")
set_file_google_API("t10k-labels-idx1-ubyte.gz")
set_file_google_API("train-labels-idx1-ubyte.gz")




""" RUN : RESULT
name: TF                               -id: 1SdP2zgmGkLh...
name: Kay-SPX                          -id: 1B_C_j0Y3eOg...
name: _static                          -id: 1UGjXt-6ikWN...
file_name: t10k-images-idx3-ubyte.gz   -id: 1SgjLhkn1TNn...
------------------------------------------------------------
file found : True

name: TF                               -id: 1SdP2zgmGkLh...
name: Kay-SPX                          -id: 1B_C_j0Y3eOg...
name: _static                          -id: 1UGjXt-6ikWN...
file_name: train-images-idx3-ubyte.gz  -id: 18wa2IhNfk_a...
------------------------------------------------------------
file found : True

name: TF                               -id: 1SdP2zgmGkLh...
name: Kay-SPX                          -id: 1B_C_j0Y3eOg...
name: _static                          -id: 1UGjXt-6ikWN...
file_name: t10k-labels-idx1-ubyte.gz   -id: 13QEESRCpM08...
------------------------------------------------------------
file found : True

name: TF                               -id: 1SdP2zgmGkLh...
name: Kay-SPX                          -id: 1B_C_j0Y3eOg...
name: _static                          -id: 1UGjXt-6ikWN...
file_name: train-labels-idx1-ubyte.gz  -id: 1-giZcq7FezI...
------------------------------------------------------------
file found : True
"""
