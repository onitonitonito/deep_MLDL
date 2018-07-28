""" for goole colab/Jupyter server.
  - google API로 저장된 화일을 사용하는 방법
"""
#!pip3 install -U -q PyDrive

# see https://stackoverflow.com/questions/48596521/how-to-read-data-from-google-drive-using-colaboratory-google
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


def set_API_file(filename):
    """ # 1. Authenticate and create the PyDrive client. """
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    """ # 2. PyDrive Documentation / References: """
    # https://googledrive.github.io/PyDrive/docs/build/html/index.html

    """ # 3. Auto-iterate through all files that matches this query """
    _root = drive.ListFile({'q': "'root' in parents and trashed=false"})
    _found = 0

    for _f1 in _root.GetList():
        if _f1['title'] == '_Colab Notebooks':
            print('title: {:11} - id: {}'.format(
                _f1['title'],
                _f1['id']))
            _d1 = drive.ListFile({'q': "'{}' in parents".format(_f1['id'])})

        for _f2 in _d1.GetList():
            if _f2['title'] == "_static":
                print('title: {:11} - id: {}'.format(
                    _f2['title'],
                    _f2['id']))
                _d2 = drive.ListFile({'q': "'{}' in parents".format(_f2['id'])})

            for _f3 in _d2.GetList():
                if _f3['title'] == filename:
                    print('title: {:11} - id: {}'.format(
                        _f3['title'],
                        _f3['id']))
                    _found = 1
                    _f = _f3

    if _found == 0:
        return False

    """ # 3. Load a file by ID and print its contents. """
    downloaded = drive.CreateFile({'id': _f['id']})

    """ # downloaded.GetContentFile("MNIST_data/%s" % (filename)) """
    downloaded.GetContentFile(filename)

    return True

# set_MNIST_file("t10k-images-idx3-ubyte.gz")
# set_MNIST_file("train-images-idx3-ubyte.gz")
# set_MNIST_file("t10k-labels-idx1-ubyte.gz")
# set_MNIST_file("train-labels-idx1-ubyte.gz")



""" 실행결과 :
title: TF, id: 1cu0H-cRK5-yNf1ni...
title: MNIST_data, id: 1VVjjG2F_pkKnK...
title: t10k-images-idx3-ubyte.gz, id: 1zbGfVQk...

title: TF, id: 1cu0H-cRK5-yNf1ni...
title: MNIST_data, id: 1VVjjG2F_pkKnK...
title: train-images-idx3-ubyte.gz, id: 1qdv4-...

title: TF, id: 1cu0H-cRK5-yNf1niD...
title: MNIST_data, id: 1VVjjG2F_pkKnK...
title: t10k-labels-idx1-ubyte.gz, id: 1OOujL9ClL...

title: TF, id: 1cu0H-cRK5-yNf1niDHZSpP...
title: MNIST_data, id: 1VVjjG2F_pkKnKEes2Nu...
title: train-labels-idx1-ubyte.gz, id: 1p3TXII1Lf9ulm_f...
True
"""
