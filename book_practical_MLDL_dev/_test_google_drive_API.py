# LOCAL FILE SYSTEM with Google Drive API
""" UPLOADING :  files from your local file system
files.upload returns a dictionary of the files which were uploaded.
The dictionary is keyed by the file name, the value is the data which
was uploaded.
"""
from google.colab import files

uploaded = files.upload()
print(uploaded)
quit()


for filename in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=filename,
        length=len(uploaded[filename])))

""" DOWNLOADING : files to your local file system
files.download will invoke a browser download of the file to the user's
local computer.
"""

with open('example.txt', 'w') as f:
    f.write('some content')

files.download('example.txt')
