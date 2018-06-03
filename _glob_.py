import glob

# 화일이 있는 위치의 모든리스트를 가져온다 (화일/폴더)
# for file in glob.glob('*'):
#     print(file)

# 서브 디렉토리 수준의 모든리스트를 가져온다 (화일/폴더)
# for file in glob.glob('**/*'):
#     print(file)

import os

# 현재 폴더에 .py 확장자를 가진 파일의 절대 경로를 출력
for file_with_path in glob.iglob(os.path.abspath('./*.py')):
    print(file_with_path)

# 모든 폴더의 파일, 절대경로 포함하여 출력
for file in glob.glob('**/*'):
    print(os.path.abspath(file))

# 모든 폴더에서 '_'로 시작하는 화일 찾기
# for file in glob.glob('**/*'):
for file in glob.glob('**/[_]*.*', recursive=True):
    print(os.path.abspath(file))


"""
그 이외 예제들...
참고 : https://docs.python.org/3/library/glob.html
"""
