import os
import sys

# '루트'와 '작업'디렉토리 설정 - for 스크립트런
DIRS = os.path.dirname(__file__).partition("k_mooc_reboot\\")
ROOT = DIRS[0] + DIRS[1]
# -----------------------------------------------------------

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread



def average_hash(fname, size=16):
    # 평균 이진해쉬를 생성해 주는 함수
    img = Image.open(fname)                         # 이미지 '오브젝트' 생성
    img = img.convert('L')                          # 그레이 스케일로 변환
    img = img.resize((size, size), Image.ANTIALIAS) # 이미지 '사이즈' 조절

    pixel_data = img.getdata()              # 픽셀 데이터 가져오기
    pixels = np.asarray(pixel_data)
    pixels = pixels.reshape((size, size))   # 2차원 배열로 변경

    avg = pixels.mean()             # 픽셀의 평균구하기  ...  (*8)
    diff = 1 * (pixels > avg)       # 평균보다 크면'1', 작으면'0'
    return diff                     # 1 or 0

def np_2_hash(fname):
    # 핵사해쉬 이미지를 생성해주는 함수
    hexa_hash = []
    for n1 in average_hash(fname).tolist():
        s1 = [str(i) for i in n1]
        s2 = "".join(s1)
        i = int(s2, 2)          # 이진수를 정수로 변환
        hexa_hash.append("%04x"% i)
    return "".join(hexa_hash)

def print_binary_mean(*fname, size=31):
    fname = os.path.join(*fname)
    print()
    print(average_hash(fname, size), flush=True)    # 이진해쉬 이미지
    print(np_2_hash(fname))                         # 핵사해쉬 이미지
    print()

    # 원본 이미지 출력 by matplotlib
    implot = imread(fname)
    plt.imshow(implot)
    plt.show()



if __name__ == '__main__':
    fname = (ROOT, "_static", "image", "category_101objects", "image_0014-convoluted.jpg")
    print_binary_mean(*fname, size=31)

    fname = (ROOT, "_static", "image", "category_101objects", "image_0014-reformed.jpg")
    print_binary_mean(*fname, size=31)

    fname = (ROOT, "_static", "image", "category_101objects", "image_0014.jpg")
    print_binary_mean(*fname, size=31)


    fname = (ROOT, "_static", "image", "gyudon", "202000499thumb.jpg")
    print_binary_mean(*fname, size=31)
