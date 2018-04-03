import os
import sys
import re
import numpy as np
from PIL import Image


# '루트'와 '작업'디렉토리 설정 - for 스크립트런
DIRS = os.path.dirname(__file__).partition("k_mooc_reboot\\")
ROOT = DIRS[0] + DIRS[1]

# categories = ['chair', 'camera', 'butterfly', 'elephant', 'flamingo']
search_dir = ROOT + "_static\\image\\category_101objects"
cache_dir = ROOT + "_static\\image\\cache_avhash"

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# 이미지 데이터를 AvHash로 변환하기  ...  (*1)
def average_hash(fname, size=31):
    fname2 = fname[len(search_dir):]
    print("--------------", fname2)
    # 이미지 캐시하가
    cache_file = cache_dir + "/" + fname2.replace("/", "_").replace(".","_") + ".csv"
    if not os.path.exists(cache_file):          # 해시 생성하기
        img = Image.open(fname)
        img = img.convert('L').resize((size,size))
        pixels = np.array(img.getdata()).reshape((size, size))
        avg = pixels.mean()
        px = 1 * ( pixels > avg)
        np.savetxt(cache_file, px, fmt="%.0f", delimiter=",")
        return px           # 1 or 0

# 해밍 거리 구하기
def hamming_dist(source, distance):
    aa = source.reshape(1, -1)    # 1차원 배열
    ab = distance.reshape(1, -1)
    dist = (aa != ab).sum()
    return dist

# 모든 폴더에 적용하기
def enum_all_files(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            fname = os.path.join(root, f)
            if re.search(r'\.(jpg|jpeg|png)$', fname):
                yield fname

# 이미지 찾기  ...  에러발생(!)
def find_image(fname, rate):
    src = average_hash(fname)
    print("---1---\n", src, fname, '\n\n', flush=True) # None for TEST
    for fname in enum_all_files(search_dir):
        dist = average_hash(fname)
        print("---2---\n", dist, fname, '\n\n', flush=True) # None for TEST
        diff_r = hamming_dist(src, dist) / 256

        if diff_r < rate:
            yield (diff_r, fname)

# 찾기  ...  (*5)
src_file = os.path.join(search_dir, "image_0014.jpg")

html = ""
sim = list(find_image(src_file, 0.25))
sim = sorted(sim, key=lambda x: x[0])

for r, f in sim:
    print(r, ">", f)
    s = """
        <div style='float:left;'>
            <h3>[차이(dist): {0} - {1} ]</h3>
            <p><a href='{2}'><img src='{2}' width=400></a></p>
        </div>""".format(str(r), os.path.basename(f), f)

    html += s

# html 로 출력하기
html = """
    <html>
        <head><meta charset='utf8'></head>
        <body>
            <h3>원래이미지</h3><p>
            <img src='{0}' width=400></p>{1}
        </body>
    </html>""".format(src_file, html)

with open("./avhash-search-outout.html", "w", encoding='utf8') as f:
    f.write(html)

print(" ... html write O.K. ...")
