""" 포토주에서 '규동' 이미지 검색하여, 긁어오기
# "https://api.photozou.jp/rest/search_public.xml?keyword=牛丼"
# "http://photozou.jp/photo/search_result?keyword=牛丼"
"""
import sys
import os
import re
import time
import urllib.request as req
import urllib.parse as parse
import json

DIRS = os.path.dirname(__file__).partition("k_mooc_reboot\\")
ROOT = DIRS[0] + DIRS[1]

# API의 URL 지정하기
PHOTOZOU_API = "https://api.photozou.jp/rest/search_public.json"
CACHE_DIR = os.path.join(ROOT, '_static')

# 포토주에서 'API'로 이미지 검색하기 --- (*1)
def search_photo(keyword, offset=0, limit=100):
    # API 쿼리 조합하기
    keyword_enc = parse.quote_plus(keyword)
    query = "?keyword={0}&offset={1}&limit={2}".format(
        keyword_enc, offset, limit)
    url = PHOTOZOU_API + query

    # 캐시 전용폴더 만들기
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    cache = CACHE_DIR + "/_logdir/" + re.sub(r"[^a-zA-Z0-9\%\#]+", "_", url)

    if os.path.exists(cache):
        return json.load(open(cache, "r", encoding='utf-8'))

    print("[API] : " + url)
    req.urlretrieve(url, cache)
    time.sleep(1)
    return json.load(open(cache, "r", encoding='utf-8'))

# 이미지 다운로드 하기 -- (*2)
def download_thumb(info, save_dir):    # 핼퍼함수()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if info is None:
        return

    if not "photo" in info["info"]:
        print("** [Error] broken info ***")
        return

    photolist = info["info"]["photo"]
    for photo in photolist:
        title = photo["photo_title"]
        photo_id = photo["photo_id"]
        url = photo["thumbnail_image_url"]
        path = save_dir + "/" + str(photo_id) + "thumb.jpg"

        if os.path.exists(path):
            continue

        try:
            print("[download] :", title, photo_id)
            req.urlretrieve(url, path)
            time.sleep(1)

        except Exception as e:
            print("** [Error] failed to download url=", url)

# 모두 검색하고 다운로드 하기 -- (*3)
def download_all(keyword, save_dir, max_photo=1000):
    offset = 0
    limit = 100
    while True:
        # API 호출
        info = search_photo(keyword, offset, limit)

        if info is None:
            print("** [Error] NO RESULTS...! **")
            return

        if not "info" in info or \
            not "photo_num" in info["info"]:
                print("** [Error] Broken DATA...! **")
                return

        photo_num = info["info"]["photo_num"]

        if not photo_num:
            print("photo_num=0, offset=", offset)
            return

        #사진 정보가 포함돼 있으면 다운로드 받기
        print("[다운로드] : offset=", offset)
        download_thumb(info, save_dir)
        offset += limit
        if offset >= max_photo:
            break


if __name__ == '__main__':
    # 모듈로 사용할 수 있게, 설정

    download_all("はな", CACHE_DIR+"/image/flower")     # --- (*4)
    # download_all("牛丼", CACHE_DIR+"/image/gyudon")     # --- (*4)
