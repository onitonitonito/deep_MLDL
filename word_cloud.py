"""
# 워드클라우드 깃허브에 예제로 잘 올라와 있다
# https://amueller.github.io/word_cloud/auto_examples/masked.html
"""
# %matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from os import path
from wordcloud import WordCloud
from wordcloud import STOPWORDS

print(__doc__)

filename = "i_have_a_dream.txt"
image_file = 'mask_alice.png'
image_result = 'wcloud_result_image.png'

destin_dir = '_statics/_made_static/'
mask_array = np.array(Image.open(destin_dir + image_file))
f = open(destin_dir + filename).read()

def main():
    wcb = get_wordcloud_basic(f)
    wcm = get_wordcloud_masked(f)

    show_wc(wcb)
    show_wc(mask_array)
    show_wc(wcm)

    save_wc_image(wcm, destin_dir+image_result)

def show_wc(image_array, figsize=(8, 4)):
    """show wordcloud obj. or img_array using imshow()"""
    plt.figure(figsize=figsize)
    plt.imshow(image_array, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def get_wordcloud_basic(f):
    """
    # If you lower the maximum font size, you will get more words
    # wc = WordCloud(max_font_size=30).generate(f) .... Line.10
    """
    global destin_dir, filename
    wc = WordCloud(max_font_size=60) \
                .generate(f)
    wc.words_
    return wc

def get_wordcloud_masked(f):
    """ MASKED WORD CLOUD
    Original Author's Page refer to HERE :
    https://amueller.github.io/word_cloud/auto_examples/masked.html
    """
    # read the mask image taken from
    # http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
    stopwords = set(STOPWORDS)
    stopwords.add("said")

    wc = WordCloud(background_color="white",
                    max_words=2000,
                    mask=mask_array,
                    stopwords=stopwords) \
               .generate(f)
    return wc

def save_wc_image(wordcloud_object, filename_with_dir):
    wordcloud_object.to_file(filename_with_dir)



if __name__ == '__main__':
    main()
