# %matplotlib inline
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

FILENAME_WITH_DIR = './_static/i_have_a_dream.pdb'

f = open(FILENAME_WITH_DIR).read()
wc = WordCloud(max_font_size=35).generate(f)
# If you lower the maximum font size, you will get more words
# wc = WordCloud(max_font_size=30).generate(f) .... Line.10

wc.words_

plt.figure(figsize=(8, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()



# %matplotlib inline

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

""" MASKED WORD CLOUD
Original Author's Page refer to HERE :
https://amueller.github.io/word_cloud/auto_examples/masked.html
"""
# d = path.dirname(__file__)
# # Read the whole f.
# f = open(path.join(d, 'alice.txt')).read()

DESTIN_DIR = './_static/'

f = open(FILENAME_WITH_DIR).read()

# read the mask image taken from
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
alice_mask = np.array(Image.open(DESTIN_DIR + "alice_mask.png"))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask,
               stopwords=stopwords)

# generate word cloud
wc.generate(f)

# store to file
wc.to_file(DESTIN_DIR + "alice.png")

# show 1
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure()

# show 2
plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()
