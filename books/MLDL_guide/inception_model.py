"""
* 구글의 이미지 인식모듈(inception), 3 min chap.11 - p.210
* 꽃사진 = https://download.rensorflow.org/example_image/flower_photo.tgz
* 학습 스크립트 =  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining/retrain.py / 단축URL = https://goo.gl/IrJ94U
* 학습사진과 모델을 ./_statics/img_class/ 에 저장
"""

# /retrain.py
# /img_class
# --/flower_photo
#     --/daisy
#     --/dandelion
#     --/roses
#     --/sunflowers
#     --/tulips

# * 실행방법
# python retrain.py --bottleneck_dir = ./img_class/bottlenecks \
# --model_dir=./img_class/inception \
# --outout_graph=./img_class/flower_graph.pb \
# --output_labels=./img_class/flowrs_labels.txt \
# --image_dir ./img_class/flower_photo \
# --how_many_training_Steps 1000

#flowers_labels.txt = daisy dandelion roses sunflowers tulips
# python predict.py img_class/flower_photos/roses/3065719996_c16ecd5551.jpg 

import os
import sys
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DIRS = os.path.dirname(__file__).partition("deep_MLDL\\")
ROOT = DIRS[0] + DIRS[1]

tf.app.flags.DEFINE_string("output_graph",
        ROOT + "/_statics/img_class/flowers_graph.pb",
        "학습된 신경망이 저장된 위치")

tf.app.flags.DEFINE_string("output_labels",
        ROOT + "/_statics/img_class/flowers_labels.txt",
        "이미지 추측 후, 이미지를 보여 줌.")

FLAGS = tf.app.flags.FLAGS

def main():
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]

    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        logits = sess.graph.get_tensor_by_name('final_result:0')
        image = tf.gfile.FastGFile(sys.argc[1], 'rb').read()
        prediction = sess.run(logits, {'DecodeJpeg/constents:0': image})

    print('--- 예측결과 ---')
    for i in range(len(labels)):
        name = labels[i]
        score = prediction[0][i]
        print("%s (%.2f%%)" % (name, score*100))

    if FLAGS.show_image:
        img = mpimg.imread(sys.argv[1])
        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    tf.app.run()
