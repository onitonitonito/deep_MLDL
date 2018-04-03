import os
import sys

ROOT_DIR = "k_mooc_reboot"
CURRENT_DIR = os.path.dirname(__file__)

""" 공통 스크립트를 패치하기 위한 사전작업 """
dirs = CURRENT_DIR.partition(ROOT_DIR)
sys.path.append(dirs[0] + dirs[1])

""" 화일, DIR 관련 패치 """
import _files_dirs_run as fd
# fd.main()         # --- ROOT_DIR 의 화일리스트가 '장식자'로 표시된다.

WORK_DIR = os.path.join(fd.get_dir(), '_static', '_csv_hunkim', '')
# print(WORK_DIR)



# %matplotlib inline
import numpy as np         # linear algebra
import pandas as pd        # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Lab 5 Logistic Regression Classifier
import tensorflow as tf



_data = pd.read_csv(WORK_DIR + 'voice_with_header.csv')
# print(_data)                          # [3168 rows x 21 columns] / 3167

# _data 컬럼의 마지막'열'에 'gender'를 추가 함, 기본값는 'NaN'
v_data = _data.reindex(columns=list(_data.columns) + ['gender'])
# print(v_data)                         # [3168 rows x 22 columns] / 3167

# 데이터 선택(슬라이싱) = v_data.at[df[0], 'A'] : 데이터프레임[0] A열 첫줄
print(v_data.at[3167, 'label'])         # 'label'열, 3167번(마지막줄) = female

_len = len(_data)                       # 헤더포함 3168 줄
print(_len)

# 마지막줄 gender를 입력한다 1.0 / 0.0
for i in range(_len):
    if v_data.at[i, 'label'] == 'male':
        v_data.at[i, 'gender'] = 1.0
    else:
        v_data.at[i, 'gender'] = 0.0

print(v_data.at[3167, 'gender'])         # 마지막 줄 0.0 = female
v_data = v_data.drop(['label'], axis=1)  # 'label' 열을 삭제한다.

# shuffling -- 데이터를 무작위로 섞는다.
v_data = v_data.reindex(np.random.permutation(v_data.index))

print(len(v_data.columns), v_data.columns)   # 21개, 컬럼 '헤더'표시
_split = _len - 1001       # 3168 - (1000+1) = 2167 + 1

train = v_data[0:_split]
test = v_data[_split:-1]

print(len(train), len(test))

print(train.tail(2))         # 1611, 532
print(test.head(2))          # 2706, 155
print(v_data.tail(3))        # 2727, 1605, 2357
print(test.tail(3))          # 2357, 1605, 2286
# v_data.size                  # 66528
# v_data.columns               #

print(train.describe())       # 2,167 개
print(test.describe())        # 1,000 개

f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(train.corr(), vmax=.8, square=True);

def strongerRelationSalePrice(f1, f2):
    f1Corr = train.corr().loc[f1,'gender']
    f2Corr = train.corr().loc[f2,'gender']

    # print(f1Corr, f2Corr)
    return (f1, f2) if (f1Corr >= f2Corr) else (f2, f1)

def print_stronger(f1, f2):
    print('{} > {}'.format(
        strongerRelationSalePrice(f1, f2)[0],
        strongerRelationSalePrice(f1, f2)[1],
        ))

print_stronger('meanfreq', 'median')
print_stronger('meanfreq', 'Q25')
print_stronger('meanfreq', 'Q75')
print_stronger('meanfreq', 'mode')
print_stronger('meanfreq', 'centroid')
print_stronger('sd', 'IQR')
print_stronger('sd', 'sfm')
print_stronger('median', 'Q25')
print_stronger('median', 'Q75')
print_stronger('median', 'mode')
print_stronger('median', 'centroid')
print_stronger('Q25', 'centroid')
print_stronger('Q75', 'centroid')
print_stronger('skew', 'kurt')
print_stronger('sp.ent', 'sfm')
print_stronger('mode', 'centroid')
print_stronger('meandom', 'maxdom')
print_stronger('meandom', 'dfrange')
print_stronger('maxdom', 'dfrange')
print_stronger('mode', 'Q75')

train = train.drop(['mode', 'meanfreq', 'centroid', 'median', 'Q25', 'sd', 'sfm', 'skew', 'sfm', 'dfrange', 'maxdom'], axis=1)
test = test.drop(['mode', 'meanfreq', 'centroid', 'median', 'Q25', 'sd', 'sfm', 'skew', 'sfm', 'dfrange', 'maxdom'], axis=1)

print(len(train.columns), train.columns)

f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(train.corr(), vmax=.8, square=True)

# I think this graph is more elegant than pandas.hist()
# train['SalePrice'].hist(bins=100)
sns.distplot(train['gender'])

fig, axes = plt.subplots(2, 6, figsize=(15, 7), sharey=True)
for col, a in zip(train.columns, axes.flatten()):
    if col == 'gender':
        a.set_title(col)
        a.scatter(df['gender'], df['gender'])
    else:
        df = train[['gender', col]].dropna()
        a.set_title(col)
        a.scatter(df[col], df['gender'])


tf.set_random_seed(743)  # for reproducibility

# collect data
# 'Q75','IQR','kurt','sp.ent', 4개 컬럼의 입력값을 분석 DATA로 활용한다.
x_data = train.loc[:,['Q75','IQR','kurt','sp.ent']].values
y_data = train.loc[:,['gender']].values

print(x_data)
print(y_data)

print(x_data[0],y_data[0])
print(len(x_data))
print(type(x_data))


X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

print(x_data, len(x_data))
print(y_data, len(y_data))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2000):
    cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print("%4s __ %s" % (step, cost_val))

x_test = test.loc[:,['Q75','IQR','kurt','sp.ent']].values
y_test = test.loc[:,['gender']].values

print('TEST: x_test = ', len(x_test))
print('TEST: y_test = ', len(y_test))

hypo_val, pred_val, accu_val = sess.run(
    [hypothesis, predicted, accuracy],
    feed_dict={X: x_test, Y: y_test})

print("Accuracy: ", accu_val)
print('Prediction =', pred_val[100:108])
print('Hypothesis =', y_test[100:108])
