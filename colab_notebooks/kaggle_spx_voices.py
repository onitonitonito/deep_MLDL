""" kaggle_SPX_voices.ipynb
# SET: Kaggle.com Voice Recognition
# 관련문서 참조
> 1. [Kaggle.com 원본: BOSTON 의 집값 예측 :](https://www.kaggle.com/sunmiyoon/one-hour-analysis-written-in-both-eng-and-kor/comments#290148)
* [Kaggle.com 원본: 남/녀 목소리 판별 :](https://www.kaggle.com/primaryobjects/voicegender)
* [Pandas Documentation (**10 minutes to Pandas** ) :](https://pandas.pydata.org/pandas-docs/stable/10min.html)
* [ **판다스** , 10분 만에 둘러보기 (한글번역) :](https://dandyrilla.github.io/2017-08-12/pandas-10min/)
"""
import os
import sys

# '루트'와 '작업'디렉토리 설정 - for 스크립트런
DIRS = os.path.dirname(__file__).partition("deep_MLDL")
ROOT = DIRS[0] + DIRS[1]
CSV_DIR = os.path.join(ROOT, "_static", "_csv_hunkim", "")
FNAME_WITH_DIR = CSV_DIR + 'voice_with_header.csv'

# 스크립트런 '한글' 표시를 위한 커스텀 모듈 실행
sys.path.append(ROOT)
import _script_run_utf8
_script_run_utf8.main()

"""#  2.Split data into 'TRAIN SET' / 'TEST SET'
> ## data를 읽은 후
 1. label(male/female)로부터 gender(0.0/1.0 = float64)를 추가한다 (y_data),
 2. **train set** (2,167 개) 과 **test set** (1000 개) 둘로 쪼갠다.
"""

import numpy as np         # linear algebra
import pandas as pd        # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv(FNAME_WITH_DIR)
# print(df)                          # [3168 rows x 21 columns] /'헤더'제외

# df컬럼의 마지막'22열'에 'gender'라는 라벨을 추가 함 (기본값는 'NaN')
# 이것은 label 값이 숫자가 아니기 때문에, 숫자로 바꾸기 위함.
v_data = df.reindex(columns=list(df.columns)+['gender'])
# print(v_data)                         # [3168 rows x 22 columns] /'헤더'제외

# 데이터 선택(슬라이싱) = v_data.at[df[0], 'A'] : 데이터프레임[0] A열 첫줄
# 데이터는 헤더를 제외하고 3,168 개 (0~3167)  ...  at[0, ] = 헤더
print(v_data.at[0, 'label'])            # male   = 'label'열, 0번 (첫번째줄)
print(v_data.at[3167, 'label'])         # female = 'label'열, 3167번(마지막줄)
_len = len(df)                          # 헤더제외, 3168 줄
print(_len)

# 마지막줄 gender를 입력한다 1.0 / 0.0
for i in range(_len):
    if v_data.at[i, 'label'] == 'male':
        v_data.at[i, 'gender'] = 1.0
    else:
        v_data.at[i, 'gender'] = 0.0


print(v_data.at[0, 'gender'])            # 첫번째 줄 1.0 = male
print(v_data.at[3167, 'gender'])         # 마지막 줄 0.0 = female
print(v_data.shape)                      # [3168,22]
v_data = v_data.drop(['label'], axis=1)  # 'label' 열을 삭제한다.
print(v_data.shape)                      # [3168,21]

# shuffling -- 데이터를 행 단위로 무작위로 섞는다  ...  맨 앞열 =index
v_data = v_data.reindex(np.random.permutation(v_data.index))

print(len(v_data.columns), v_data.columns)   # 리스트갯수=21개, 컬럼'헤더'리스트
_split = _len - 1000       # 3168 - 1000 = 2168

train = v_data[0:_split]   # 2168 개 = 학습 세트
test = v_data[_split:]     # 1000 개 = 테스트세트

print()
print("학습데이터 : %s 개" % len(train))
print("검증데이터 : %s 개"% len(test))

"""## Data scanning / 데이터 스캔
* 분석 전에 날것의 데이터를 그대로 스캔하는 것이 중요합니다. 어떤 변수들이 들어있고,
각자의 데이터 타입은 무엇인지, 어떤 변수는 dummy로 사용하고 어떤 변수는 numerical 로
사용 할 것인지 염두해 두면서 데이터를 스캔합니다.
"""
# 랜덤으로 행단위 셔플을 했기때문에 그때,그때 달라 짐
# print(train.tail(2))         # 1611, 532
# print(test.head(2))          # 2706, 155
# print(v_data.tail(3))        # 2727, 1605, 2357
# print(test.tail(3))          # 2357, 1605, 2286
print(v_data.size)                  # 66528
print(len(v_data.columns), v_data.columns) # 21 index,

"""# 3.Summary Statistics (요약통계)
* ## 대표 8개 값 - 요약 통계 (cnt, mean, std, min, 25, 50, 75%, max)
 1. **트레이닝 세트** 와 **테스트 세트** 의 대표적인 8개의 통계값을 확인.
 2. 대표적인 통계 값의 편차를 확인 한다.
"""

print(train.describe())       # 2,168 개
print(test.describe())        # 1,000 개

"""# 4.Correlation matrix
* 첫번째로, 상관관계 테이블을 이용하여 변수들간의 상관관계를 살펴봅니다. 만약 독립변수
(Xs)들 간에 상관관계가 높다면 회귀분석의 결과(coefficient)를 신뢰하기가 힘들어 집니다.
예를 들면, A라는 변수와 B라는 두 독립변수의 상관관계가 매우 높다면 A의 계수를 온전히
A가 Y에 미치는 영향이라고 해석하기 어렵습니다.
"""
import matplotlib.pyplot as plt
import seaborn as sns

f, ax = plt.subplots(1,2, figsize=(15,5))
sns.heatmap(train.corr(), vmax=.8, square=True)
plt.show()


"""* 두 변수의 상관관계가 높은 것이 히트맵에서는 하얀 칸으로 표시됩니다.
상관 관계가 매우 높은 페어들은 다음과 같습니다.
 * meanfreq - median, Q25, Q75, mode, centroid
 * sd - IQR, sfm
 * median - Q25, Q75, mode, centroid
 * centroid - (median), Q25, Q75, mode
 * skew - kurt
 * sp.ent - sfm
 * meandom - maxdom, dfrange
 * maxdom - dfrange
 * mode - Q75
"""

""" 각 페어 당 label과 상관관계가 더 높은 변수 하나만 포함 시킵니다.
* 특정 변수가 Y에 미치는 영향을 정확하게 설명하고 싶다면, 이 과정이 필수적입니다.
하지만 단지 Y를 잘 예측하는 것이 모델의 목적이라면 굳이 이 단계에서 변수를 제거 할
필요는 없습니다.
"""
def print_stronger(f1, f2):
    print('{} > {}'.format(stronger_relation_sale_price(f1, f2)[0], stronger_relation_sale_price(f1, f2)[1]))

def stronger_relation_sale_price(f1, f2):       # 헬퍼()
    f1_corr = train.corr().loc[f1,'gender']
    f2_corr = train.corr().loc[f2,'gender']
    # print(f1_corr, f2_corr)
    return (f1, f2) if (f1_corr >= f2_corr) else (f2, f1)

print_stronger('meanfreq', 'median')    # median > meanfreq
print_stronger('meanfreq', 'Q25')       # meanfreq > Q25
print_stronger('meanfreq', 'Q75')       # Q75 > meanfreq
print_stronger('meanfreq', 'mode')      # mode > meanfreq
print_stronger('meanfreq', 'centroid')  # meanfreq > centroid
# Q75 > mode > median > meanfreq > centroid > Q25
#   ... 이 계열에선, Q75만 포함시키고 나머지는 드롭(Drop:제거) 한다
#   ... 유사한 영향력을 발휘하는 인자 들, 중에, Q75의 영향력이 가장 크기 때문이다

print_stronger('sd', 'IQR')
print_stronger('sd', 'sfm')

print_stronger('median', 'Q25')
print_stronger('median', 'Q75')
print_stronger('median', 'mode')
print_stronger('median', 'centroid')

print_stronger('Q25', 'centroid')
print_stronger('Q75', 'centroid')
print_stronger('mode', 'centroid')

print_stronger('skew', 'kurt')
print_stronger('sp.ent', 'sfm')

print_stronger('meandom', 'maxdom')
print_stronger('meandom', 'dfrange')

print_stronger('maxdom', 'dfrange')

print_stronger('mode', 'Q75')

""" 영향력 상위 지표 5개(Q75, IQR, kurt, sp.ent, dfrange)만 남기고
나머지 비슷한 영향력을 발휘하는 인자들은 대부분 'Drop'(제거)한다.

Q75 > mode > median > meanfreq > centroid > Q25
IQR > sd > sfm
kurt > skew
sp.ent > sfm
dfrange > maxdom > meandom

제거 리스트 (10개) :
[mode, median, meanfreq, centroid, Q25, sd, skew, sfm, maxdom, meandom]
"""

"""* tolkien= Q75 > mode > meanfreq > centroid > median > Q25
* IQR > sd > sfm
* kurt > skew
* sp.ent > sfm
* meandom > dfrange > maxdom
"""

train = train.drop(['mode', 'median', 'meanfreq', 'centroid', 'Q25',
                    'sd', 'skew', 'sfm', 'maxdom', 'meandom'], axis=1)

test = test.drop(['mode', 'median', 'meanfreq', 'centroid', 'Q25',
                    'sd', 'skew', 'sfm', 'maxdom', 'meandom'], axis=1)

print(len(train.columns), train.columns)
sns.heatmap(train.corr(), vmax=.8, square=True)
plt.show()

"""## Histogram and Scatter chart / 히스토그램과 스캐터 차트
* 먼저, 가장 중요한 gender 데이터를 확인 해 봐야 합니다. gender에 음수는 없는지,
말도 안되게 큰 값은 없는지 체크하면서 데이터가 과연 활용할만한 데이터인지 살펴봐야
합니다. 물론 캐글은 깨끗한 데이터를 제공하니 이런 문제가 많지 않지만, 웹 크롤링을 얻은
데이터나 실생활 또는 업무에서 만날 수 있는 데이터들은 이처럼 깨끗하지 않습니다.

아웃라이어를 처리하는 방법에는 winsorization, truncation 등이 있습니다.
"""

# I think this graph is more elegant than pandas.hist()
# train['SalePrice'].hist(bins=100)
sns.distplot(train['gender'])

"""## Scatter chart / 산점도, 스캐터 차트
* Y축은 모두 gender이고, 독립변수를 X축에 맞춰 모든 변수에 대해 산점도를 그렸습니다.
데이터 시각화는 보기에 멋있어 보이기 위해 하는 것만은 아닙니다. 이렇게 모든 독립변수
들에 대해 산점도를 뿌리면 어떤 변수가 특히 종속변수과 연관이 있는지, 다시 말하면
얼마나 종속변수를 설명해 주는지 한 눈에 볼 수 있습니다.

예를 들어, OverallQual과 SalePrice가 양의 관계를 보이며 산점도가 우상향하는 형태를
보이는 것을 확인 할 수 있습니다. 데이터가 수평에 가깝게 뿌려져 있다면 그 변수는
gender 와 낮은 관계를 가진다고 해석할 수도 있습니다. 하지만 이 데이터셋의 경우에는
변수의 갯수도 많고, 변수들간 관계가 복잡하기 때문에 회귀분석을 하기 이전에 성급하게
결론을 내려서는 안 됩니다.
"""

fig, axes = plt.subplots(2, 6, figsize=(15, 7), sharey=True)
for col, a in zip(train.columns, axes.flatten()):
    if col == 'gender':
        a.set_title(col)
        a.scatter(df['gender'], df['gender'])
    else:
        df = train[['gender', col]].dropna()
        a.set_title(col)
        a.scatter(df[col], df['gender'])

"""* 이 data를 lab5-ex.ipynb에 적용해보자."""

# Lab 5 Logistic Regression Classifier
import tensorflow as tf

tf.set_random_seed(743)  # for reproducibility

# collect data
# 'Q75','IQR','kurt','sp.ent', 4개 컬럼의 입력값을 분석 DATA로 활용한다.
x_data = train.loc[:,['Q75','IQR','kurt','sp.ent']].values
y_data = train.loc[:,['gender']].values
print('TRAIN: x_data = ', len(x_data), x_data.shape)
print('TRAIN: y_data = ', len(y_data), y_data.shape)

"""# 5.Build a model
* 트레이닝 데이터 세트 : x_data = [2168, 4]
* 라벨 데이터 세트 : y_data = [2168, 1]
"""

print(x_data[0], y_data[0])     # [n1,n2,n3,n4] [1.]
print(len(x_data))              # 2168 = TRAIN

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

"""# 6.Train a model"""

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2_000):
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
    step % 100 == 0:
        print("%4s __ %s" % (step, cost_val))


""" # 6. Test a model """
x_test = test.loc[:,['Q75','IQR','kurt','sp.ent']].values
y_test = test.loc[:,['gender']].values
print('TEST: x_test = ', len(x_test), x_test.shape)
print('TEST: y_test = ', len(y_test), y_test.shape )

hypo_val, pred_val, accu_val = sess.run(
    [hypothesis, predicted, accuracy], feed_dict={X: x_test, Y: y_test})

print("Accuracy: ", accu_val)
print('Prediction =', pred_val[100:104])
print('Hypothesis =', y_test[100:104])
