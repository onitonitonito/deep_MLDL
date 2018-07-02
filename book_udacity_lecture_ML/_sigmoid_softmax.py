"""
* 시그모이드 함수그래프와 소프트맥스를 구현한다
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def show_result(*x):
    x = np.array(*x)
    print(sigmoid(x))
    # [0.26894142 0.73105858 0.88079708]


def show_graph(x1=-9.0, x2=9.0, step=0.1):
    x = np.arange(x1, x2, step)
    y = sigmoid(x)

    """ 1. 그래프 플로팅 """
    plt.plot(x, y)

    """ 2. 그래프에 텍스트를 입히는 옵션 """
    plt.title("Simple SIGMOID Plot")        # 타이틀
    plt.xlabel('x')                         # x 라벨
    plt.ylabel('y = 1/[1 + exp(-x)]')       # y 라벨

    """ 3. 기타 라인/제한 옵션 """
    plt.grid(b=None, which='major', axis='both')
    # plt.ylim(1, 2.75)
    # plt.xlim(0, x_args[1])

    """ 4. 그리기 """
    plt.show()


# show_result((-0.01, 0.5, 1, 2, 3, 4, 5, 10))
#
show_graph(-20,10)
