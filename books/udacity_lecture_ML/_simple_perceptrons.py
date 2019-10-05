"""
* 적절한 가중치 값을 사용하면, 간단한 퍼셉트론 식을 구성할 수 있다.
"""

import numpy as np


def _and(x1, x2):           # AND 논리식은 간단히 구현가능
    """ AND 게이트 INPUT=1 or 0 일 때,
    * (0.5 X1 + 0.5 X2) >= 0.7 로 판단식을 구성
    """
    w1, w2, theta = 0.5, 0.5, 0.7
    temp = x1 * w1 + x2 * w2

    if temp <= theta:
        return 0
    elif temp > theta:
        return 1

def _or(x1, x2):            # OR 논리식은 간단히 구현가능
    """ OR 게이트 INPUT=1 or 0 일 때,
    * (X1 + X2) >= 1.0 로 판단식을 구성
    """
    w1, w2, theta = 1.0, 1.0, 1.0
    temp = x1 * w1 + x2 * w2

    if temp < theta:
        return 0
    elif temp >= theta:
        return 1

def _nand(x1, x2):          # NAND 논리식은 간단히 구현가능
    """ NAND 게이트 INPUT=1 or 0 일 때,  ... numpy 로 구현해 본다.
    * (X1 + X2) >= 1.0 로 판단식을 구성
    """
    x = np.array([x1, x2])      # 입력값
    w = np.array([-0.5, -0.5])    # 가중치
    b = 0.7                    # 편향

    h = np.sum(x * w) + b         # Hypothesis H =
    # print(H)                    # -0.19999999999999996

    if h <= 0:
        return 0
    else:
        return 1

def _xor(x1, x2):           # 적절히 논리식들을 조합하면 구현가능
    """ p.58 - NAND, OR, NAND 를 조합하면 XOR을 실현할 수 있다.
    * 입력값 X1, X2 가 동시에 NAND와 OR게이트로 입력해서
    * 결과값 S1, S2를 AND 게이트로 묶는다
    """
    s1 = _nand(x1, x2)
    s2 = _or(x1, x2)
    y = _and(s1, s2)
    return y

def show_result(logic_name):
    _input = [[0, 0], [1, 0], [0, 1], [1, 1]]
    print("\n\n--- This is '%s' GATE ---" % logic_name)

    for x in _input:
        if logic_name is "AND":
            print("  %s = %s" % (x, _and(*x)))

        elif logic_name is "OR":
            print("  %s = %s" % (x, _or(*x)))

        elif logic_name is "NAND":
            print("  %s = %s" % (x, _nand(*x)))

        elif logic_name is "XOR":
            print("  %s = %s" % (x, _xor(*x)))

        else:
            print("*** unlknow logic name! ***")


if __name__ == '__main__':
    show_result("AND")
    show_result("OR")
    show_result("NAND")
    show_result("XOR")
    show_result("XAND")
