""" Tensorflow Manipulation : lect08_lab1
텐서플로우 명령어 이해하기
"""
import pprint as pp
import numpy as np
import tensorflow as tf

def show_shape(array):          # helper()
    """ t = tf.constant()
     # tf.shape(t) = # rank / dim
     # t.shape =     # axis / matrics
     # t =           # matrics property
     """
    print("%s # rank=dim"% tf.shape(array))
    print("{:53} = {} \n".format(
        str(array),
        array.shape))

def pretty_print_module():
    """ pprint module (pprint)
     [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
     array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
     """
    t = [[float(n) for n in range(0+i, 3+i)] for i in range(0, 10, 3)]
    print(np.array(t), '\n')
    pp.pprint(np.array(t))

    """ Pretty print :
     :차원이 올라가면 자동적으로 들여쓰기를 넣는다.
     [   ['spam', 'eggs', 'lumberjack', 'knights', 'ni'],
        'spam',
        'eggs',
        'lumberjack',
        'knights',
        'ni']
     """
    stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
    stuff.insert(0, stuff[:])
    pty = pp.PrettyPrinter(indent=4)
    pty.pprint(stuff)

def test0_shape_of_rank_axis():
    """ Shape(행렬) / Rank(차원) / Axis(대각주): Axis=0,1,2,3
     # Tensor("Shape:0", shape=(1,), dtype=int32)   # rank = dim
     # Tensor("Shape_1:0", shape=(2,), dtype=int32) # rank = dim
     # Tensor("Shape_2:0", shape=(3,), dtype=int32) # rank = dim
     # Tensor("Shape_3:0", shape=(4,), dtype=int32) # rank = dim
         t1 = [1 2 3 4]
         t2 = [[1 2 3], [4 5 6]]
         t3 = [[[1,2]], [[3,4]]]
         t4 = [[[[ 1  2  3  4], [ 5  6  7  8], [ 9 10 11 12]]
                [[13 14 15 16], [17 18 19 20],  [21 22 23 24]]]]
     """
    t_nums =[
        [np.float(n) for n in range(1,5)],
        [[np.float(n) for n in range(1+i, 4+i)] for i in range(0, 4, 3)],
        [[[np.float(n) for n in range(1+i, 3+i)]] for i in range(0, 4, 2)],
        [[[[np.float(n) for n in range(1+i, 5+i)]
            for i in range(0+j, 10+j, 4)]
                for j in range(0, 13, 12)]],]

    for t_num in t_nums:
        print(np.array(t_num), '\n')

    for t_num in t_nums:
        show_shape(tf.constant(t_num))

def test1_simple_matmul():
    """
     Tensor("Const:0", shape=(2, 2), dtype=float32)
     Tensor("Const_1:0", shape=(2, 1), dtype=float32)

     _c =  Tensor("MatMul:0", shape=(2, 1), dtype=float32)
     Tensor("Mean:0", shape=(2,), dtype=float32)
     Tensor("Mean_1:0", shape=(2,), dtype=float32)
     """
    _a = [[1.,2.],[3.,4.]]     # (2,1,2) .. (2,2) : (rank, colum)
    _b = [[3.],[9.]]           # (2,1,1) .. (2,1) : (rank, column)
    _c = tf.matmul(tf.constant(_a), tf.constant(_b))   # (2,2) * (2,1) = (2,1)

    print(tf.constant(_a))
    print(tf.constant(_b))
    print("\n_c = ",_c)

    print(tf.reduce_mean(_a, axis=0))
    print(tf.reduce_mean(_a, axis=1))

def test2_np_array_making():
    """ making np_array with comp-list
     array([[[[  0.,   1.,   2.],
             [  3.,   4.,   5.],
             [  6.,   7.,   8.],
             [  9.,  10.,  11.]]],

           [[[ 11.,  12.,  13.],
             [ 14.,  15.,  16.],
             [ 17.,  18.,  19.],
             [ 20.,  21.,  22.]]],

           [[[ 22.,  23.,  24.],
             [ 25.,  26.,  27.],
             [ 28.,  29.,  30.],
             [ 31.,  32.,  33.]]]])
     't-dimension = 4 dims'
     't-shape(L,c)= (3, 1, 4, 3)'
     """
    t = np.array([[[[float(n)
        for n in range(0+i, 3+i)]            # 첫번째 = 행 배열: 0~2
            for i in range(0+j, 10+j, 3)]]   # 두번째 = 열 배열: 0~9
                for j in range(0, 23, 11)])  # 세번째 =배열[0]간격: 0,11,22

    pp.pprint(t)
    pp.pprint("t-dimension = %s dims" %t.ndim)   # rank
    pp.pprint("t-shape(L,c)= %s" %str(t.shape))  # shape
    print(t)

def test3_manipulate():
    """"
     [[[[ 0  1  2]
       [ 3  4  5]
       [ 6  7  8]
       [ 9 10 11]]]

     [[[11 12 13]
       [14 15 16]
       [17 18 19]
       [20 21 22]]]

     [[[22 23 24]
       [25 26 27]
       [28 29 30]
       [31 32 33]]]]
     Tensor("Shape:0", shape=(4,), dtype=int32)
     tf.shape().eval() = [3 1 4 3]
     sess.run() =        [3 1 4 3]
     sess.run() =        [6 2 8 6]

     tf.add() = Tensor("Add_1:0", shape=(4,), dtype=int32)
     2 * _a =   Tensor("mul:0", shape=(4,), dtype=int32)
     """
    t = [[[[n
        for n in range(0+i, 3+i)]            # 첫번째 = 행 배열: 0~2
            for i in range(0+j, 10+j, 3)]]   # 두번째 = 열 배열: 0~9
                for j in range(0, 25, 11)]   # 세번째 =배열[0]간격: 0,11,22

    for _a in [np.array(t), tf.shape(t)]:
        print("%s \n"% _a)

    # print(_a.get_shape())
    # print(_a.get_shape().as_list())

    with tf.Session() as sess:
        print("\ntf.shape().eval()= %s"% _a.eval())
        print("sess.run() =%s "% sess.run(_a))
        print("sess.run() =%s \n"% sess.run(tf.add(_a,_a)))

    print("tf.add() = %s"% tf.add(_a, _a))
    print("2 * _a =   %s"% str(2*_a))

def test4_eval_function():
    """ simple tensorflow test
     tf.constant(9) = Tensor("Const:0", shape=(), dtype=int32)
     sess.run(a) = 9 (scala)
     a.eval() = 9 (scala)
     """
    a = tf.constant(9)
    print("\ntf.constant(9) = %s\n"% a)

    with tf.Session() as sess:
        print("\nsess.run(a) = %s (scala)"% sess.run(a))
        print("a.eval() = %s (scala)"% a.eval()) # 주의: eval()은 세션 안에서 설행 됨!

def test5_reduce_mean_sum():
    """ Broadcast : warning(!)=automatically arrange ranks
    # axis = 0 : column direction
    # axis = 1 : line direction
    """
    x = [
        [1., 2.],
        [3., 4.],]

    with tf.Session() as sess:
        sum_none = tf.reduce_sum(x).eval()
        sum_0 = tf.reduce_sum(x, axis=0).eval()
        sum_1 = tf.reduce_sum(x, axis=1).eval()         # same as, axis = -1

        average_none = tf.reduce_mean(x).eval()
        average_0 = tf.reduce_mean(x, axis=0).eval()
        average_1 = tf.reduce_mean(x, axis=1).eval()    # same as, axis = -1
    print("\n\n", np.array(x))

    print("\n\nsummary axis = None :", sum_none)
    print("summary axis = 0 :", sum_0)
    print("summary axis = 1 :", sum_1)

    print("\n\naverage axis = None :", average_none)
    print("average axis = 0 :", average_0)
    print("average axis = 1 :", average_1)

def test6_argmax_position():
    """ Argmax : show MAX [POS] number
     array([[0, 1, 2],
           [2, 1, 0]])
     Argmax pos axis = 0 : [1 0 0]
     Argmax pos axis = 1 : [2 0]
     """
    x = [
        [0, 1, 2],
        [2, 1, 0],]

    with tf.Session() as sess:
        argmax_0 = tf.argmax(x, axis=0).eval()
        argmax_1 = tf.argmax(x, axis=1).eval()      # same as, axis = -1

    print()
    pp.pprint(np.array(x))
    print("\n\nArgmax pos axis = 0 :", argmax_0)
    print("Argmax pos axis = 1 :", argmax_1)

def show_tensor_shape(array):    # helper()
    """ show array in various ways
    """
    print(np.array(array), "\n")
    print(np.array(array).shape)        # (2,3,3)
    print(tf.constant(array), "\n")     # Tensor, shape= (2,3,3)

def test7_reshape_sqz_expd():
    """ reshape ** / squeeze / expand_dims
     [[[ 0  1  2]
       [ 3  4  5]]

      [[ 6  7  8]
       [ 9 10 11]]]             # .shape = (2, 3, 3)
    """
    t = [[[n for n in range(0+i, 3+i)] for i in range(0+j, 4+j, 3)]
        for j in range(0, 7, 6)]
    show_tensor_shape(t)    # (2,3,3) / Tensor, shape= (2,3,3)

    with tf.Session() as sess:
        t_reshape1 = tf.reshape(t, shape=[-1, 3]).eval()
        t_reshape2 = tf.reshape(t, shape=[-1, 1, 3]).eval()
        t_reshape3 = tf.reshape(t, shape=[-1, 1]).eval()
        t3_squeeze = tf.squeeze(t_reshape3).eval()
        t3sq_expand = tf.expand_dims(t3_squeeze, 1).eval()
        print()

    show_tensor_shape(t_reshape1)   # (4,3)   / Tensor, shape= (4,3)
    show_tensor_shape(t_reshape2)   # (4,1,3) / Tensor, shape= (4,1,3)
    show_tensor_shape(t_reshape3)   # (12,1)  / Tensor, shape= (12,1)

    """ back to t_reshape3 : using expand_dims method """
    show_tensor_shape(t3_squeeze)   # (12,)   / Tensor, shape= (12,)
    show_tensor_shape(t3sq_expand)  # (12,1)  / Tensor, shape= (12,1)

def test8_1hot_cast_stack_func():
    """ if one_hot(), rank += 1 --> need to reshape(!)
    """
    x = [[0], [1], [2], [0],]       # rank = 2

    with tf.Session() as sess:
        one_hots = tf.one_hot(indices=x, depth=3).eval()
        reshaped_1hot = tf.reshape(one_hots, shape=[-1, 3]).eval()
        print()

    print("*** ONE-HOT FUNCTION ***")
    show_shape(np.array(x))
    show_shape(one_hots)
    show_shape(reshaped_1hot)

    """ cast function converts array into specific 'dtype'
     x_int = [1 2 3 4]
     Tensor("Const:0", shape=(4,), dtype=int32)      # column

     x_int2 = [1 0 1 0]
     Tensor("Const_1:0", shape=(4,), dtype=int32)    # column
     """
    x = [1.8, 2.2, 3.3, 4.9]
    bools = [True, False, 1==1, 1==0]

    with tf.Session() as sess:
        x_int = tf.cast(x=x, dtype=tf.int32).eval()
        x_int2 = tf.cast(x=bools, dtype=tf.int32).eval()

    print("\n\n*** CAST() : CONVERT DTYPE ***")
    show_tensor_shape(x_int)
    show_tensor_shape(x_int2)

    """ stack array along the axis
    """
    x, y, z = [1, 4], [2, 5], [3, 6]

    with tf.Session() as sess:
        stack_xyz0 = tf.stack([x, y, z]).eval()
        stack_xyz1 = tf.stack([x, y, z], axis=1).eval()

    print("\n\n*** STACK() : ALONG AXIS ***")
    show_shape(stack_xyz0)
    show_shape(stack_xyz1)

x = [[0,1,2], [3,4,5], [6,7,8]]

with tf.Session() as sess:
    ones = tf.ones_like(tensor=x, dtype=tf.float32).eval()
    zeros = tf.zeros_like(tensor=x, dtype=tf.float32).eval()

print("\n*** ones_like, zeros_like ***")
show_shape(np.array(x))
show_shape(ones)
show_shape(zeros)


if __name__ == '__main__':
    # pretty_print_module()
    # test0_shape_of_rank_axis()
    # test1_simple_matmul()
    # test2_np_array_making()
    # test3_manipulate()
    # test4_eval_function()
    # test5_reduce_mean_sum()
    # test6_argmax_position()
    # test7_reshape_sqz_expd()
    # test8_1hot_cast_stack_func()
    pass
