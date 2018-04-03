def _18_numerical_stability():
    """ 18. Numerical stability
    # - Too big / too small
    # - Importance of well-defined, well-optimized.
    """

    a =  pow(10, 9)              # a billion : 10**9 / pow(10,9)
    for i in range(10**6):
        a += 1e-6

    print(a - 10**9)                    # 0.95367431640625

    """ # CHECK RESULT """
    print((1e-6)*10**6)                 # 1.0
    print(a + (1e-6)*10**6)             # 1000000001.9536743
    print(1000000001.9536743 - 10**9)   # 1.95367431640625
_18_numerical_stability()
