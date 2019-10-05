"""
# calculation of matrics size
#
"""
# f[2x2], s:[1] = [6.0 x 6.0]
#
# f[3x3], s:[1] = [5.0 x 5.0]
# f[3x3], s:[2] = [3.0 x 3.0]
#
# f[4x4], s:[1] = [4.0 x 4.0]
# f[4x4], s:[3] = [2.0 x 2.0]
#
# f[5x5], s:[1] = [3.0 x 3.0]
# f[5x5], s:[2] = [2.0 x 2.0]
# print(__doc__)

def proper_filter(n, filter_max, stride_max):
    """
    # size of image is    ... [n x n]
    # filter sizes are   ... up to filter_max
    # stride length is   ... up to stride_max
    """
    for f in range(2, filter_max+1):
        for stride in range(1, stride_max+1):
            f_size = (n - f)/stride + 1
            trail = int(str(f_size).split('.')[1])
            if not trail:
                print(f"f[{f}x{f}], s:[{stride}] = [{f_size:.1f} x {f_size:.1f}]")
        print()


if __name__ == '__main__':
    proper_filter(n=7, filter_max=5, stride_max=3)

"""
"""
