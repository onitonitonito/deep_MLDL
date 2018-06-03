def is_proper_filter(n, f_end, stride_end):
    for f in range(2, f_end+1):
        for stride in range(1, stride_end+1):
            f_size = (n - f)/stride + 1
            trail = int(str(f_size).split('.')[1])
            if not trail:
                print(f"f[{f}x{f}], s:[{stride}] = [{f_size:.1f}x{f_size:.1f}]")
        print()


is_proper_filter(7, 5, 3)
