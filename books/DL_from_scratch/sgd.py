"""
# Page 191
# Error: 아직 안되는 중... !
# params, grads 가 안바뀌는 이유 몰겠다.
#
"""
print(__doc__)


class SGD(object):
    def __init__(self, lr=0.01):
        # set default learning_rate = 0.01
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            print(params)
            params[key] -= self.lr * grads[key]
        return params, grads


def main(params, grads):
    optimizer = SGD(lr=20)

    # x_batch, t_batch = get_mini_batch()
    # grads = network.gradient(x_batch, t_batch)
    # params = network.params
    return optimizer.update(params, grads)


if __name__ == '__main__':

    params = {
        'low_key': 1000,
        'hige' : 3000,
        'mid' : 2000,
    }

    grads = {
        'low_key': 1000,
        'hige' : 3000,
        'mid' : 2000,
    }

    params, grads = main(params, grads)

    print(params)
    print(grads)
