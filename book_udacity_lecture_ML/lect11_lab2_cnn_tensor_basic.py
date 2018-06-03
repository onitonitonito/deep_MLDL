import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image_arr = np.array([[
    [[1],[2],[3]],
    [[4],[5],[6]],
    [[7],[8],[9]]]], dtype=np.float32)

# Toy Image = Image:[1,3,3,1] filter:[2,2,1,1] stride:[1x1] Padding=VALID
# 이미지 [3x3x1] 에 필터가 [2x2x1] 이면
print(image_arr)
print(image_arr.shape)              # 1,(3,(3,1)) = 1,3,3,1


weight = tf.constant(
    [
        [
            [[1]],[[1]]
            ],
        [
            [[1]], [[1]]
            ]], dtype='float32')
print("weight shape : ", weight.shape)

conv2d = tf.nn.conv2d(image_arr, weight, strides=[1,1,1,1], padding='VALID')
conv2d_img = conv2d.eval()

print("conv2d_image.shape :", conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)

for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1, 2, i+1), plt.imshow(one_img.reshape(2,2), cmap=plt.cm.Greys)


plt.imshow(image_arr.reshape(3,3), cmap=plt.cm.Greys)
plt.colorbar()
plt.show()
