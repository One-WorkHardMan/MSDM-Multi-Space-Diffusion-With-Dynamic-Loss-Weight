import torchvision.datasets as datasets

train_set = datasets.CIFAR10("./datasets",train=True,download = True)

import pickle

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

import numpy as np
import matplotlib.pyplot as plt

# 读取第二份训练数据
data = unpickle('datasets/cifar-10-batches-py/data_batch_1')

# 取出第一张图片的数据
img_data = data[b'data'][0]

# 把数据转换成图片形状
img = np.reshape(img_data, (3, 32, 32))

# 把通道顺序调整为(32, 32, 3)
img = img.transpose((1, 2, 0))

# 显示图片
plt.imshow(img)
plt.show()

# 打印图片的标签和文件名
label = data[b'labels'][0]
filename = data[b'filenames'][0]
print(label, filename)


#--------------------------------------------------------------------------------------