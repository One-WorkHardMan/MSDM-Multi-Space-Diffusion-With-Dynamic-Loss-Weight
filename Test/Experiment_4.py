#计算信息熵
import numpy as np
import torch as th
from scipy.stats import entropy

data = th.rand((3,512,32,32))
print(data)

# 定义一个空列表，用于存储每个子tensor的信息熵
entropies = []

# 遍历tensor的第一维度，得到四个子tensor
for sub_tensor in data:
    # 将子tensor展平成一维数组
    sub_tensor = sub_tensor.flatten().numpy()
    # 计算子tensor的总和，作为归一化的分母
    total = np.sum(sub_tensor)
    # 将子tensor除以总和，得到一个概率分布
    prob = sub_tensor / total
    # 调用entropy函数，计算概率分布的信息熵，并添加到列表中
    entropies.append(entropy(prob,base=2))

entropies = np.mean(np.array(entropies))

# 打印结果
print(entropies)

#-----------------------------------------------------------------------------------------------------------------------







