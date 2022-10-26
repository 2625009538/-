import torch

print('4.矩阵的计算')
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print('A:', A)
print('B:', B)
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print('X:', X)
print('a + X:', a + X)  # 矩阵的值加上标量
print('a * X:', a * X)
print((a * X).shape)

print('5.矩阵的sum运算')
print('A:', A)
print('A.shape:', A.shape)
print('A.sum():', A.sum())
print('A.sum(axis=0):', A.sum(axis=0))  # 沿0轴汇总以生成输出向量
print('A.sum(axis=1):', A.sum(axis=1))  # 沿1轴汇总以生成输出向量
print('A.sum(axis=1, keepdims=True)', A.sum(axis=1, keepdims=True))  # 计算总和保持轴数不变
print('A.sum(axis=[0, 1]):', A.sum(axis=[0, 1]))  # Same as `A.sum()`
print('A.mean():', A.mean())
print('A.sum() / A.numel():', A.sum() / A.numel())
