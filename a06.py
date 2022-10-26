import torch

x = torch.arange(4.0,requires_grad=True)
print(x)
y = 2*torch.dot(x,x)
print(y)
print(y.backward())
print(x.grad)#tensor([ 0.,  4.,  8., 12.])
# if x.grad == 4*x:# 这么写不行，因为是布尔值组成的数组。tensor([True,True...])
#     print('duile')
# else:
#     print('cuole')
print(x.grad==4*x)#tensor([True, True, True, True])

x.grad.zero_()# 如果注释了，不全赋值0，梯度会累加tensor([ 1.,  5.,  9., 13.])
y = x.sum()
y.backward()
print(x.grad)

# 非标量变量的反向传播
x.grad.zero_()
print('x:', x)
y = x * x
print(y)
y.sum().backward()
print('x.grad:', x.grad)

def f(a):
    b = a * 2
    print(b.norm())
    while b.norm() < 1000:  # 求L2范数：元素平方和的平方根
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c