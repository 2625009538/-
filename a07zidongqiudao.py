import torch
x = torch.arange(4.0,requires_grad=True)
# print(x.grad) # 可以直接获取x的梯度
y = 2 * torch.dot(x,x)#tensor(28., grad_fn=<MulBackward0>),后边是因为用隐式求导，所以告诉你是从x求导过来的。
# print(y)

print(y.backward())#None,必须要有y.backward来求导，才能用x.grad访问！

print(x.grad)#tensor([ 0.,  4.,  8., 12.])
print(x.grad == 4 * x)#验证一下

#因为在默认情况下，pytorch会累计梯度，所以要先清除,下划线就是重写的函数，会把0重写进去
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 如果你不想grad直接从3级到1级，只想到2级怎么办？
x.grad.zero_()
y = x * x
u = y.detach()#就是把y的结果看成一个标量，不会再从2级到1级了
z = u * x
z.sum().backward()#必须要有啊1
print(x.grad == u)

