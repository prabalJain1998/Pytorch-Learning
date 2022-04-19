import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32,
                         device = device, requires_grad=True)

# print(my_tensor)
# print(my_tensor.device)
# print(my_tensor.dtype)
# print(my_tensor.shape)
# print(my_tensor.requires_grad)

# Other Common Initialization Methods
# Sometimes non zero initialization occurs for torch.empty as it is random.
x = torch.empty(size = (3,3))
x = torch.zeros((3,3))
# print(x)
# print(y)

# values b/w 0 and 1
x = torch.rand((3,3))
x = torch.ones((3,3))
# Identity Matrix.
x = torch.eye(2,3)

# end is non Inclusive.
x= torch.arange(start =0, end = 5, step = 1)

x =  torch.linspace(start = 0.1, end = 1, steps = 10)

x = torch.empty(size = (1,5)).normal_(mean = 0, std = 1)
x = torch.empty(size = (1,5)).uniform_(0,1)

x = torch.diag(torch.ones(3))

# How to initialize tensors to different types and convert them to different types :
# x = torch.arange(4)
# print(x.bool())
# print(x.short())
# print(x.long())
# print(x.half())
# print(x.float())
# print(x.double())

# Conversion
# import numpy as np
# np_array = np.zeros((5,5))
# tensor = torch.from_numpy(np_array)
# np_array_back = tensor.numpy()
# print(np_array_back)


# Tensor Math and Comparision Operations
x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])

# Addition
# z1  = torch.empty(3)
# torch.add(x,y, out = z1)
# z2 = torch.add(x,y)
z = x+y

# Subtraction
z = x-y

# Division
z = torch.true_divide(x,y)
#Element wise division if they have equal shape

#inplace operation
# t = torch.zeros(3)
# t.add_(x)

# t += x

# Exponentiation
# z = x.pow(2)
z = x ** 2

# Simple Comparision
z = x>0

# Matrix Multiplication
x = torch.rand((2,3))
y = torch.rand((3,5))
z = x.mm(y)

# Matrix Exponentiation
matrix_exp = torch.rand((5,5))
# Multiplying same matrix 3 times.
#print(matrix_exp.matrix_power(3))

# Element Wise Multiplication
x = torch.tensor([1,2,3])
y= torch.tensor([4,5,6])
z = x*y
# print(z)

# Dot Product
z = torch.dot(x,y)
# print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor_1 = torch.rand((batch, n,m))
tensor_2 = torch.rand((batch,m ,p))
out_bmm = torch.bmm(tensor_1,tensor_2)
# print(out_bmm.shape)

# Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
# Here x2 row's will be broadcasted to 4 more rows to match it with the first tensor
z = x1-x2
z = x1**x2
# print(z)

# Other useful Tensor Operation here dim argument is for which dimension you want to consider
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
# Here z will be index
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)

mean_x = torch.mean(x.float(), dim=0)

# For Equality
z = torch.eq(x,y)
# print(z)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

# Clamping : if will make value 0 if any value in x is less than 0 and make it 10 if any value is > 10
# z = torch.clamp(x, min = 0, max =10)

# x = torch.tensor([1,0,1,1,0], dtype=float)
# z = torch.any(x)
# z = torch.all(x)
# print(z)

# print(torch.zeros(3,2,3))





