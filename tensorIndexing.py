# tensor Indexing
import torch
batch_size = 10
features = 25

x = torch.rand((batch_size,features))

# x[0,:]
# print(x[0].shape)
# print(x[:, 0].shape)

# print(x[2, 0:10].shape)
# print(x)
# x[0,0] = 10
# print(x)

# Fancy Indexing
# indices = [2,4,6]
# x = torch.arange(10)
# print(x[indices])
x = torch.rand((3,5))
# rows = torch.tensor([1,0])
# col = torch.tensor([4,2])
# print(x[rows,col])

# x = torch.arange(10)
# print(x[(x<2) | (x>8)])

# print(x[x.remainder(2) == 0])  # Even elements are selected.

# Useful Operations
# if --- else
# print(torch.where(x>5, x, x*2))
# print(torch.tensor([0,1,2,4,4,4,4]).unique())
print(x)
print(x.ndimension())
print(x.numel()) # Number of element in the X


