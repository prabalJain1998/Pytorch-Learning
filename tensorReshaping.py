import torch

# x = torch.arange(9)
# # x_3x3 = x.view(3,3)
# x_3x3 = x.reshape(3,3)

# x1 = torch.rand((2,5))
# x2 = torch.rand((2,5))
# z = torch.cat((x1,x2), dim = 0)
# print(z.shape)

batch = 24
x = torch.rand((batch, 2,4))
# we want batch dimension same but flatten the other two dimension
# print(x.reshape(batch , -1).shape)

# Switch the Axis of x to batch ,4, 2
z = x.permute(0, 2,1) # Put index of dimension
# print(z.shape)
x = torch.arange(10)
# print(x)
# print(x.unsqueeze(0).shape)
# print(x.unsqueeze(1).shape)
# print(torch.arange(10).unsqueeze(0).unsqueeze(1).shape)
