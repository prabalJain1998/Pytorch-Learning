# 1 Didn't overfit a Single Batch First.
# So every time you setup your Neural Network Overfit a Single Batch first Just do it.

# 2. Forgot to Set training or eval mode when testing
# use model.eval() to remove the Dropout/batch-norm.
# so basically at the time of Testing do model.eval() before nad toggle it back to model.train() before training.

# 3. Forgot .zero_grad()
# optimizer.zero_grad() because you want the Gradient step on the Batch not on the whole previous batches.
# with zero_grad() gradient step on current batch ( No accumulated Gradient)

# 4. Softmax with Cross Entropy Loss.
# Don't do it

# 5. Using Bias when using Batch Norm.
# So basically when using a BatchNorm after Conv layer make Bias = False of Conv Layer.

# 6. Using View as Permute. [Hint : Use Permute]
# import torch
# x = torch.tensor([[1,2,3], [4,5,6]])
# print(x)
# print(x.view(3,2))
# print(x.permute(1,0))

# 7. Using Bad Data Augmentation

# 8. Not Shuffling the Data
# So DataLoader mai Shuffle = True karo but in case of Time Series don't do it (Mostly).

# 9. Not Normalizing the Data
# ToTensor divides everything by 255
# So use transform.Normalize(mean = [], std = []) enter values as per channel.

# 10. Not Clipping Gradients (RNNs, GRUs, LSTMs)
# After loss.backward() do torch.nn.utils.clip_grad_norm(model.parameters(), max_norm = 1)


