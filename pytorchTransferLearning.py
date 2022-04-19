import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


device = torch.device('cpu')
model = torchvision.models.vgg16(pretrained=True)

for params in model.parameters():
    params.requires_grad = False

# print(model)
model.avgpool = Identity()
model.classifier = nn.Sequential(
    nn.Linear(512, 100),
    nn.Dropout(0.5),
    nn.Linear(100, 10)
)
print(model)

model.to(device)
# Load Data

# Hyperparameters
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Load Data
train_dataset = datasets.CIFAR10(root='CIFAR/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='CIFAR/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

# Train Network
for epoch in range(num_epochs):
    print("Epoch ", epoch)
    for batch_index, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Check Accuracy on Training Set and see how good your model is
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking Acc on Train Data")
    else:
        print("Checking Acc. on Test Data")
    num_correct = 0
    num_sample = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, prediction = scores.max(1)
            num_correct += (prediction == y).sum()
            num_sample += prediction.size(0)
            print("accuracy")
            print(num_correct / num_sample)
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
