import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from CustomDataImage import CatsAndDogsDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
in_channel = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 32
num_epochs = 2

# Load Data
dataset = CatsAndDogsDataset(csv_file='data/cats_dogs.csv', root_dir='data/cats_dogs_resized',
                             transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [7, 3])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

for params in model.parameters():
    params.requires_grad = False

num_f = model.fc.in_features
model.fc = nn.Linear(num_f, 2)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_ix, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss)

        loss.backward()

        optimizer.step()

    print('Cost', sum(losses) / len(losses))


# Check Accuracy on Training Set and see how good your model is
def check_accuracy(loader, model):
    num_correct = 0
    num_sample = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            score = model(x)
            _, prediction = score.max(1)
            num_correct += (prediction == y).sum()
            num_sample += prediction.size(0)
            print("accuracy")
            print(num_correct / num_sample)
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
