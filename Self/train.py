import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import GoogLeNet_Weights

from CustomDataSet import CovidDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
in_channel = 3
num_classes = 3
learning_rate = 1e-3
batch_size = 32
num_epochs = 50

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load Data
dataset = CovidDataset(csv_file='data.csv', transform=transform)

train_set, test_set = torch.utils.data.random_split(dataset, [300, 17])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
vgg16 = models.vgg16(weights=True)

for param in vgg16.features.parameters():
    param.require_grad = False

num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1]
features.extend([nn.Linear(num_features, 3)])  # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier

# num_f = model.fc.in_features
# model.fc = nn.Linear(num_f, 3)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.parameters(), lr=learning_rate)

# Train Network
overall_loss = []
vgg16.train()
for epoch in range(num_epochs):
    losses = []

    for batch_ix, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        scores = vgg16(data)
        print(batch_ix)
        loss = criterion(scores, targets)
        losses.append(loss)
        loss.backward()
        optimizer.step()
    print('Epoch ', epoch, ' : ', 'Cost ', sum(losses) / len(losses))
    overall_loss.append(sum(losses) / len(losses))


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


# check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)

torch.save(vgg16, 'model_1.pth')
