import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn


# Methods for dealing with Imbalanced Dataset
# 1. Oversampling
# 2. Class Weighting.

# For Class Weighting
# loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 50]))

# For Oversampling
def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(root_dir, transform=my_transforms)
    class_weights = [1, 50]
    sample_weights = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        cw = class_weights[label]
        sample_weights[idx] = cw

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader


def main():
    loader = get_loader(root_dir="./dataset", batch_size=8)
    for data, labels in loader:
        print(labels)


if __name__ == "__main__":
    main()
