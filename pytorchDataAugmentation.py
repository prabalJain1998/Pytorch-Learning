import torch
import  torchvision.transforms as transforms
from torchvision.utils import save_image
from CustomDataImage import CatsAndDogsDataset

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees = 45),
    transforms.RandomVerticalFlip(p = 0.05),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.0, 0.0, 0.0], std = [1.0,1.0,1.0])
])

dataset = CatsAndDogsDataset(csv_file='data/cats_dogs.csv', root_dir='data/cats_dogs_resized', transform= my_transforms)

for image, label in dataset:
    print(image.shape)
    print(label)

