import numpy as np
import torch
from PIL import Image
import os

model = torch.load('model_1.pth')


def testing_one_sample(model):
    for img_name in os.listdir('./SingleImageTest'):
        img = Image.open('./SingleImageTest/' + img_name).convert('RGB')
        img = img.resize((224,224))
        model.eval()
        im = np.array(img)
        im = im.transpose(2, 0, 1)
        my_tensor = torch.tensor(im)
        my_tensor.unsqueeze_(0)
        print(my_tensor.shape)
        score = model(my_tensor)
        _, prediction = score.max(1)
        print(prediction)

testing_one_sample(model)
