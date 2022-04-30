import gradio as gr
import numpy as np
import torch
from PIL import Image
import os

model = torch.load('model_1.pth')


def testing_one_sample(model):
    for img_name in os.listdir('./SingleImageTest'):
        img = Image.open('./SingleImageTest/' + img_name).convert('RGB')
        img = img.resize((224, 224))
        model.eval()
        im = np.array(img)
        im = im.transpose(2, 0, 1)
        my_tensor = torch.tensor(im)
        my_tensor.unsqueeze_(0)
        print(my_tensor.shape)
        score = model(my_tensor)
        print(score)
        _, prediction = score.max(1)
        print(prediction)


class_names = ['Covid', 'Normal', 'Viral Pneumonia']
class_map = {'Covid': 0, 'Normal': 1, 'Viral Pneumonia': 2}


def predict_image(img):
    img.convert('RGB')
    img = img.resize((224, 224))
    model.eval()
    im = np.array(img)
    im = im.transpose(2, 0, 1)
    my_tensor = torch.tensor(im)
    my_tensor.unsqueeze_(0)
    score = model(my_tensor)
    _, prediction = score.max(1)
    my_dict = {}
    for cls in class_names:
        if class_map[cls] == prediction:
            my_dict[cls] = 1
        else:
            my_dict[cls] = 0
    return my_dict


image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label(num_top_classes=3)

iface = gr.Interface(predict_image, inputs=image, outputs='label')

iface.launch()
