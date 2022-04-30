import pandas as pd
import numpy as np
import os

class_map = {'Covid': 0, 'Normal': 1, 'Viral Pneumonia': 2}

ol = []
# For training Data
for label in os.listdir('./trainConverted'):
    new_path = os.path.join('./trainConverted', label)
    for img in os.listdir(new_path):
        ol.append(['./trainConverted/' + label + '/' + img, class_map[label]])

# For testing Data
for label in os.listdir('./testConverted'):
    new_path = os.path.join('./testConverted', label)
    for img in os.listdir(new_path):
        ol.append(['./testConverted/' + label + '/' + img, class_map[label]])

df = pd.DataFrame(ol, columns=['Path', 'Label'])
df.to_csv('data.csv', index=False)
