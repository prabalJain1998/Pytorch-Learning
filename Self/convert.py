from PIL import Image
import os
import shutil

for label in (os.listdir('./train')):
    for img_name in os.listdir('./train/'+label):
        img = Image.open('./train/'+label+'/'+img_name).convert('RGB')
        print(img_name)
        img.save('./trainConverted/'+label+'/'+img_name)

for label in (os.listdir('./test')):
    for img_name in os.listdir('./test/'+label):
        img = Image.open('./test/'+label+'/'+img_name).convert('RGB')
        print(img_name)
        img.save('./testConverted/'+label+'/'+img_name)


