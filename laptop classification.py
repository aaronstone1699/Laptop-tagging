
import csv
import requests
import matplotlib.pyplot as plt
import cv2
import numpy as np
import shutil
from keras.applications import VGG16
from keras.layers import Input,Dense,Flatten
from keras.models import Model
from keras import models


vggmodel_or = VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)),classes=11)
vggmodel_in = VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)),classes=2)



def predict(weights,img,cl):

    if(cl==1):
        vggmodel = vggmodel_or
        
        vggmodel.trainable=False
        model = models.Sequential()
        model.add(vggmodel)
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(124,activation='relu'))
        model.add(Dense(11,activation='sigmoid'))
    else:
        vggmodel = vggmodel_in

        vggmodel.trainable=False
        model = models.Sequential()
        model.add(vggmodel)
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(124,activation='relu'))
        model.add(Dense(2,activation='sigmoid'))
    
    model.compile(optimizer='RMSProp',loss='binary_crossentropy',metrics=['accuracy'])
    model.load_weights(weights)

    return(model.predict(img))


def preprocess(path):
    img_array = cv2.imread(path)
    img_array=cv2.resize(img_array,(224,224))
    img_yuv_1 = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
    img_yuv = cv2.cvtColor(img_yuv_1,cv2.COLOR_RGB2YUV)
    y,u,v = cv2.split(img_yuv)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
    y = clahe.apply(y)
    y = cv2.GaussianBlur(y,(3,3),1)
    img_array_1 = cv2.merge((y,u,v))
    img_array = cv2.cvtColor(img_array_1,cv2.COLOR_YUV2RGB)
    img = cv2.resize(img_array,(224,224))
    return(img)
            
def check_logo(img):
    brands = []
    intel = intel_classifier.detectMultiScale(img,2,5)
    google = google_classifier.detectMultiScale(img,2,5)
    windows = windows_classifier.detectMultiScale(img,2,5)
    if(len(intel)>0):
        brands.append('intel')
    if(len(google)>0):
        brands.append('google')
    if(len(windows)>0):
        brands.append('windows')
    return(brands)


path = input("enter the path of csv file with image's url : ")
weights_orientation = input('enter the path for the weights of orientation detection : ')
weights_in_use = input('enter the path for the weights of in use detection : ')
intel_path = input('enter the path to intel classifier : ')
google_path = input('enter the path to google classifier : ')
windows_path = input('enter the path to windows classifier : ')


intel_classifier = cv2.CascadeClassifier(intel_path)
google_classifier = cv2.CascadeClassifier(google_path)
windows_classifier = cv2.CascadeClassifier(windows_path)

in_use_classes = ['in use','not in use']
orientation_classes = ['flat at angle','back','back at angle','closed at angle','flat back','flat front','front','inverted','at angle','side','side open']

with open(path, newline='') as csvfile:
    imreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    num=0
    results = []
    for row in imreader:
        tags = dict()
        num=num+1
        image_url = list(row[0].split(','))[1]
        filename = str(num)+".jpg"

        tags['id'] = num
        tags['url'] = image_url

        # Open the url image, set stream to True, this will return the stream content.
        r = requests.get(image_url, stream = True)
        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            with open('temp.jpg', 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            print('file download sucessful') 
            proc_img = preprocess('temp.jpg')
            poc_img.reshape(-1,224,224,3)
            tags['orientation'] = orientation_classes[predict(weights_orientation,proc_img,0).index(1)]
            tags['in_use'] = in_use_classes[predict(weights_in_use,proc_img,1).index(1)]
            tags['brands'] = check_logos(proc_img)

        else :
            print('file download failed ;file not found')
        results.append(tags)

