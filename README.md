# Image-Recognition
import tensorflow as tf
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import*
from tensorflow.keras.models import*
from tensorflow.keras.layers import*
from tensorflow.keras.utils import load_img

lt=[cv2.ROTATE_180,cv2.ROTATE_90_COUNTERCLOCKWISE,cv2.ROTATE_90_CLOCKWISE]
def brightness(img):
  value=random.uniform(0.5,2)
  hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  hsv=np.array(hsv,dtype=np.float64)
  hsv[:,:,1]=hsv[:,:,1]*value
  hsv[:,:,1][hsv[:,:,1]>255]=255
  hsv[:,:,2]=hsv[:,:,2]*value
  hsv[:,:,2][hsv[:,:,2]>255]=255
  hsv=np.array(hsv,dtype=np.uint8)
  img=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
  return img

  from google.colab import drive
drive.mount('/content/drive')

import pathlib
import glob
directory=pathlib.Path("/content/drive/My Drive/Face Recognition")
resultant="/content/augmentedimages"
import os
import random
import cv2
items=os.listdir(directory)
classes=[]
count=0
images=[]
labels=[]
for i in items:
  i1=0
  print(i)
  classes.append(i)
  path1=f"{directory}/{i}"
  a=random.randint(5,10)
  img=cv2.imread(path1)
  img=cv2.resize(img,(224,224))
  k=i.split(".")[0]
  cv2.imwrite(f"{resultant}\{k}{i1}.jpeg",img)
  i1+=1
  while a!=0:
    img=cv2.rotate(img,lt[random.randint(0,2)])

    images.append(img)
    cv2.imwrite(f"{resultant}\{k}{i1}.jpeg",img)
    i1+=1
    labels.append(count)
    if a%2==0:
       img=brightness(img)
       images.append(img)
       cv2.imwrite(f"{resultant}\{k}{i1}.jpeg",img)
       i1+=1
       labels.append(count)
    a-=1
  count+=1
images=np.array(images)
labels=np.array(labels)

images.shape

from keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.models import *
from keras.applications.vgg16 import VGG16,preprocess_input

model=VGG16(weights="imagenet")
for i in model.layers:
  i.trainable=False
len(model.layers)

model.summary()

transferVGG= Sequential()

from keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.models import *
from keras.applications.vgg16 import VGG16,preprocess_input

model=VGG16(weights="imagenet")
for i in model.layers:
  i.trainable=False
len(model.layers)

model.summary()

transferVGG= Sequential()

from torchvision import transforms
for i in range(18):
  transferVGG.add(model.layers[i])
transferVGG.add(Flatten())
transferVGG.add(Dense(512,activation="relu"))
transferVGG.add(Dense(128,activation="relu"))
transferVGG.add(Dense(12,activation="softmax")) # Changed activation to 'softmax'
transferVGG.summary()

import tensorflow as tf
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    print("call")
    if(logs.get("accuracy")>.99):
      print("\nReached %2.2f%% accuracy,so stopping training!!"%(99))
      self.model.stop_training=True
callbacks=myCallback()

transferVGG.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
transferVGG.fit(images,labels,epochs=100,callbacks=[callbacks])

transferVGG.evaluate(images,labels)

def predict(i,transferVGG,labels):
  path1=f"{directory}/{i}"
  img=cv2.imread(path1)
  img=cv2.resize(img,(224,224))
  a=np.argmax(transferVGG.predict(np.array([img])))
  img=cv2.putText(img,labels[a],(25,25),cv2.FONT_HERSHEY_SIMPLEX,1,(225,225,0),3,cv2.LINE_AA)
  plt.imshow(img)
predict("birds.jpeg",transferVGG,classes)

predict("Dora.jpeg",transferVGG,classes)

predict("pets.jpeg",transferVGG,classes)

predict("sunshine1.jpeg",transferVGG,classes)

