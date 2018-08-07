# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 16:25:43 2018

@author: dahoreAm
"""

import os
import pandas as pd
import cv2
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Activation
from keras.optimizers import SGD,RMSprop,Adadelta
from sklearn import preprocessing

img_data_list=[]
for file_type in ['C:/Users/DahoreAm/Downloads/Beginner/Training/All']:
                  for images in os.listdir(file_type):
                      current_path=str(file_type)+'/'+str(images)
                      print(current_path)
                      from keras.preprocessing import image
                      img=image.load_img(str(current_path),
                                         target_size=(224,224))
                      img=image.img_to_array(img)
                      img=np.expand_dims(img,axis=0)
                      img=preprocess_input(img)
                      img_data_list.append(img)
            
img_data=np.array(img_data_list)
img_data=img_data.astype('float32')
img_data/=255
print(img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print(img_data.shape)
img_data=img_data[0]
print(img_data.shape)

num_samples=img_data.shape[0]

label=np.ones((num_samples,),dtype='int64')

label[0:695]=0                       #antelope
label[695:951]=1                      #bat
label[951:1084]=2                      #beaver
label[1084:1502]=3                      #bobcat
label[1502:2108]=4                      #buffalo
label[2108:2494]=5                      #chihuahua
label[2494:2971]=6                      #chimpanzee
label[2971:3651]=7                      #collie
label[3651:4010]=8                      #dalmatian
label[4010:4697]=9                      #german+shepherd
label[4697:5280]=10                      #grizzly+bear
label[5280:5744]=11                      #hippopotamus
label[5744:6855]=12                      #horse
label[6855:7048]=13                      #killer+whale
label[7048:7108]=14                      #mole
label[7108:7584]=15                      #moose
label[7584:7708]=16                      #mouse
label[7708:8227]=17                      #otter
label[8227:8728]=18                      #ox
label[8728:9219]=19                      #persian+cat
label[9219:9565]=20                      #raccoon
label[9565:9785]=21                      #rat
label[9785:10262]=22                      #rhinoceros
label[10262:10927]=23                      #seal
label[10927:11268]=24                      #siamese+cat
label[11268:11457]=25                      #spider+monkey
label[11457:12265]=26                      #squirrel
label[12265:12413]=27                      #walrus
label[12413:12597]=28                      #weasel
label[12597:13000]=29                      #wolf

Y=np_utils.to_categorical(labels,30)       
                
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

x,y=shuffle(img_data,Y,random_state=2)

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=4)

input_image=Input(shape=(224,224,3))

model=VGG16(input_tensor=input_image,include_top=True,weights='imagenet')
model.summary()

last_layer=model.get_layer('fc2').output
out=Dense(30,activation='softmax',name='output')(last_layer)
custom_vgg_model=Model(input_image,out)
custom_vgg_model.summary()
from keras.optimizers import Adam
ADAM=Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=None, decay=0.0, amsgrad=False)

custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='ADAM',metrics=['accuracy'])

for layers in custom_vgg_model.layers[:-1]:
    layer.trainable=False
    
custom_vgg_model.layers[3].trainable

hist=custom_vgg_model.fit(X_train,Y_train,batch_size=32,validation_data=(X_test,Y_test),epochs=40)

