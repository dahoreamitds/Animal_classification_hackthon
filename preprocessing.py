import os
import pandas as pd
import cv2
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Activation
from keras.optimizers import SGD,RMSprop,Adadelta
from sklearn import preprocessing

dataset=pd.read_csv('C:/Users/DahoreAm/Downloads/Beginner/train.csv')
dataset.head(20)
dataset[dataset['Animal']=='bat']
dataset['Animal'].unique()

for folders in dataset['Animal'].unique():
    os.makedirs('C:/Users/DahoreAm/Downloads/Beginner/'+str(folders))

dataset['Animal']

data=dataset.loc[dataset['Image_id']=='Img-10.jpg','Animal']

i=0
for file_type in ['C:/Users/DahoreAm/Downloads/Beginner/train']:
    for images in os.listdir(str(file_type)):
        current_path=str(file_type)+'/'+str(images)
        image_name=current_path[43:]
        #print(image_name)
        image=cv2.imread(str(current_path))
        data=dataset.loc[dataset['Image_id']==image_name,'Animal']
        np_data=np.array(data)
        if (np_data=='hippopotamus'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'hippopotamus/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='squirrel'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'squirrel/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='grizzly+bear'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'grizzly+bear/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='ox'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'ox/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='german+shepherd'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'german+shepherd/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='buffalo'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'buffalo/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)                       
        elif(np_data=='otter'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'otter/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='bobcat'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'bobcat/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='wolf'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'wolf/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='persian+cat'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'persian+cat/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='collie'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'collie/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='antelope'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'antelope/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='seal'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'seal/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='dalmatian'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'dalmatian/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='siamese+cat'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'siamese+cat/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='moose'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'moose/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='horse'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'horse/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='killer+whale'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'killer+whale/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='mouse'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'mouse/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='walrus'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'walrus/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='beaver'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'beaver/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='rhinoceros'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'rhinoceros/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='chimpanzee'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'chimpanzee/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='weasel'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'weasel/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='spider+monkey'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'spider+monkey/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='raccoon'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'raccoon/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='rat'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'rat/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='chihuahua'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'chihuahua/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='mole'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'mole/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)
        elif(np_data=='bat'):
            cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/'+'bat/'+
                        str(np_data[0])+'_'+str(i)+'.jpg',image)

        #cv2.imwrite(file_type+'/'+str(np_data[0])+'_'+str(i)+'.jpg',image)

        i=i+1

i=0
for file_type in ['C:/Users/DahoreAm/Downloads/Beginner/Training/wolf']:
    for images in os.listdir(file_type):
        currently_path=str(file_type)+'/'+str(images)
        image=cv2.imread(str(currently_path))
        cv2.imwrite('C:/Users/DahoreAm/Downloads/Beginner/Training/All/'+str(i)+'.jpg',image)
        i=i+1
        print(i)



