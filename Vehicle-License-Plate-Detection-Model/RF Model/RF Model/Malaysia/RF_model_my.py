###################################################################################################################################
# MODEL 
####################################################################################################################################
import cv2
import joblib
import numpy as np
import pandas as pd
#%matplotlib
import matplotlib
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import warnings 
import matplotlib.gridspec as gridspec
warnings.filterwarnings(action= 'ignore')
from os import listdir
import matplotlib.cm as cm
from os.path import isfile, join, splitext

import tensorflow as tf
from sklearn.metrics import f1_score 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D, Activation
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler

import os
from skimage.transform import resize
from skimage.io import imread

input_dir = "my"
# output_dir = "output"
numImages = 230

onlyfiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
file = onlyfiles[0:numImages]
files =  sorted(file,key=lambda x: int(x.split('Cropped_')[1].split('.png')[0]))
count = 0

for i, name in enumerate(files):
    img = cv2.imread(input_dir+ '/' + name, 0)
    plate_image = cv2.convertScaleAbs(img, alpha=(255.0))
    binary = cv2.threshold(img, 180, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    def sort_contours(cnts,reverse = False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
        return cnts

    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()

    # Initialize a list which will be used to append charater image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=3.5: # Only select contour with defined ratio
            if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                # Sperate number and gibe prediction
                curr_num = binary[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

###############################################################################################################################################

    Categories=['0', '1', '2', '3', '4', '5',
                '6', '7', '8', '9', 'A', 'B',
                'C', 'D', 'E', 'F', 'G', 'H',
                'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z']

    flat_data_arr=[] #input array
    target_arr=[] #output array
    
    datadir='C:/Users/Jia Yu/Desktop/Sem 3/VIP/Project/Project/Cropped_License_Plate/Characters' 
    # please change to your dir which contains all the categories of images (CHARACTER folder)
    
    for i in Categories:

        #print(f'loading... category : {i}')
        path=os.path.join(datadir,i)
        for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(150,150,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
        #print(f'loaded category:{i} successfully')

    flat_data=np.array(flat_data_arr)
    target=np.array(target_arr)

    df=pd.DataFrame(flat_data) #dataframe
    df['Target']=target

    x=df.iloc[:,:-1] #input data 
    y=df.iloc[:,-1] #output data

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)

    loaded_rf = joblib.load("./RF_compressed_my.joblib")
    loaded_rf.predict(x_test)
    
    output =[]
    
    for i,ch in enumerate(crop_characters):
        img_resize=resize(ch,(150,150,3))
        l=[img_resize.flatten()]
        probability=loaded_rf.predict_proba(l)
        a_list = Categories[loaded_rf.predict(l)[0]]
        output.append(a_list) #storing the result in a list

    plate_number = ''.join(output)
    print(plate_number, file=open("output_RF_my.csv", "a"))