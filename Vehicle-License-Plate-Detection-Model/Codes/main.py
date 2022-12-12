import cv2
import numpy as np
import pandas as pd
import random
import imutils
import joblib
import glob
import re
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

###############################################################################################################################################
## PLATE DETECTION

'''
   folder (original)         dirName (saved)
   ------------------        ---------------
1) Kaggle              -     Kaggle_cropped
2) Kaggle_subset       -     Kaggle_output     --> subset
3) Malaysia            -     Malaysia_cropped   
4) Malaysia_subset     -     Malaysia_output   --> subset

'''

# fetch current working directory
current_path = os.getcwd()

## make changes based on each different testing

# folder is fetched where license plate images are located dynamically
folder = os.path.join(current_path, 'Kaggle_subset')  #"Kaggle_subset" is the folder name contains the subset car images  (change the folder name for different testing)
# folder to save cropped license plate
dirName = 'Kaggle_output'   #"Kaggle_output" is the folder name to store the cropped license plates (change the folder name for each save)
count = 1

for filename in os.listdir(folder):
    img = cv2.imread((os.path.join(folder, filename)))
    write_folder = os.path.join(current_path, dirName)

    # name of file which is to be written to chosed folder
    image_name_to_write = (os.path.join(write_folder,'Cropped_{}.png'.format(count)))

    # convert to Grayscale Image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # noise removal with bilateral filter
    biFilter = cv2.bilateralFilter(gray, 13, 17, 17)

    # Canny edge detection
    canny = cv2.Canny(biFilter, 170, 200)

    # find contours based on Edges
    contours = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort contours based on minimum area 30
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    # create copy of original image to draw contours
    image_copy = img.copy()
    _ = cv2.drawContours(image_copy, contours, -1, (255,0,255),2)

    contour_with_license_plate = None
    license_plate = None
    x = None
    y = None
    w = None
    h = None
    counts=1

    # loop over contours to find the best possible approximate contour of license plate
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4:
            contour_with_license_plate = approx
            x, y, w, h = cv2.boundingRect(c)
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            license_plate = gray[y:y + h, x:x + w]
            break

    if license_plate is None:
        detected = 0
        print("No contour detected!")
    else:
        detected = 1

    if detected == 1:
        image_copy = img.copy()
        cv2.drawContours(image_copy, contours, -1, (255,0,255),2)
        mask = np.zeros(gray.shape,np.uint8)
        cv2.drawContours(mask,[contour_with_license_plate],0,255,-1)

        (x, y) = np.where(mask == 255)
        (top_x, top_y) = (np.min(x), np.min(y))
        (bottom_x, bottom_y) = (np.max(x), np.max(y))
        crop = img[top_x:bottom_x+1, top_y:bottom_y+1]
        crop2 = gray[top_x:bottom_x+1, top_y:bottom_y+1]  #cropped license plate

    # image file is written using cv2.imwrite function
    write_images = cv2.imwrite(image_name_to_write, crop2)
    count = count+1

#######################################################################################################################################################################
#######################################################################################################################################################################

## MODAL MENU
print(' ')
print('*** Choice Selection ***')
print('Choice 1. Kaggle Dataset')
print('Choice 2. Kaggle Subset Data')
print('Choice 3. Malaysia Dataset')
print('Choice 4. Malaysia Subset Data')
print(' ')
choice = int(input('Enter a choice (1-4): '))
print(' ')

####################################################################################
# READ IMAGE DIRECTORY

'''
    input_dir          numImages
   -----------        -----------
1) Kaggle_cropped    - 230 images
2) Kaggle_output     - 30 images   --> subset
3) Malaysia_cropped  - 76 images   
4) Malaysia_output   - 30 images   --> subset

'''

## ** <PLEASE MAKE CHANGES ACCORDING TO THE IMAGE FILE DIRECTORY >** 
# make changes based on each different testing inputs
input_dir = "Kaggle_output"       
numImages = 30

onlyfiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
file = onlyfiles[0:numImages]
files =  sorted(file,key=lambda x: int(x.split('Cropped_')[1].split('.png')[0]))
# print(files)
count = 0

############################################################################################################################################
############################################################################################################################################

for i, name in enumerate(files):
    img = cv2.imread(input_dir+ '/' + name, 0)
    plate_image = cv2.convertScaleAbs(img, alpha=(255.0))


#  | | | | | | | | | | | |  | | | | | | | | | | |  | | | | | | | | | | |
#  V V V V V V V V V V V V  V V V V V V V V V V V  V V V V V V V V V V V 
## COMMENT OUT ONE OF THE BINARY BASED ON THE REQUIREMENTS FOR KAGGLE AND MALAYSIA DATASETS

##    ** <PLEASE CHANGE ACCORDINGLY WITH YOUR TESTING DATASETS> **
    binary = cv2.threshold(img, 180, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]   # FOR KAGGLE DATASETS
    #binary = cv2.threshold(img,180,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]        # FOR MALAYSIA DATASETS
    
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

    Categories=['0', '1', '2', '3', '4', '5',
                '6', '7', '8', '9', 'A', 'B',
                'C', 'D', 'E', 'F', 'G', 'H',
                'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z']
    
    flat_data_arr=[] #input array
    target_arr=[] #output array

##  ** <Please change to your directory which contains all the categories of images (CHARACTER folder)> **
    datadir='C:/Users/user/Desktop/VIP Project/Characters'  

############################################################################################################################################
############################################################################################################################################
# FOR KAGGLE DATASET

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

    df=pd.DataFrame(flat_data) 
    df['Target']=target

    x=df.iloc[:,:-1] #input data 
    y=df.iloc[:,-1] #output data

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)

    loaded_rf = joblib.load("./RF_compressed.joblib")
    loaded_rf.predict(x_test)
    
    output =[]

##########################################################################################
##########################################################################################
    if choice == 1:
        print('Choice 1 selected.')
        # RF
        for i,ch in enumerate(crop_characters):
            img_resize=resize(ch,(150,150,3))
            l=[img_resize.flatten()]
            probability=loaded_rf.predict_proba(l)
            a_list = Categories[loaded_rf.predict(l)[0]]
            output.append(a_list) #storing the result in a list

        plate_number = ''.join(output)
        print(plate_number, file=open("RF_output_Kaggle.csv", "a"))

        # ANN
        def load_keras_model(model_name):
        # Load json and create model
            json_file = open('./{}.json'.format(model_name), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # Load weights into new model
            model.load_weights("./{}.h5".format(model_name))
            return model  
    
        # store_keras_model(model, 'model_License_Plate')
        pre_trained_model = load_keras_model('model_License_Plate')
        model = pre_trained_model
        output = []
 
        def fix_dimension(img):
            new_img = np.zeros((28,28,3))
            for i in range(3):
                new_img[:,:,i] = img 
            return new_img

        for i,ch in enumerate(crop_characters): #iterating over the characters
            img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
            img = fix_dimension(img_)
            img = img.reshape(1,28,28,3) #preparing image for the model
            y_ = model.predict_classes(img)[0] #predicting the class
            character = Categories[y_] #
            output.append(character) #storing the result in a list

        plate_number = ''.join(output)
        print(plate_number, file=open("ANN_output_Kaggle.csv", "a"))

        # OCR
        #path for data_drive
        path_for_license_plates = "C:/Users/user/Desktop/VIP Project/Kaggle_cropped/*"
        list_license_plates = []
        predicted_license_plates = []

        #searching through the directory and printing out the list of file names
        file_list = glob.glob(path_for_license_plates)
        #print(file_list)
        #inidicator
        print(len(file_list))

        #testing with OCR 
        for path_to_license_plate in glob.glob(path_for_license_plates, recursive = True):
            
            license_plate_file2 = path_to_license_plate.split("/")[-1]
            license_plate_file1 = license_plate_file2.split("\\")[-1]
            license_plate_file = license_plate_file1.split("_")[-1]
        
            license_plate, _ = os.path.splitext(license_plate_file)
        
            #Here we append the actual license plate to a list
        
            list_license_plates.append(license_plate)
            
            '''
            Read each license plate image file using openCV
            '''
            img = cv2.imread(path_to_license_plate)
            
            predicted_result = pytesseract.image_to_string(img, lang='eng',config ='--psm 7 --oem 3')
            
            filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
            predicted_license_plates.append(filter_predicted_result)

        #setting up a datframe and cleaning the data 
        df1 = pd.DataFrame({'a':list_license_plates,'b':predicted_license_plates})

        df1['a'] = df1['a'].astype(int)
        df2 = df1.sort_values(by=['a'])

        df3=df2.set_index('a')
        p = re.compile(r'[^A-Z0-9]')
        df3['b'] = [p.sub('',x)for x in df3['b']]
        df3.to_csv('OCR_output_Kaggle.csv', header=False, index=False)

##########################################################################################
##########################################################################################
    elif choice == 2:
        print('Choice 2 selected.')
        # RF
        for i,ch in enumerate(crop_characters):
            img_resize=resize(ch,(150,150,3))
            l=[img_resize.flatten()]
            probability=loaded_rf.predict_proba(l)
            a_list = Categories[loaded_rf.predict(l)[0]]
            output.append(a_list) #storing the result in a list

        plate_number = ''.join(output)
        print(plate_number, file=open("RF_subset_Kaggle.csv", "a"))

        # ANN
        def load_keras_model(model_name):
        # Load json and create model
            json_file = open('./{}.json'.format(model_name), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # Load weights into new model
            model.load_weights("./{}.h5".format(model_name))
            return model  
    
        # store_keras_model(model, 'model_License_Plate')
        pre_trained_model = load_keras_model('model_License_Plate')
        model = pre_trained_model
        output = []
 
        def fix_dimension(img):
            new_img = np.zeros((28,28,3))
            for i in range(3):
                new_img[:,:,i] = img 
            return new_img

        for i,ch in enumerate(crop_characters): #iterating over the characters
            img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
            img = fix_dimension(img_)
            img = img.reshape(1,28,28,3) #preparing image for the model
            y_ = model.predict_classes(img)[0] #predicting the class
            character = Categories[y_] #
            output.append(character) #storing the result in a list

        plate_number = ''.join(output)
        print(plate_number, file=open("ANN_subset_Kaggle.csv", "a"))

        # OCR
        #path for data_drive
        path_for_license_plates = "C:/Users/user/Desktop/VIP Project/Kaggle_output/*"
        list_license_plates = []
        predicted_license_plates = []

        #searching through the directory and printing out the list of file names
        file_list = glob.glob(path_for_license_plates)
        #print(file_list)
        #inidicator
        print(len(file_list))

        #testing with OCR 
        for path_to_license_plate in glob.glob(path_for_license_plates, recursive = True):
            
            license_plate_file2 = path_to_license_plate.split("/")[-1]
            license_plate_file1 = license_plate_file2.split("\\")[-1]
            license_plate_file = license_plate_file1.split("_")[-1]
        
            license_plate, _ = os.path.splitext(license_plate_file)
        
            #Here we append the actual license plate to a list
        
            list_license_plates.append(license_plate)
            
            '''
            Read each license plate image file using openCV
            '''
            img = cv2.imread(path_to_license_plate)
            
            predicted_result = pytesseract.image_to_string(img, lang='eng',config ='--psm 7 --oem 3')
            
            filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
            predicted_license_plates.append(filter_predicted_result)

        #setting up a datframe and cleaning the data 
        df1 = pd.DataFrame({'a':list_license_plates,'b':predicted_license_plates})

        df1['a'] = df1['a'].astype(int)
        df2 = df1.sort_values(by=['a'])

        df3=df2.set_index('a')
        p = re.compile(r'[^A-Z0-9]')
        df3['b'] = [p.sub('',x)for x in df3['b']]
        df3.to_csv('OCR_subset_Kaggle.csv', header=False, index=False)
    

############################################################################################################################################
############################################################################################################################################
# FOR MALAYSIA DATASET

    elif choice == 3:
        print('Choice 3 selected.')
        # RF
        for i,ch in enumerate(crop_characters):
            img_resize=resize(ch,(150,150,3))
            l=[img_resize.flatten()]
            probability=loaded_rf.predict_proba(l)
            a_list = Categories[loaded_rf.predict(l)[0]]
            output.append(a_list) #storing the result in a list

        plate_number = ''.join(output)
        print(plate_number, file=open("RF_output_MY.csv", "a"))

        # ANN
        def load_keras_model(model_name):
        # Load json and create model
            json_file = open('./{}.json'.format(model_name), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # Load weights into new model
            model.load_weights("./{}.h5".format(model_name))
            return model  
    
        # store_keras_model(model, 'model_License_Plate')
        pre_trained_model = load_keras_model('model_License_Plate')
        model = pre_trained_model
        output = []
 
        def fix_dimension(img):
            new_img = np.zeros((28,28,3))
            for i in range(3):
                new_img[:,:,i] = img 
            return new_img

        for i,ch in enumerate(crop_characters): #iterating over the characters
            img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
            img = fix_dimension(img_)
            img = img.reshape(1,28,28,3) #preparing image for the model
            y_ = model.predict_classes(img)[0] #predicting the class
            character = Categories[y_] #
            output.append(character) #storing the result in a list

        plate_number = ''.join(output)
        print(plate_number, file=open("ANN_output_MY.csv", "a"))

        # OCR
        #path for data_drive
        path_for_license_plates = "C:/Users/user/Desktop/VIP Project/Malaysia_cropped/*"
        list_license_plates = []
        predicted_license_plates = []

        #searching through the directory and printing out the list of file names
        file_list = glob.glob(path_for_license_plates)
        #print(file_list)
        #inidicator
        print(len(file_list))

        #testing with OCR 
        for path_to_license_plate in glob.glob(path_for_license_plates, recursive = True):
            
            license_plate_file2 = path_to_license_plate.split("/")[-1]
            license_plate_file1 = license_plate_file2.split("\\")[-1]
            license_plate_file = license_plate_file1.split("_")[-1]
        
            license_plate, _ = os.path.splitext(license_plate_file)
        
            #Here we append the actual license plate to a list
        
            list_license_plates.append(license_plate)
            
            '''
            Read each license plate image file using openCV
            '''
            img = cv2.imread(path_to_license_plate)
            
            predicted_result = pytesseract.image_to_string(img, lang='eng',config ='--psm 7 --oem 3')
            
            filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
            predicted_license_plates.append(filter_predicted_result)

        #setting up a datframe and cleaning the data 
        df1 = pd.DataFrame({'a':list_license_plates,'b':predicted_license_plates})

        df1['a'] = df1['a'].astype(int)
        df2 = df1.sort_values(by=['a'])

        df3=df2.set_index('a')
        p = re.compile(r'[^A-Z0-9]')
        df3['b'] = [p.sub('',x)for x in df3['b']]
        df3.to_csv('OCR_output_MY.csv', header=False, index=False)

##########################################################################################
##########################################################################################
    elif choice == 4:
        print('Choice 4 selected.')
        # RF
        for i,ch in enumerate(crop_characters):
            img_resize=resize(ch,(150,150,3))
            l=[img_resize.flatten()]
            probability=loaded_rf.predict_proba(l)
            a_list = Categories[loaded_rf.predict(l)[0]]
            output.append(a_list) #storing the result in a list

        plate_number = ''.join(output)
        print(plate_number, file=open("RF_subset_MY.csv", "a"))

        # ANN
        def load_keras_model(model_name):
        # Load json and create model
            json_file = open('./{}.json'.format(model_name), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # Load weights into new model
            model.load_weights("./{}.h5".format(model_name))
            return model  
    
        # store_keras_model(model, 'model_License_Plate')
        pre_trained_model = load_keras_model('model_License_Plate')
        model = pre_trained_model
        output = []
 
        def fix_dimension(img):
            new_img = np.zeros((28,28,3))
            for i in range(3):
                new_img[:,:,i] = img 
            return new_img

        for i,ch in enumerate(crop_characters): #iterating over the characters
            img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
            img = fix_dimension(img_)
            img = img.reshape(1,28,28,3) #preparing image for the model
            y_ = model.predict_classes(img)[0] #predicting the class
            character = Categories[y_] #
            output.append(character) #storing the result in a list

        plate_number = ''.join(output)
        print(plate_number, file=open("ANN_subset_MY.csv", "a"))

        # OCR
        #path for data_drive
        path_for_license_plates = "C:/Users/user/Desktop/VIP Project/Malaysia_output/*"
        list_license_plates = []
        predicted_license_plates = []

        #searching through the directory and printing out the list of file names
        file_list = glob.glob(path_for_license_plates)
        #print(file_list)
        #inidicator
        print(len(file_list))

        #testing with OCR 
        for path_to_license_plate in glob.glob(path_for_license_plates, recursive = True):
            
            license_plate_file2 = path_to_license_plate.split("/")[-1]
            license_plate_file1 = license_plate_file2.split("\\")[-1]
            license_plate_file = license_plate_file1.split("_")[-1]
        
            license_plate, _ = os.path.splitext(license_plate_file)
        
            #Here we append the actual license plate to a list
        
            list_license_plates.append(license_plate)
            
            '''
            Read each license plate image file using openCV
            '''
            img = cv2.imread(path_to_license_plate)
            
            predicted_result = pytesseract.image_to_string(img, lang='eng',config ='--psm 7 --oem 3')
            
            filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
            predicted_license_plates.append(filter_predicted_result)

        #setting up a datframe and cleaning the data 
        df1 = pd.DataFrame({'a':list_license_plates,'b':predicted_license_plates})

        df1['a'] = df1['a'].astype(int)
        df2 = df1.sort_values(by=['a'])

        df3=df2.set_index('a')
        p = re.compile(r'[^A-Z0-9]')
        df3['b'] = [p.sub('',x)for x in df3['b']]
        df3.to_csv('OCR_subset_MY.csv', header=False, index=False)

    else:
        print('Error! Please input an integer value only from 1 to 4 (1-4).')
        break
