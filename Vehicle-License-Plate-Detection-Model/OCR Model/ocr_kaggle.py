import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

#importing pytesseract from drive
import pytesseract
pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#path for data_drive
path_for_license_plates = "C:/Users/User/Documents/Vip/Cropped_license_plate (230)/*"
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
df3.to_csv('Kaggle.csv', header=False, index=False)




