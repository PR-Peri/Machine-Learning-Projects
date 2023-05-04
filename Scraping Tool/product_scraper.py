#!/usr/bin/env python
# coding: utf-8

# ## SCRAPING TOOL
# The goal is to automate the manual process of scraping by automating the collection of product media (images and videos) for items available on the H&M website.
# 
# <li> The script allows all the product media to be easily obtained without having to manually access or open the website. </li>
# <li> All media are placed into different subfolders of items. </li>
# <br></br>
# 
# <b> Reason for choosing Selenium: </b>
# <br>
# <p> Selenium is considered one of the best choices for scraping data from the H&M website for several reasons. Firstly, the H&M website is primarily built using JavaScript, which can make it difficult to scrape with traditional scraping tools. However, Selenium is capable of executing JavaScript code on a website, making it ideal for scraping dynamic websites like H&M.
# 
# Secondly, Selenium provides a high level of automation, which is particularly useful when scraping large amounts of data from a website. With Selenium, you can automate the process of navigating to specific web pages, interacting with various elements, and extracting the desired data, all without manual intervention.
# 
# Lastly, Selenium offers a wide range of programming language support, making it accessible to developers with various programming backgrounds. This flexibility allows developers to choose a programming language that they are most comfortable with, making it easier to write and maintain the scraping code. </p>
# 
# 

# ## Importing Libaries
import csv
import os
import re
import urllib.request
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ## Helper Functions

def reading_data_into_dataframe(prod_dataset):
    """
    Reading dataset into dataframe
    """
    
    prod_data = pd.read_excel(prod_dataset, usecols=[0, 1, 2]) #Reads the dataset into pandas dataframe
    prod_code = prod_data['Article Number'].dropna().astype(int).astype(str).tolist() #Converts datatype 
    prod_name = prod_data['Product Name'].dropna().tolist()
    prod_color = prod_data['Color'].dropna().tolist()
    
    prod_data = pd.DataFrame(list(zip(prod_code, prod_name, prod_color )), columns =['prod_code', 'prod_name', 'prod_color'])
    
    return prod_data, prod_code

def get_len_prod_code(article_code):
    """
    Checking product code length
    """
    
    return len(str(article_code))

def adjust_prod_code(article_code_lesser):
    """
    If the product code has problem running the url for any articles with length '9'
    this module can be used to change the format to add '0' and make the length to '10'
    """
    
    if len(str(article_code_lesser)) == 9:
           return '0' + article_code_lesser 
    else:
           return article_code_lesser

def scrape_product_images(product_codes):
    """
    Scrapes all the product media (images and video) into local directory
    """
    
    # Create a directory for the output files
    os.makedirs("output", exist_ok=True)
    
    for product_code in product_codes:
        
        try:
            product_code = str(product_code)
            # Format the product URL based on the product code
            product_url = f"https://www2.hm.com/en_my/productpage.{product_code}.html"        
            # Create a directory for the product code
            os.makedirs(f"output/{product_code}", exist_ok=True)

            # Initialize the webdriver
            service = Service('/path/to/chromedriver')
            driver = webdriver.Chrome(service=service)
            driver.get(product_url)

            # Wait for the images to load
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.product-detail-main-image-container img, .pdp-secondary-image img')))

            # Get the image sources
            images = driver.find_elements(By.CSS_SELECTOR, '.product-detail-main-image-container img, .pdp-secondary-image img')
            image_sources = [re.sub('^//', 'https://', image.get_attribute('src')) for image in images]

            # Download and save the images into the directory
            for i, image_source in enumerate(image_sources):
                filename = f"output/{product_code}/image_{i+1}.jpg"
                urllib.request.urlretrieve(image_source, filename)
                
            print(f'Images for product code {product_code} downloaded successfully.')
            
        except Exception as e:
            
            print(f'Error downloading images for product code {product_code}: {str(e)}')
            
        # Quit the webdriver
        driver.quit()


# ## Main Script 

#READS THE DATA INTO DATAFRAME AND EXTRACTS THE PRODUCT CODE INTO LIST
product_data, product_code = reading_data_into_dataframe('data.xlsx')

#GENERATES A NEW COLUMN TO CHECK LENGTH OF PRODUCT CODE
product_data['prod_code_len'] = product_data['prod_code'].apply(get_len_prod_code)

#CHECKS LENGTH OF THE PRODUCT CODE
product_data.prod_code_len.value_counts()

#APPLIES '0' TO STANDARDIZE THE STRING LENGTH
product_data['prod_code_updated'] = product_data['prod_code'].apply(adjust_prod_code)


#WE USE THIS ONLY IF THE URL HAS PROBLEM
updated_product_code = product_data['prod_code_updated'].tolist()

#RUNS THE SCRAPING MODULE AND SAVES FINAL OUTPUT TO LOCAL DIRECTORY
scraper = scrape_product_images(updated_product_code)

