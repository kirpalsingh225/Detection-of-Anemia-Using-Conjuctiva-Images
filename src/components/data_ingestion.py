import os
import sys
import numpy as np
from src.exception import CustomException
import cv2 as cv
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class DataIngestion:
    
    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion component")
        try:
            data = []
            data_dir = 'artifacts'
            dirs = os.listdir(data_dir)
            
            for i in dirs:
                path = os.path.join(data_dir,i)
                for img in os.listdir(path):
                    image = cv.imread(os.path.join(path,img))
                    image = cv.resize(image, (64,64))
                    data.append([image, i])

            logging.info("ingestion of data is completed")
            logging.info("shuffling the data")
            np.random.shuffle(data)

            images = []
            labels = []

            for img, lab in data:
                images.append(img)
                labels.append(lab)

            return(images, labels)
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    images, labels = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    en_images, en_labels = data_transformation.initiate_data_transformation(images, labels)

    x_train, x_test, y_train, y_test = train_test_split(en_images, en_labels, test_size=0.1, random_state=22)

    modeltrainer = ModelTrainer()
    modeltrainer.initiate_model_trainer(x_train, y_train, x_test, y_test)