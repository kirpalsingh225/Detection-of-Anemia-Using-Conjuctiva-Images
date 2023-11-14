import sys
import os
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging

class DataTransformation:

    def image_data_transformation(self, image_arr):
        """ This method apply image preprocessing steps to image """

        try:
            logging.info("Inside image transformation method")

            images = np.array(image_arr)
            images = images/255.
            return images
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def labels_data_transformation(self, labels_arr):
        """ This method apply encoding to the labels """

        try:
            logging.info("Inside labels transformation method")

            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels_arr)

            logging.info("saving the encoding")
            pickle.dump(label_encoder, open('assets/encoder.pkl','wb'))
            
        except Exception as e:
            raise CustomException(e, sys)
        return encoded_labels
        
    def initiate_data_transformation(self, images, labels):
        """This method initiates the data transformation process"""
        try:
            logging.info("Inside initiate data transformation method")

            preprocessed_images = self.image_data_transformation(images)
            preprocessed_labels = self.labels_data_transformation(labels)

            return (preprocessed_images, preprocessed_labels)
              
        except Exception as e:
            raise CustomException(e, sys)
        

