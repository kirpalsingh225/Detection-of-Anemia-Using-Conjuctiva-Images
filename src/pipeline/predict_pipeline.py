import cv2 as cv
import numpy as np
from src.logger import logging
from keras.models import Sequential
from keras.models import load_model
from src.components.data_transformation import DataTransformation

class PredictPipeline:

    def predict(self, img_path):
        """This method will predict whether the person has anemia or not"""

        logging.info("reading the image")
        image = cv.imread(img_path)

        logging.info("loading the model")
        model = load_model('model\model_weights.h5')

        logging.info("preprocessing the image")
        preprocessed_image = DataTransformation().image_data_transformation(image)

        logging.info("prediction")
        model = Sequential()
        model.load_weights('model\model_weights.h5')

        prediction = model.predict(image.reshape(1, image.shape[0], image.shape[1], 3))

        print(prediction)


if __name__=="__main__":
    pp = PredictPipeline()
    pp.predict('assets\test_img.jpg')

