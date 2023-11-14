import os
import sys
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import confusion_matrix, classification_report

class ModelTrainer:

    def create_model(self):
        """This method will create the Convolutional Model"""
        try:
            logging.info("Inside the create model method")

            model = Sequential()

            logging.info("Creating the model structure")

            model.add(Conv2D(32, (2,2), input_shape=(64,64,3), padding="same", activation="relu"))
            model.add(MaxPool2D(2,2))
            model.add(Conv2D(64, (2,2), padding="same", activation="relu"))
            model.add(MaxPool2D(2,2))
            model.add(Conv2D(128, (2,2), padding="same", activation="relu"))
            model.add(MaxPool2D(2,2))

            model.add(GlobalAveragePooling2D())

            model.add(Dense(100, activation="relu"))
            model.add(Dense(2, activation="sigmoid"))

            logging.info("model creation done")

            return model

        except Exception as e:
            raise CustomException(e, sys)

        
    
    def initiate_model_trainer(self, train_img, train_lab, test_img, test_lab):
        try:
            logging.info("inside initiate model training method")

            logging.info("Calling the model method")

            model = self.create_model()

            logging.info("compiling the model")
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

            logging.info("fitting the model")
            history = model.fit(train_img, train_lab, validation_data=(test_img,test_lab), epochs=50, batch_size=8)
            
            logging.info("predicting with the model")
            y_pred = model.predict(test_img)
            y_pred = y_pred.argmax(axis=1)

            logging.info("evaluating the model")

            print("CONFUSION MATRIX")
            print(confusion_matrix(y_pred, test_lab))
            print("CLASSIFICATION REPORT")
            print(classification_report(y_pred, test_lab))

            logging.info("saving the model")
            model.save_weights('model/model_weights.h5', overwrite=True)


        except Exception as e:
            raise CustomException(e, sys)