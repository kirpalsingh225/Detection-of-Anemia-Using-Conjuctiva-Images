
## Anemia Detection using Conjuctiva Images

Aim is to develop a robust and efficient model for the early detection of anemia using Convolutional Neural Networks (CNN) applied to conjunctiva images. Anemia, a condition characterized by a deficiency of red blood cells or hemoglobin, is a prevalent health concern globally. Early detection is crucial for timely intervention and effective management.
## Dataset

The dataset consists of 4262 conjuctiva images. The conjunctiva is a thin, clear membrane that protects your eye. The aim is to classify the images into anemic or non-anemic. The dataset is almost balanced and ready for the model building.

![img](https://github.com/kirpalsingh225/Detection-of-Anemia-Using-Conjuctiva-Images/blob/main/assets/dataset.png)


    

## Components

- **Data Ingestion**
  
  Data Ingestion Component of an Anemia Detection project, handles the data ingestion process for conjunctiva images using Convolutional Neural Networks (CNN). The DataIngestion class reads images from the 'artifacts' directory, resizes them to a uniform 64x64 pixels, shuffles the data, and returns the preprocessed images along with their corresponding labels. The main script instantiates this class, performs data ingestion, and subsequently passes the data to the DataTransformation component for further preprocessing.

- **Data Transformation**

    Data Transformation Component forms part of an Anemia Detection project, focuses on data transformation for input images and corresponding labels. The DataTransformation class offers two key methods. The image_data_transformation method preprocesses the input image array, scaling pixel values to a range between 0 and 1. This normalization enhances the model's convergence during training. The labels_data_transformation method encodes the input labels using LabelEncoder from sklearn. The encoded labels are then saved using pickle, and the encoder itself is persisted for potential later use. The class also provides an initiate_data_transformation method, serving as an entry point for the overall data transformation process. This method orchestrates the image and label transformations, returning a tuple containing the preprocessed images and encoded labels. Throughout the code, detailed logging is implemented to track the progress of the data transformation steps, and custom exceptions are raised to facilitate error handling. The script complements the data ingestion component, ensuring that the input data is appropriately preprocessed for subsequent use in the Anemia Detection pipeline.


- **Model Trainer**

    This Python script contributes to the Anemia Detection project and is responsible for training a Convolutional Neural Network (CNN) model using the Keras library. The `ModelTrainer` class contains methods to create, compile, and train the CNN model, as well as to evaluate its performance.

    The `create_model` method initializes a sequential model structure with convolutional layers, max-pooling layers, and densely connected layers. This architecture is designed to extract features from the conjunctiva images provided as input.

    The `initiate_model_trainer` method orchestrates the entire training process. It calls the `create_model` method to obtain the CNN model, compiles it using the Adam optimizer and sparse categorical crossentropy loss, and then fits the model to the training data. The training process is monitored and validated using the test data over 50 epochs with a batch size of 8.

    After training, the model's performance is evaluated by predicting on the test set, and metrics such as confusion matrix and classification report are printed. Additionally, the trained model weights are saved for potential future use.

    Throughout the code, detailed logging is implemented to track the progress and identify any issues during the model training process. Custom exceptions are raised to facilitate error handling. The script completes the data processing pipeline, providing a trained model ready for anemia detection based on conjunctiva images.   


## Architecture 

![img](https://github.com/kirpalsingh225/Detection-of-Anemia-Using-Conjuctiva-Images/blob/main/assets/Architecture.png)

## Run Locally

Clone the project

```bash
  git clone https://github.com/kirpalsingh225/Detection-of-Anemia-Using-Conjuctiva-Images
```

Go to the project directory

```bash
  cd Detection-of-Anemia-Using-Conjuctiva-Images
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the model

```bash
  python src/components/data_ingestion.py
```

