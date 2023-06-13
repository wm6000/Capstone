import matplotlib.pyplot as plt 
import numpy as np    
import tensorflow as tf   
from keras.layers import GlobalMaxPooling2D, Dense
from keras.models import Sequential  
import cv2

# Load and preprocess the image using ResNet50 pre-processing function
load_img = tf.keras.preprocessing.image.load_img
img_to_array = tf.keras.preprocessing.image.img_to_array
ResNet_50 = tf.keras.applications.resnet50.ResNet50
preprocess_input = tf.keras.applications.resnet50.preprocess_input

def extract_Resnet50(tensor):
	# Load pre-trained ResNet50 model
    # Set 'include_top=False' to exclude the final fully connected layers
	return ResNet_50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

# Load pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# Load list of dog breeds
dog_names = []
with open('saved_models/dogs.txt', 'r') as file:
    for line in file:
        dog_names.append(line.split(".")[-1].strip().lower())

def load_breed_model():
    model = Sequential()
    model.add(GlobalMaxPooling2D(input_shape=(7, 7, 2048)))  # Use GlobalMaxPooling2D layer
    model.add(Dense(133, activation='softmax'))
    model.load_weights('saved_models/Resnet50_model.hdf5')
    return model


# Load the pre-trained breed classification model
breed_model = load_breed_model()



def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = breed_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]#.split(".")[1]


# Define ResNet50 model for general image classification
ResNet50_model = ResNet_50(weights='imagenet')

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def read_image(img):
    if face_detector(img) == True:
        print('This is a human, but if they were a dog they would be a' + Resnet50_predict_breed(img))
        return('This is a human, but if they were a dog they would be a' + Resnet50_predict_breed(img))
    elif dog_detector(img) == True:
        print('This dog is an ' + Resnet50_predict_breed(img))
        return('This dog is an ' + Resnet50_predict_breed(img))
    else:
        print('Human nor Dog not detected.')
        return(R'Human nor Dog not detected.')

