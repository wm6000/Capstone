# Dog Breed Classification with Convolutional Neural Networks

This project aims to develop a Convolutional Neural Network (CNN) pipeline to 
classify images of dogs into their respective breeds. Additionally, it suggests 
resembling dog breeds for images of humans.

## Code
Open dog_app.ipynb to view project.

## Instructions
1. Clone this repository
2. Install requirements from requirement.txt
3. Run Flask app - python run.py
4. Open web browser and use link: http://127.0.0.1:3001

## Libraries Used

The project utilizes the following libraries:

- TensorFlow and Keras for building and training the CNN model.
- NumPy for numerical computations and array manipulation.
- Matplotlib for data visualization.
- OpenCV for image preprocessing.
- Flask for creating a web application to interact with the trained model

## Motivation

The motivation behind this project is to explore the capabilities of CNNs 
in image classification tasks. By building a CNN pipeline, we aim to learn 
and apply techniques for preprocessing images, designing CNN architectures, 
and training models using labeled data. The project also offers insights 
into transfer learning by utilizing pre-trained models for improved performance.

## Repository Files

The repository contains the following files:

- dog_app.ipynb: Jupyter Notebook containing the project code, including data loading, model building, training, and evaluation.
- report.pdf: Report of project and showcase of results.
- extract_bottleneck_features.py: Python script used to extract bottleneck features from pre-trained CNN models.
- classifier.py: Python script that takes functions from the dog_app.ipynb file to classify photos for website.
- run.py: Runs flask app
- requirements.txt: Text file specifying the required Python libraries and their versions.
- haarcascades/haarcascade_frontalface_alt.xml: Haar cascade file used for face detection in human images.
- images/: Directory containing sample images for testing the trained model.
- saved_models: Saved models for classification.
- templates/index.html: Contains Website html to classify photos.

## Summary of Results
After implementing the CNN pipeline and training the model on a dataset of 
dog images, we achieved significant improvements in classification accuracy. 
Starting from a baseline accuracy of 4%, we progressively enhanced the model's 
performance to 40% and ultimately reached an accuracy of 81% using transfer 
learning with the ResNet-50 model.

The trained model demonstrated the ability to classify various dog breeds 
accurately and provided meaningful suggestions for resembling dog breeds 
when presented with images of humans. Despite certain challenges, such as 
class imbalance in the dataset and architectural adjustments, the project 
yielded satisfactory results.

## Acknowledgements

I acknowledge the Udacity course for providing that dataset of labeled dog images
as well as the human images and the very well structured project prompt.




