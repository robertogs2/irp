import numpy as np
import mnist

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
from mnist_helpers import *

#Load test data
print("Loading dataset...")
test_images = mnist.test_images()
test_labels = mnist.test_labels()

#Normalize images from [-0.5, 0.5]
test_images = (test_images / 255) - 0.5

#Flatten images to a single vector of 784
test_images = test_images.reshape((-1, 784))

#Model building
model = Sequential([
  #Input shape for network
  Dense(64, activation='sigmoid', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(64, activation='sigmoid'),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax')
  ])

#Model loading
model.load_weights('../Keras_Models/model.h5')

#Predict
predictions = model.predict(test_images)
#Vector with winner class for each image
max_predictions = np.argmax(predictions, axis=1)

matrix = confusion_matrix(test_labels, max_predictions)
report = classification_report(test_labels, max_predictions)

print("CONFUSION MATRIX\n" + str(matrix))
print("\nCLASSIFICATION REPORT\n" + str(report))

plot_confusion_matrix(matrix)