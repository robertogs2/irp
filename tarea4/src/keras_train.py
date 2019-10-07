#Example from https://victorzhou.com/blog/keras-neural-network-tutorial/

import numpy as np 
import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import Adam

#Load train and test data

print("Loading dataset...")
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

#Normalize images from [-0.5, 0.5]
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

#Flatten images to a single vector of 784
train_images = train_images.reshape((-1, 784))
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


#Model compiling
model.compile(
  #gradient optimizer
  optimizer='adam',
  #loss function for softmax output layer
  loss='categorical_crossentropy',
  #metrics for classification problem
  metrics=['accuracy'])


#Model training
model.fit(
  #X data
  train_images,
  #Y data encoded as canonic vector
  to_categorical(train_labels),
  #Iterations for the whole data set
  epochs=150,
  #Samples per gradient update
  batch_size=64,
  #Validation fraction from training data
  validation_split=0.3)

#Model testing
loss, accuracy = model.evaluate(
                  test_images,
                  to_categorical(test_labels))

print("Loss: " + str(loss) + "\nAccuracy: " + str(accuracy))

#Model saving
model.save_weights('../Keras_Models/model.h5')