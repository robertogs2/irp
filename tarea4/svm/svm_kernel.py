# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt

from sklearn import datasets, svm, metrics 	# Import datasets, classifiers and performance metrics
from sklearn.datasets import fetch_mldata 	# Fetch original mnist database
from sklearn.externals import joblib 		# To store model
from sklearn.utils import shuffle			# To shuffle vectors
from mnist_helpers import * 				# import custom module

def load_images_targets():
	mnist = fetch_mldata('MNIST original', data_home='./') 	# it creates mldata folder in your root project folder
	mnist.keys() 											# Check for correct keys data, COL_NAMES, DESCR, target fields
	images = mnist.data 									# Data field is 70k x 784 array, each row represents pixels from 28x28=784 image
	targets = mnist.target
	return images, targets

def split_data(images, targets, param_test_size=10000/70000, param_random_state=42):
	X_data = images/255.0 									#Normalize the data
	Y = targets
	#Split data to train and test
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=param_test_size, random_state=param_random_state, shuffle=False)
	#Now shuffle that data
	X_train, y_train = shuffle(X_train, y_train, random_state=0)
	X_test, y_test = shuffle(X_test, y_test, random_state=0)
	return X_train, X_test, y_train, y_test

def train_new_svm(param_C=5, param_gamma=0.05, param_max_iterations=100, param_kernel='rbf', param_verbose=False):
	# Create a classifier: a support vector classifier

	classifier = svm.SVC(C=param_C,gamma=param_gamma,verbose=param_verbose, max_iter=param_max_iterations, kernel=param_kernel)

	# Training
	start_time = dt.datetime.now()
	print('Start learning at {}'.format(str(start_time)))

	classifier.fit(X_train, y_train)

	end_time = dt.datetime.now() 
	print('Stop learning {}'.format(str(end_time)))
	elapsed_time= end_time - start_time
	print('Elapsed learning {}'.format(str(elapsed_time)))
	return classifier

def save_model(model, path, name):
	# Stores the model
	joblib.dump(model, path + name + '.sav')

def test_model(classifier, X_test, y_test):
	expected = y_test
	predicted = classifier.predict(X_test)

	show_some_digits(X_test,predicted,title_text="Predicted {}")

	print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))
	      
	cm = metrics.confusion_matrix(expected, predicted)
	print("Confusion matrix:\n%s" % cm)

	plot_confusion_matrix(cm)

	print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))

#---------------- classification begins -----------------

images, targets = load_images_targets()

#pick  random indexes from 0 to size of our dataset
#show_some_digits(images,targets)

X_train, X_test, y_train, y_test = split_data(images, targets)
print(len(X_train))
print(len(X_test))
################ Classifier with good params ###########
param_C = 5
param_gamma = 0.05
max_iterations = 100
myclassifier = train_new_svm(param_C, param_gamma, max_iterations, 'rbf', False)

save_model(myclassifier, './models/', 'model_iter:'+str(max_iterations)+'_'+str(time.time()))

########################################################
# Now predict the value of the test

test_model(myclassifier, X_test, y_test)


