# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt

from sklearn import datasets, svm, metrics 	# Import datasets, classifiers and performance metrics
from sklearn.datasets import fetch_mldata 	# Fetch original mnist database
#from sklearn.externals import joblib 		# To store model
import joblib
from sklearn.utils import shuffle			# To shuffle vectors
from mnist_helpers import * 				# import custom module
import mnist

import sys
import argparse

def load_images_targets():
	train_images = mnist.train_images()
	train_labels = mnist.train_labels()
	test_images = mnist.test_images()
	test_labels = mnist.test_labels()

	train_images = train_images.reshape((-1, 784))
	test_images = test_images.reshape((-1, 784))

	images = np.concatenate((train_images, test_images),axis=0)
	targets = np.concatenate((train_labels,test_labels),axis=0)

	return images,targets


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

def train_new_svm(param_C=5, param_max_iterations=100, param_kernel='rbf', param_gamma=0.05, param_degree=3, param_coef0 = 0.0, param_verbose=False):
	# Create a classifier: a support vector classifier

	classifier = svm.SVC(C=param_C,
						gamma=param_gamma,
						verbose=param_verbose, 
						max_iter=param_max_iterations, 
						kernel=param_kernel, 
						coef0=param_coef0, 
						degree=param_degree)

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
	print('Model saved to: ' + path + name + '.sav')

def load_model(filepath):
	return joblib.load(filepath)

def test_model(classifier, X_test, y_test):
	expected = y_test
	predicted = classifier.predict(X_test)

	show_some_digits(X_test,predicted,title_text="Predicted {}")

	print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))
	      
	cm = metrics.confusion_matrix(expected, predicted)
	print("Confusion matrix:\n%s" % cm)

	plot_confusion_matrix(cm)

	print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))

def get_model_name(param_C=0, param_max_iterations=0, param_kernel=0, param_gamma=0, param_degree=0, param_coef0=0):
	return 'model_iter:'\
			+str(param_max_iterations)\
			+'_gamma:'+str(param_gamma)\
			+'_C:'+str(param_C)\
			+'_kernel:'+str(param_kernel)\
			+'_degree:'+str(param_degree)\
			+'_bcoef:'+str(param_coef0)\
			+'_'+str(time.time()) \

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	#parser.add_argument("-h", "--help", action='store_true', help="print help")
	parser.add_argument("-v", "--verbose", action='store_true', help="set verbose")
	parser.add_argument("-k", "--kernel", help="set kernel")
	parser.add_argument("-c", "--cparam", type=float, help='Set soft margin parameter C, '
													'the higher the C, the more you penalize making errors')
	parser.add_argument("-i", "--max_iterations", type=int, help="set maximun iterations")
	parser.add_argument("-g", "--gamma", type=float, help="set gamma")
	parser.add_argument("-d", "--degree", type=float, help="set degree")
	parser.add_argument("-b", "--bcoef", type=float, help="set coefficient b")
	parser.add_argument("-m", "--metrics", action='store_true', help="show metrics")
	parser.add_argument("-t", "--test", help="tests a model")
	args = parser.parse_args()

	# Loads the data first
	images, targets = load_images_targets()
	X_train, X_test, y_train, y_test = split_data(images, targets)
	if not args.test:
		verbose=False
		metrics_active=False
		kernel = 'rbf'
		max_iterations = 2
		C = 5
		gamma = 'scale'
		degree=3
		bcoef = 0.0
		

		kernels_avaliable = ['rbf', 'linear', 'sigmoid', 'poly']
		## Verbose process
		if args.verbose:
			verbose = True
		if args.metrics:
			metrics_active = True
		## Kernel Process
		if args.kernel:
			kernel = args.kernel
			if kernel not in kernels_avaliable:
				print('Kernel' + str(kernel) + 'not in kernels list, using default')
				print('Avaliable kernels are:')
				print(kernels_avaliable)
				kernel = 'rbf'
		else:
			print('Using default kernel ' + str(kernel))
		if args.max_iterations:
			max_iterations = args.max_iterations
		else:
			print('Using default max iterations ' + str(max_iterations))
		if args.cparam:
			C = args.cparam
		else:
			print('Using default soft margin parameter C ' + str(C))
		if args.gamma:
			gamma = args.gamma
		else:
			print('Using default gamma ' + str(gamma))
		if args.degree:
			degree = args.degree
		else:
			print('Using default degree ' + str(degree))
		if args.bcoef:
			bcoef = args.bcoef
		else:
			print('Using default bcoef ' + str(bcoef))

		#pick  random indexes from 0 to size of our dataset
		if metrics_active:
			show_some_digits(images,targets)

		# ############### Classifier with good params ###########
		myclassifier = train_new_svm(param_C=C, param_max_iterations=max_iterations, 
									param_kernel=kernel, param_gamma=gamma, 
									param_degree=degree, param_coef0 = bcoef, 
									param_verbose=verbose)

		save_model(myclassifier, '../SVM_Models/', get_model_name(param_C=C, param_max_iterations=max_iterations, 
															param_kernel=kernel, param_gamma=gamma, 
															param_degree=degree, param_coef0 = bcoef))
	else:
		metrics_active = True
		myclassifier = load_model(args.test)
	# ########################################################
	# # Now predict the value of the test

	if metrics_active:
		test_model(myclassifier, X_test, y_test)