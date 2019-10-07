from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
from PIL import Image
import pyscreenshot as ImageGrab

# Libraries for SVM
from mnist_helpers import * 				# import custom module
import joblib
from sklearn import datasets, svm
import numpy

# SVM constants
classifier_path = ""
classifier = None
# Drawing constants
image_base=28
canvas_size = 600
pixel_scaled = canvas_size/image_base

def get_content_image():
	x=root.winfo_rootx()+drawing_canvas.winfo_x()
	y=root.winfo_rooty()+drawing_canvas.winfo_y()
	x1=x+drawing_canvas.winfo_width()
	y1=y+drawing_canvas.winfo_height()
	im=ImageGrab.grab((x,y,x1,y1))
	im=im.resize((image_base,image_base),Image.ANTIALIAS)
	im=im.convert('L')
	im=numpy.array(list(im.getdata()))
	im=1-im/255.0
	return list(im)

def paint_rectangle(event):
	x1,y1=(event.x-pixel_scaled/2),(event.y-pixel_scaled/2)
	x2,y2=(event.x+pixel_scaled/2),(event.y+pixel_scaled/2)
	drawing_canvas.create_rectangle(x1,y1,x2,y2,fill="black")

def clean_rectangles():
	drawing_canvas.delete(ALL)

def load_model():
	global classifier_path_var
	global classifier_path
	global classifier
	classifier_path = filedialog.askopenfilename(initialdir = "./models/",title = "Select classifier model",filetypes = (("model files","*.sav"),("all files","*.*")))
	classifier_path_var.set(classifier_path[-50:])

	classifier = joblib.load(classifier_path)
	print(classifier)
	print("Loaded classifier" + classifier_path)

def classify():
	global prediction_var
	if classifier is not None:
		im = get_content_image()
		y = classifier.predict([im])
		prediction_var.set('Prediction: ' + str(y[0]))
		print('Prediction: ' + str(y[0]))
	else:
		messagebox.showerror("Model error", "Model not loaded, load one")


root = Tk()
root.resizable(False, False)
classifier_path_var = StringVar()
classifier_path_var.set('Not model loaded')
prediction_var = StringVar()
prediction_var.set('No prediction yet')
root.title( "Number prediction" )
drawing_canvas = Canvas(root, bg = "white", width=canvas_size, height=canvas_size)
drawing_canvas.pack(side=TOP, expand = NO, fill = BOTH)
root.bind("<Any-KeyPress>", lambda event: get_content_image())#print(event.keysym))
drawing_canvas.bind("<B1-Motion>", paint_rectangle)


bottom_canvas=Canvas(root,width=canvas_size,height=200)
bottom_canvas.pack(side=BOTTOM)

load_button=Button(bottom_canvas,text="Load model",command=load_model)
load_button.pack(side=LEFT)

load_label=Label(bottom_canvas,textvariable=classifier_path_var)
load_label.pack(side=LEFT)

clear_button=Button(bottom_canvas,text="Clear",command=clean_rectangles)
clear_button.pack(side=LEFT)

classify_button=Button(bottom_canvas,text="Classify",command=classify)
classify_button.pack(side=LEFT)

prediction_label=Label(bottom_canvas,textvariable=prediction_var)
prediction_label.pack(side=LEFT)

mainloop()