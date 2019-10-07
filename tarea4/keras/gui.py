from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
from PIL import Image
import pyscreenshot as ImageGrab

# Libraries for Keras
from mnist_helpers import *         # import custom module
from keras.models import Sequential
from keras.layers import Dense
import numpy

class App:
  def __init__(self):

    # keras constants
    #Model building
    self.model = Sequential([
    #Input shape for network
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
    ])

    self.classifier_path = ""
    # Drawing constants
    self.image_base=28
    self.canvas_size = 600
    self.pixel_scaled = self.canvas_size/self.image_base

    self.root = Tk()
    self.root.resizable(False, False)
    self.classifier_path_var = StringVar()
    self.classifier_path_var.set('Not model loaded')
    self.prediction_var = StringVar()
    self.prediction_var.set('No prediction yet')
    self.root.title( "Number prediction" )
    self.drawing_canvas = Canvas(self.root, bg = "white", width=self.canvas_size, height=self.canvas_size)
    self.drawing_canvas.pack(side=TOP, expand = NO, fill = BOTH)
    self.root.bind("<Any-KeyPress>", lambda event: self.get_content_image())#print(event.keysym))
    self.drawing_canvas.bind("<B1-Motion>", self.paint_rectangle)


    self.bottom_canvas=Canvas(self.root,width=self.canvas_size,height=200)
    self.bottom_canvas.pack(side=BOTTOM)

    self.load_button=Button(self.bottom_canvas,text="Load model",command=self.load_model)
    self.load_button.pack(side=LEFT)

    self.load_label=Label(self.bottom_canvas,textvariable=self.classifier_path_var)
    self.load_label.pack(side=LEFT)

    self.clear_button=Button(self.bottom_canvas,text="Clear",command=self.clean_rectangles)
    self.clear_button.pack(side=LEFT)

    self.classify_button=Button(self.bottom_canvas,text="Classify",command=self.classify)
    self.classify_button.pack(side=LEFT)

    self.prediction_label=Label(self.bottom_canvas,textvariable=self.prediction_var)
    self.prediction_label.pack(side=LEFT)

    mainloop()

  def get_content_image(self):
    x=self.root.winfo_rootx()+self.drawing_canvas.winfo_x()
    y=self.root.winfo_rooty()+self.drawing_canvas.winfo_y()
    x1=x+self.drawing_canvas.winfo_width()
    y1=y+self.drawing_canvas.winfo_height()
    im=ImageGrab.grab((x,y,x1,y1))
    im=im.resize((self.image_base,self.image_base),Image.ANTIALIAS)
    im=im.convert('L')
    im=numpy.array(list(im.getdata()))
    im=-(im/255 - 0.5)
    return im

  def paint_rectangle(self,event):
    x1,y1=(event.x-self.pixel_scaled/2),(event.y-self.pixel_scaled/2)
    x2,y2=(event.x+self.pixel_scaled/2),(event.y+self.pixel_scaled/2)
    self.drawing_canvas.create_rectangle(x1,y1,x2,y2,fill="black")

  def clean_rectangles(self):
    self.drawing_canvas.delete(ALL)

  def load_model(self):
    self.classifier_path = filedialog.askopenfilename(initialdir = "./Models/",title = "Select classifier model",filetypes = (("model files","*.h5"),("all files","*.*")))

    if self.classifier_path != ():
      self.classifier_path_var.set(self.classifier_path.split("/")[-1])
      #Model loading
      self.model.load_weights(self.classifier_path)
      print("Loaded classifier" + self.classifier_path)

  def classify(self):
    if self.classifier_path != "":
      im = self.get_content_image()
      im = im.reshape((-1, 784))
      y = numpy.argmax(self.model.predict(im), axis=1)
      self.prediction_var.set('Prediction: ' + str(y[0]))
      print('Prediction: ' + str(y[0]))
    else:
      messagebox.showerror("Model error", "Model not loaded, load one")


app = App()