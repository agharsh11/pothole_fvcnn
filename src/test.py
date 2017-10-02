import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
import tensorflow as tf
import sklearn 
import scipy
import glob
import numpy as np
import cPickle
from lxml import etree
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)
with open('/home/sukhad/Desktop/model.pkl','rb') as fid:
	loaded = cPickle.load(fid)
print "loaded"
img_path = '/home/sukhad/potholes/45.jpg'
src = cv2.imread(img_path)
img = image.load_img(img_path)
dst = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
src1 = src.copy()
blur = cv2.blur(dst,(7,7))
edges = cv2.Canny(blur,20,100)
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(edges, kernel, iterations=2)
_, contours, _= cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	check =[]
	x,y,w,h = cv2.boundingRect(cnt)
	area = cv2.contourArea(cnt)
	if(area>300 and area <100000):
		img1 = img.crop((x,y,w,h))
		img1 = img1.resize((224,224),Image.ANTIALIAS)
		x1 = image.img_to_array(img1)
		x1 = np.expand_dims(x1, axis=0)
		x1 = preprocess_input(x1)
		features = model.predict(x1)
		features = features.reshape(1,512*7*7)
		features = features.tolist()
		X= features[0]
		a = loaded.predict(X)
		print a
		if(a==1):
			src1 = cv2.rectangle(src1,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imwrite('/home/sukhad/Desktop/final2.jpg',src1)
print "done"