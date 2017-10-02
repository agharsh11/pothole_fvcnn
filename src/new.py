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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
#img_path = '/home/sukhad/torch-feature-extract/images/1_4.jpg'
model = VGG16(weights='imagenet', include_top=False)
directory = '/home/sukhad/torch-feature-extract/images/*.jpg'
filenames = glob.glob(directory)
X = []
Y = []
count = 0
for file in filenames:
	img = image.load_img(file)
	w,h = img.size
	img1 = img.crop((w/4,h/4,w/2,h/2))
	img1 = img1.resize((224,224),Image.ANTIALIAS)
	x1 = image.img_to_array(img1)
	x1 = np.expand_dims(x1, axis=0)
	x1 = preprocess_input(x1)
	features = model.predict(x1)
	features = features.reshape(1,512*7*7)
	features = features.tolist()
	X.append(features[0])
	count = count + 1
	Y.append(-1)
doc = etree.parse("/home/sukhad/potholes/positive.xml")
memoryElem = doc.find('images')
for i in memoryElem:
	if(len(i.getchildren())>0):
		img_path = i.get('file')
		#print img_path
		img = image.load_img(img_path)
		for j in i.iter('box'):
			y = int(j.get('top'))
	   		x = int(j.get('left'))
			width = int(j.get('width'))
			height = int(j.get('height'))
			#print x
			#print y
			#print width
			#print height
			img1 = img.crop((x,y,width,height))
			img1 = img1.resize((224,224),Image.ANTIALIAS)
			x1 = image.img_to_array(img1)
			x1 = np.expand_dims(x1, axis=0)
			x1 = preprocess_input(x1)
			features = model.predict(x1)
			features = features.reshape(1,512*7*7)
			features = features.tolist()
			#for k in range(35):
			X.append(features[0])
			count = count + 1
			Y.append(1)
a = np.array(X)
print a.shape
print len(X)
print len(Y)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X,Y)
with open('model.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)