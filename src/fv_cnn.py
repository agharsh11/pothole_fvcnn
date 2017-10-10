import sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
import tensorflow as tf
import scipy
import cPickle
from lxml import etree
from PIL import Image
from keras.applications.vgg16 import VGG16
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
def dictionary(descriptors, N):
	em = em = cv2.ml.EM_create(N)
	em.train(descriptors)

	return np.float32(em.getMat("means")), \
		np.float32(em.getMatVector("covs")), np.float32(em.getMat("weights"))[0]

def image_descriptors(file):
	img = cv2.imread(file, 0)
	img = cv2.resize(img, (256, 256))
	_ , descriptors = cv2.SIFT().detectAndCompute(img, None)

	return descriptors

def folder_descriptors(folder):
	files = glob.glob(folder + "/*.jpg")
	print("Calculating descriptos. Number of images is", len(files))
	return np.concatenate([image_descriptors(file) for file in files])

def likelihood_moment(x, ytk, moment):	
	x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
	return x_moment * ytk

def likelihood_statistics(samples, means, covs, weights):
	gaussians, s0, s1,s2 = {}, {}, {}, {}
	samples = zip(range(0, len(samples)), samples)
	
	g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]
	for index, x in samples:
		gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

	for k in range(0, len(weights)):
		s0[k], s1[k], s2[k] = 0, 0, 0
		for index, x in samples:
			probabilities = np.multiply(gaussians[index], weights)
			probabilities = probabilities / np.sum(probabilities)
			s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
			s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
			s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)

	return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
	return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
	return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
	return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
	v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
	return v / np.sqrt(np.dot(v, v))

def fisher_vector(samples, means, covs, w):
	s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
	T = samples.shape[0]
	covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
	a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
	b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
	c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
	fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
	fv = normalize(fv)
	return fv

def generate_gmm(X, N):
	
	means, covs, weights = dictionary(X, N)
	#Throw away gaussians with weights that are too small:
	th = 1.0 / N
	means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
	covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
	weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])

	np.save("means.gmm", means)
	np.save("covs.gmm", covs)
	np.save("weights.gmm", weights)
	return means, covs, weights

def get_fisher_vectors_from_folder(X, gmm):
	return np.float32([fisher_vector(X[i], *gmm) for i in range(len(X))])

def fisher_features(X, gmm):
	features = get_fisher_vectors_from_folder(X, gmm)
	return features

def train(gmm, features):
	X = np.concatenate(features.values())
	Y = np.concatenate([np.float32([i]*len(v)) for i,v in zip(range(0, len(features)), features.values())])

	clf = svm.SVC()
	clf.fit(X, Y)
	return clf

def success_rate(classifier, features):
	print("Applying the classifier...")
	X = np.concatenate(np.array(features.values()))
	Y = np.concatenate([np.float32([i]*len(v)) for i,v in zip(range(0, len(features)), features.values())])
	res = float(sum([a==b for a,b in zip(classifier.predict(X), Y)])) / len(Y)
	return res
	
def load_gmm(folder = ""):
	files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
	return map(lambda file: load(file), map(lambda s : folder + "/" , files))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d' , "--dir", help="Directory with images" , default='.')
    parser.add_argument("-g" , "--loadgmm" , help="Load Gmm dictionary", action = 'store_true', default = False)
    parser.add_argument('-n' , "--number", help="Number of words in dictionary" , default=5, type=int)
    args = parser.parse_args()
    return args

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
with open('model.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)

gmm=generate_gmm(X,N)
	 
fisher_features = fisher_features(working_folder, gmm)
#TBD, split the features into training and validation
classifier = train(gmm, fisher_features)
rate = success_rate(classifier, fisher_features)
print("Success rate is", rate)
a = np.array(X)
print a.shape
print len(X)
print len(Y)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X,Y)
