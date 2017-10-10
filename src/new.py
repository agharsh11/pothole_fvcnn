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
import pdb

from sklearn.datasets import make_classification
from sklearn.mixture import GMM


#img_path = '/home/sukhad/torch-feature-extract/images/1_4.jpg'
def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covars_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def main():
    # Short demo.
    K = 64
    N = 1000

    xx, _ = make_classification(n_samples=N)
    xx_tr, xx_te = xx[: -100], xx[-100: ]

    gmm = GMM(n_components=K, covariance_type='diag')
    gmm.fit(xx_tr)

    fv = fisher_vector(xx_te, gmm)
	pdb.set_trace()




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
			N=len(X[0])
R=N/10;
gmm = GMM(n_components=N, covariance_type='diag')
gmm.fit(X[:-N])

fv = fisher_vector(X[-N:], gmm)
pdb.set_trace()
a = np.array(X)
print a.shape
print len(X)
print len(Y)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X,Y)
with open('model.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)