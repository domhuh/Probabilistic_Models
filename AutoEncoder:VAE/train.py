import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from AutoEncoders import *

def binarize(X): return 1. * (X >= 0.5)

if __name__ == "__main__":
	data = datasets.load_digits()
	x = data.images/16.0
	rx = x.reshape(-1,64)
	brx = binarize(brx)
	
	print("Training AutoEncoder")
	ae = AutoEncoder()
	ae.compile('adam', "binary_crossentropy")
	ae.fit(rx,rx, epochs = 100)

	print("Training VAE")
	vae = VAE()
	vae.compile('adam', "binary_crossentropy")
	vae.fit(rx,rx, epochs = 100)

	print("Running inference on AutoEncoder")
	before = rx[0][None,:]+np.random.uniform(-0.5,0.5,[64])[None,:]
	img = ae.predict(before).reshape(8,8)
	plt.imshow(before.reshape(8,8))
	plt.imshow(img.reshape(8,8))
	plt.imshow(rx[0].reshape(8,8))

	print("Running inference on VAE")
	before = rx[0][None,:]+np.random.uniform(-0.5,0.5,[64])[None,:]
	img = vae.predict(before).reshape(8,8)
	plt.imshow(before.reshape(8,8))
	plt.imshow(img.reshape(8,8))
	plt.imshow(rx[0].reshape(8,8))