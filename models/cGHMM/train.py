import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import logfbank, mfcc
from cGaussianHMM import *
import cPickle

if __name__ == "__main__":
	signals, labels = [], []
	train_path = "../input/audio/digits/train"
	fs = [os.path.join(train_path,x)  for x in os.listdir(train_path)]
	for pp in fs:
	    for cp in os.listdir(pp):
	        signals.append(wav.read(os.path.join(pp, cp))[-1])
	        labels.append(cp[0])
	signals, labels = np.array(signals), np.array(labels)
	maxlen = max([s.shape[0] for s in signals])

	features = []
	for s in signals:
	    features.append(mfcc(s))#np.pad(s, (0,maxlen-s.shape[0]),'constant')))
	    #[tmp[i*200:i*200+200] for i in range(tmp.shape[0]) if i < tmp.shape[0]//200])
	features = np.array(features)

	model = cGaussianHMM(10,noisy=True)
	model.fit(features, labels)
	print(model.score(features, labels))

	with open("model.pkl", "wb") as out:
		cPickle.dump(model, out)



