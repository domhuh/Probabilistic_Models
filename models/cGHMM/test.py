from train import *

if __name__ == "__main__":
	tsignals, tlabels = [], []
	test_path = "../input/audio/digits/test"
	fs = [os.path.join(test_path,x)  for x in os.listdir(test_path)]
	for pp in fs:
	    for cp in os.listdir(pp):
	        tsignals.append(wav.read(os.path.join(pp, cp))[-1])
	        tlabels.append(cp[0])
	tsignals, tlabels = np.array(tsignals), np.array(tlabels)

	with open("model.pkl", "rb") as input_file:
		model = cPickle.load(input_file)

	print(model.score(tsignals, tlabels, preprocess=True))
