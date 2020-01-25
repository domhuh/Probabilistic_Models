from hmmlearn import hmm
from collections import defaultdict
from tqdm import tqdm

class cGaussianHMM():
    def __init__(self, n_components, covariance_type="diag", noisy = False):
        self.hmms = defaultdict(list)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.noisy = noisy
    def fit(self,X, Y, num_models = 1):
        for _ in range(num_models):
            for x, y in tqdm(zip(X,Y), total = len(X)):
                m = hmm.GaussianHMM(n_components=self.n_components,
                                covariance_type=self.covariance_type)
                if self.noisy: x += np.random.uniform(size = x.shape)
                m.fit(x)
                self.hmms[y].append(m)
    def predict(self,X):
        maxScore, maxLabel = float('-inf'),0
        for label, models in self.hmms.items():
            for m in models:
                s = m.score(X)
                if s > maxScore:
                    maxScore, maxLabel = s, label
        return maxLabel
    def score(self, X, Y, preprocess = False):
        correct, total = 0,0
        for x, y in tqdm(zip(X,Y), total=len(X)):
            if preprocess: x = mfcc(x)
            if y == self.predict(x):
                correct+=1.0
            total+=1.0
        return correct/total