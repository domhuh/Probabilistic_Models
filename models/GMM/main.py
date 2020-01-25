import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from GaussianMixture import *
from collections import defaultdict

if __name__=="__main__":
	u1, std1, u2, std2 = *np.random.uniform(size = 2), *np.random.uniform(size = 2)
	x1 = np.random.normal(u1, std1, 500)
	x2 = np.random.normal(u2, std2, 500)

	x = np.append(x1,x2)

	plt.scatter(x, stats.norm(np.mean(x), np.std(x)).pdf(x))
	sns.distplot(x, bins=100, kde=False, norm_hist=True)


	model = GaussianMixture(2)
	model.fit(x[:,None],5)
	clusters = defaultdict(list)
	for e,p in zip(x,model.predict(x)):
	    clusters[p].append(e)
	    
	for c,v in clusters.items():
	    color = tuple(np.random.choice(range(256), size=3)/255)
	    plt.scatter(v,stats.norm(np.mean(v), np.std(v)).pdf(v), color = color)

	sns.distplot(x, bins=100, kde=False, norm_hist=True)
