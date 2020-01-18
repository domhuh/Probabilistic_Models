import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

x = np.random.randn(100)
w, b, s = *np.random.uniform(5,size=2), np.random.uniform(0.1)
noise = np.random.randn(x.shape[0])*s
y = b + w*x + noise

#plt.scatter(x, y)

basic_model = pm.Model()

with basic_model:
    weight = pm.Normal('weight', mu=2.5, sd=5)
    bias = pm.Normal('bias', mu=2.5, sd=5)
    sigma = pm.HalfNormal('sigma', sd=1)
    mu = bias + weight*x
    
    Y = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)


mapParams = pm.find_MAP(model=basic_model)

with basic_model:
    trace = pm.sample(draws=500, tune=100, chains=3)


_ = pm.traceplot(trace)

pm.summary(trace)


plt.scatter(x, mapParams["weight"]*x + mapParams["bias"]+mapParams["sigma"]*noise)
plt.scatter(x, y)
plt.show()