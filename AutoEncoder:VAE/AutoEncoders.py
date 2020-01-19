from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp

class AutoEncoder(Model):
    def __init__(self):
        super().__init__()
        self.encode = Sequential([Dense(32, activation = 'sigmoid'),
                                  Dense(16, activation = 'sigmoid'),
                                  Dense(8, activation = 'sigmoid')])
        self.decode = Sequential([Dense(16, activation = 'sigmoid'),
                                  Dense(32, activation = 'sigmoid'),
                                  Dense(64, activation = 'sigmoid')])
    def call(self, X):
        fm = self.encode(X)
        return self.decode(fm)

def kl_reg(wm):
    u,std = tf.math.reduce_mean(wm), tf.math.reduce_std(wm)
    return 1e-3 * tfp.distributions.kl_divergence(tfp.distributions.Normal(u,std),
                     tfp.distributions.Normal(0,1))

class VAE(AutoEncoder):
    def __init__(self):
        super().__init__()
        self.encode = Sequential([Dense(32, activation = 'sigmoid'),
                                  Dense(16, activation = 'sigmoid'),
                                  Dense(8, activation = 'sigmoid',
                                        kernel_regularizer = kl_reg)])
