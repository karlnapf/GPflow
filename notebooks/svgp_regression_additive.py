from matplotlib import pyplot as plt

import GPflow.kernels
from GPflow.likelihoods import Gaussian
from GPflow.svgp import SVGP
from GPflow.svgp_additive import SVGP_additive
import numpy as np


# build a very simple data set:
N = 100
M = 20
D = 2
X = np.random.rand(N, D)
Z = X[np.random.permutation(N)[:M]]

s_n = 0.1
if D == 1:
    F = np.sin(12 * X) + 0.66 * np.cos(25 * X)
else:
    F = np.sin(12 * X[:,0].reshape(N,1)) + 0.66 * np.cos(25 * X[:,0].reshape(N,1))

Y = F + np.random.randn(N,1) * 0.1

# build the GPR object
k = GPflow.kernels.Matern52(D)
likelihood = Gaussian()
m = SVGP(X, Y, k, likelihood, Z)

k = [GPflow.kernels.Matern52(1) for d in range(D)]
Z = [Z[:,d].reshape(M, 1).copy() for d in range(D)]
m = SVGP_additive(X, Y, k, likelihood, Z)


m.optimize()
Yp, Vp = m.predict_f(X)
Sp = np.sqrt(Vp)


rmse = np.sqrt(np.mean((Yp - Y) ** 2))
print "RMSE: %.3f" % rmse

