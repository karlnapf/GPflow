
from matplotlib import pyplot as plt
import GPflow.kernels
from GPflow.likelihoods import Gaussian
from GPflow.svgp import SVGP
from GPflow.svgp_additive import SVGP_additive
import numpy as np

np.random.seed(0)

# build a very simple data set:
N = 1000
M = 20
D = 2
X = np.random.rand(N, D)
Z = X[np.random.permutation(N)[:M]]

f = lambda x : np.sin(12 * x) + 0.66 * np.cos(25 * x)

s_n = 0.1
F = np.zeros((N,1))
Fs = np.zeros((N,D))
if D == 1:
    F = f(x)
else:
    for d in range(D):
        Fs[:,d] = f(X[:,d])
    F = np.sum(Fs,axis=1).reshape(N,1)
Y = F + np.random.randn(N,1) * 0.1

likelihood = Gaussian()

k = GPflow.kernels.Matern52(1)
Z = [Z[:,d].reshape(M, 1).copy() for d in range(D)]
m = SVGP_additive(X, Y, k, likelihood, Z)
m.likelihood.variance = 0.01

m.optimize()

Yp, Vp = m.predict_f(X)
Sp = np.sqrt(Vp)

rmse = np.sqrt(np.mean((Yp - Y) ** 2))
print "RMSE: %.3f" % rmse

