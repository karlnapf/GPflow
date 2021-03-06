import tensorflow as tf
from .model import GPModel
from .param import Param
from .densities import multivariate_normal
from .mean_functions import Zero
import likelihoods
from tf_hacks import eye

class GPR(GPModel):
    def __init__(self, X, Y, kern, mean_function=Zero()):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x multivariate_norma is an appropriate GPflow object

        kern, mean_function are appropriate GPflow objects

        This is a vanilla implementation of a GP regression with a Gaussian
        likelihood. 
        """
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

    def build_likelihood(self):
        """
        Constuct a tensorflow function to compute the likelihood of a general GP model.

            \log p(Y, V | theta).

        """
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)

        return multivariate_normal(self.Y, m, L)

    def build_predict(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kd = self.kern.Kdiag(Xnew)
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.user_ops.triangular_solve(L, Kx, 'lower')
        V = tf.user_ops.triangular_solve(L, self.Y - self.mean_function(self.X), 'lower')
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)
        fvar = Kd - tf.reduce_sum(tf.square(A), reduction_indices=0)
        return fmean, tf.tile(tf.reshape(fvar, (-1,1)), [1, self.Y.shape[1]])

 

