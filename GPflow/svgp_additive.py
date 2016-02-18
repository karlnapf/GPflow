import tensorflow as tf
import numpy as np
from param import Param, ParamList
from .model import GPModel
import transforms
import conditionals
from .mean_functions import Zero
from tf_hacks import eye
import kullback_leiblers


class SVGP_additive(GPModel):
    def __init__(self, X, Y, kern, likelihood, Z, mean_function=None, num_latent=None, q_diag=False, whiten=True):
        # kern, Z, mean_function are all lists of univariate GP counterparts for each dimension
        # after has finished, num_inducing, qu_mu, q_sqrt are also lists
        
        # TODO: check same length of all lists
        # TODO: check dimensions of all Z elements
        # TODO: all elements of Z should have dimension mx1, where m is inducing points
        # TODO: allow for passing single elements which are then broadcasted to a list automatically
        
        if mean_function is None:
            mean_function = [Zero() for _ in range(len(Z))]
        
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.q_diag, self.whiten = q_diag, whiten
        self.num_latent = num_latent or Y.shape[1]
        self.num_inducing = [z.shape[0] for z in Z]



        self.Z = ParamList([Param(z) for z in Z])
        self.q_mu = ParamList([Param(np.zeros((z.shape[0], self.num_latent))) for z in Z])

        if self.q_diag:
            self.q_sqrt = ParamList([Param(np.ones((z.shape[0], self.num_latent)), transforms.positive) for z in Z])
        else:
            self.q_sqrt = ParamList([Param(np.array([np.eye(z.shape[0]) for _ in range(self.num_latent)]).swapaxes(0,2)) for z in Z])



    def build_prior_KL(self):
        KL = None
        
        for d in xrange(self.X.shape[1]):
            q_mu_d = self.q_mu[d]
            q_sqrt_d = self.q_sqrt[d]
            Z_d = self.Z[d]
            
            if self.whiten:
                if self.q_diag:
                    KL_d = kullback_leiblers.gauss_kl_white_diag(q_mu_d, q_sqrt_d, self.num_latent)
                else:
                    KL_d = kullback_leiblers.gauss_kl_white(q_mu_d, q_sqrt_d, self.num_latent)
            else:
                K = self.kern.K(Z_d) + eye(self.num_inducing[d]) * 1e-6
                if self.q_diag:
                    KL_d = kullback_leiblers.gauss_kl_diag(q_mu_d, q_sqrt_d, K, self.num_latent)
                else:
                    KL_d = kullback_leiblers.gauss_kl(q_mu_d, q_sqrt_d, K, self.num_latent)
                    
            # add things up, we were too lazy to check the type of KL_d
            if KL is None:
                KL = KL_d
            else:
                KL += KL_d
                
        return KL


    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()
    
        fmean = None
        fvar = None
        for d in xrange(self.X.shape[1]):
            x_d_as_2d = self.X[:, d].reshape(len(self.X), 1)
            q_mu_d = self.q_mu[d]
            q_sqrt_d = self.q_sqrt[d]
            Z_d = self.Z[d]
            
            # Get conditionals
            if self.whiten:
                fmean_d, fvar_d = conditionals.gaussian_gp_predict_whitened(x_d_as_2d, Z_d, self.kern, q_mu_d, q_sqrt_d, self.num_latent)
            else:
                fmean_d, fvar_d = conditionals.gaussian_gp_predict(x_d_as_2d, Z_d, self.kern, q_mu_d, q_sqrt_d, self.num_latent)
    
            # add in mean function to conditionals.
            fmean_d += self.mean_function[d](x_d_as_2d)
            
            # add things up, we were too lazy to check the type of fmean_d, fvar_d
            if fmean is None or fvar is None:
                fmean = fmean_d
                fvar = fvar_d
            else:
                fmean += fmean_d
                fvar += fvar_d
        
        # Get variational expectations.
        variational_expectations = self.likelihood.variational_expectations(fmean, fvar, self.Y)
        
        return tf.reduce_sum(variational_expectations) - KL

    def build_predict_single(self, Xnew, d):

        xnew_d_as_2d = tf.expand_dims(Xnew[:, d],1)
        q_mu_d = self.q_mu[d]
        q_sqrt_d = self.q_sqrt[d]
        Z_d = self.Z[d]

        if self.whiten:
            mu_d, var_d = conditionals.gaussian_gp_predict_whitened(xnew_d_as_2d, Z_d, self.kern, q_mu_d, q_sqrt_d, self.num_latent)
        else:
            mu_d, var_d = conditionals.gaussian_gp_predict(xnew_d_as_2d, Z_d, self.kern, q_mu_d, q_sqrt_d, self.num_latent)
        mu_d += self.mean_function[d](xnew_d_as_2d)
        return mu_d, var_d

    def build_predict(self, Xnew):
        mu = None
        var = None

        for d in xrange(self.X.shape[1]):

            mu_d, var_d = self.build_predict_single( Xnew, d)

            # add things up, we were too lazy to check the type of fmean_d, fvar_d
            if mu is None or var is None:
                mu = mu_d
                var = var_d
            else:
                mu += mu_d
                var += var_d
        
        return mu, var



