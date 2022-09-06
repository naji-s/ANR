from sklearn.base import BaseEstimator
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, brier_score_loss
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
import sys
sys.path.append('/home/scratch/nshajari/')
from mcmc_lib import run_chain
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
class ProbLinPsychM(BaseEstimator):
    def __init__(self, name='ProbLinPsychM', 
                 encoder=None,
                # psych_gamma_lambda_triple_samples=None,
                # sampler_stat=None,
                # psych_alpha_samples=None,
                # psych_beta_samples=None,
                # sig_b_samples=None,
                # sig_a_samples=None,
                # psych_lambda=None,
                # psych_z=None,
                # psych_gamma=None,                
                ):
        self.name = name
        self.encoder = encoder
        # self.psych_gamma_lambda_triple_samples = psych_gamma_lambda_triple_samples 
        # self.sampler_stat = sampler_stat 
        # self.psych_alpha_samples = psych_alpha_samples 
        # self.psych_beta_samples = psych_beta_samples 
        # self.sig_b_samples = sig_b_samples 
        # self.sig_a_samples = sig_a_samples 
        # self.psych_lambda = psych_lambda 
        # self.psych_z = psych_z 
        # self.psych_gamma = psych_gamma         
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        if not hasattr(self, 'psych_gamma_lambda_triple_samples'):
            return {}
        return {
            "psych_gamma_lambda_triple_samples" : self.psych_gamma_lambda_triple_samples,
            "sampler_stat" : self.sampler_stat,
            "psych_alpha_samples" : self.psych_alpha_samples,
            "psych_beta_samples" : self.psych_beta_samples,
            "sig_b_samples" : self.sig_b_samples,
            "sig_a_samples" : self.sig_a_samples,
            "psych_lambda" : self.psych_lambda,
            "psych_z" : self.psych_z,
            "psych_gamma" : self.psych_gamma,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self    
    
    def prob_comp_graph(self, X, sig_a, sig_b, 
            psych_alpha, 
            psych_beta,                         
            psych_gamma_lambda):
        """Creates the computation probabilistic graphical model for all the variables and the observations"""
        expanded_gamma = tf.expand_dims(tf.gather(psych_gamma_lambda, 0, axis=-1), axis=1)
        expanded_z = tf.expand_dims(tf.gather(psych_gamma_lambda, 2, axis=-1), axis=1)
        expanded_psych_beta = tf.expand_dims(psych_beta, axis=1)
        expanded_sig_b = tf.expand_dims(sig_b, axis=1)
        psych_out = tf.math.sigmoid(tf.linalg.matmul(psych_alpha, X, transpose_b=True) + expanded_psych_beta)
        sig_out= tf.math.sigmoid(tf.linalg.matmul(sig_a, X, transpose_b=True) + expanded_sig_b)
        # tf.print ("sig_a:", sig_a, sig_a.shape)
        # tf.print ("sig_b:", sig_b)
        # tf.print ("psych_alpha:", psych_alpha, psych_alpha.shape)
        # tf.print ("psych_beta:", psych_beta)
        # tf.print ("pych_alpha * X:", tf.linalg.matmul(psych_alpha, X, transpose_b=True))
        # tf.print ("pych_alpha * X + beta:", tf.linalg.matmul(psych_alpha, X, transpose_b=True) + expanded_psych_beta)
        # tf.print ("psych_out:", psych_out)
        # tf.print ("psych_gamma_lambda:", psych_gamma_lambda, psych_gamma_lambda.shape)
        # tf.print ("tf.multiply(expanded_z, psych_out):", tf.multiply(expanded_z, psych_out), tf.multiply(expanded_z, psych_out).shape)
        # tf.print ("expanded_gamma + tf.multiply(expanded_z, psych_out):", expanded_gamma + tf.multiply(expanded_z, psych_out), ( expanded_gamma + tf.multiply(expanded_z, psych_out)).shape)
        # tf.print ("sig_out:", sig_out)
        # tf.print ("the whole product:", (tf.multiply(expanded_gamma + tf.multiply(expanded_z, psych_out), sig_out)))
        log_p = tf.math.log(expanded_gamma + tf.multiply(expanded_z, psych_out)) + tf.math.log(sig_out)
        log_1_minus_p = tf.math.log(1 - tf.math.multiply(expanded_gamma + tf.multiply(expanded_z, psych_out), sig_out))
        return tfd.Independent(tfd.Bernoulli(logits=log_p - log_1_minus_p, validate_args=True)
        # return tfd.Independent(tfd.Bernoulli(probs=tf.multiply(expanded_gamma + tf.multiply(expanded_z, psych_out), sig_out), validate_args=True)
            , reinterpreted_batch_ndims=2)

    def build(self, X, 
         alpha =[1., 1.,  1.],
         mu_psych_alpha=None,
         mu_psych_beta=0., 
         mu_sig_alpha=None,
         mu_sig_beta=0., 
         scale_sig_alpha = 1., 
         scale_psych_alpha = 1.,
         scale_sig_beta = 1., 
         scale_psych_beta = 1., 
                            ):
        if mu_psych_alpha is None:
            mu_psych_alpha = [0.] * X.shape[-1]
        if mu_sig_alpha is None:
            mu_sig_alpha = [0.] * X.shape[-1]
        """Creates a joint distribution for the varying intercept model."""
        # mu_psych_alpha = tf.Variable([0., 0., 0.],  dtype=common_dtype)
        # mu_psych_beta = tf.Variable(0., dtype=common_dtype)
        # mu_sig_alpha = tf.Variable([0., 0., 0.],  dtype=common_dtype)
        # mu_sig_beta = tf.Variable(0., dtype=common_dtype)
        return tfd.JointDistributionSequential([
            tfd.Sample(tfd.Dirichlet(concentration=alpha), sample_shape=1),# psych_gamma_lambda
            tfd.Sample(tfd.Normal(loc=mu_psych_beta, scale=scale_psych_beta), sample_shape=1), # psych_beta
            tfd.Sample(tfd.Independent(tfd.Normal(loc=mu_psych_alpha, scale=[scale_psych_alpha] * X.shape[-1]), reinterpreted_batch_ndims=1), sample_shape=1), #sig_b        
            tfd.Sample(tfd.Normal(loc=mu_sig_beta, scale=scale_sig_beta), sample_shape=1), # psych_beta
            tfd.Sample(tfd.Independent(tfd.Normal(loc=mu_sig_alpha, scale=[scale_sig_alpha] * X.shape[-1]), reinterpreted_batch_ndims=1), sample_shape=1), #sig_a        
            lambda sig_a, sig_b, psych_alpha, psych_beta, psych_gamma_lambda:  self.prob_comp_graph(X, sig_a, sig_b, psych_alpha, psych_beta, psych_gamma_lambda)   #y
            # tfd.Independent(tfd.Deterministic(tf.linalg.matmul(alpha, X)),  reinterpreted_batch_ndims=2)

            # tfd.Bernoulli(tf.math.sigmoid(tf.linalg.matmul(alpha, X)))

            # tfd.HalfCauchy(loc=0., scale=5),	# sigma_a
            # lambda sigma_a, mu_a: tfd.MultivariateNormalDiag(	# a
            #         loc=affine(tf.ones([num_counties]), mu_a[..., tf.newaxis]),
            #         scale_identity_multiplier=sigma_a),
            # tfd.Normal(loc=0., scale=1e5),		# b
            # tfd.HalfCauchy(loc=0., scale=5),	# sigma_y
            # lambda sigma_y, b, a: tfd.MultivariateNormalDiag(	# y
            #         loc=affine(floor, b[..., tf.newaxis], tf.gather(a, county, axis=-1)),
            #         scale_identity_multiplier=sigma_y)
    ])
    def fit(self, X, l, nchain=10):        
        X = X['sig_input']
        l_y_cat_decimal = self.encoder.inverse_transform(l)
        l = (l_y_cat_decimal / 2).astype(int)
        
        self.prob_model = self.build(X)
        
        alpha,\
        mu_psych_alpha,\
        mu_psych_beta,\
        mu_sig_alpha,\
        mu_sig_beta, _= self.prob_model.sample(nchain)
        
        from time import time
        t = time()
        init_state = [alpha,
        mu_psych_alpha,
        mu_psych_beta,
        mu_sig_alpha,
        mu_sig_beta]
        step_size = [.1] * len(init_state)
        target_log_prob_fn = lambda *x: self.prob_model.log_prob(x + (l, ))

        # bijector to map contrained parameters to real
        unconstraining_bijectors = [tfb.SoftmaxCentered()] + [tfb.Identity()] * (len(self.prob_model.sample())-2)
        
        # y = np.mod(l, 2)
        
        
        samples, sampler_stat = run_chain(
            init_state, step_size, target_log_prob_fn, unconstraining_bijectors)
        print ("time taken for sampling:", time() - t, "seconds...")        
        self.psych_gamma_lambda_triple_samples,\
        self.psych_beta_samples,\
        self.psych_alpha_samples,\
        self.sig_b_samples,\
        self.sig_a_samples = samples
        self.sampler_stat = sampler_stat
        
        self.psych_alpha_samples = self.psych_alpha_samples.numpy()
        self.psych_beta_samples = self.psych_beta_samples.numpy()
        self.sig_b_samples = self.sig_b_samples.numpy()
        self.sig_a_samples =  self.sig_a_samples.numpy()
        
        self.psych_gamma = self.psych_gamma_lambda_triple_samples[..., 0][..., None].numpy()
        self.psych_lambda = self.psych_gamma_lambda_triple_samples[..., 1][..., None].numpy()
        self.psych_z = self.psych_gamma_lambda_triple_samples[..., 2][..., None].numpy()
        del(self.prob_model)
        self.prob_model = None
        
        
        return self
        
        
    def predict_prob_y_given_x(self, X, lookback_num=500, raw=False):
        X_ = X['sig_input']
        main_out = expit(self.sig_a_samples[-lookback_num:].dot(X_.T) + self.sig_b_samples[-lookback_num:][..., None])
        if raw:
            return main_out
        else:
            return main_out.mean(axis=0).mean(axis=0)[0]

    def predict_prob_l_given_y_x(self, X, lookback_num=500, raw=False):
        X_ = X['sig_input']
        
        main_out = (self.psych_gamma + self.psych_z * expit(self.psych_alpha_samples[-lookback_num:].dot(X_.T) + self.psych_beta_samples[-lookback_num:][..., None]))
        if raw:
            return main_out
        else:
            return main_out.mean(axis=0).mean(axis=0)[0]
        
    def predict_proba(self, X, loockback_num=500):        
        t_of_x = self.predict_prob_y_given_x(X, raw=True)
        s_of_x = self.predict_prob_l_given_y_x(X, raw=True)
        return np.multiply(t_of_x, s_of_x).mean(axis=0).mean(axis=0)[0]
        
        
    def predict(self, X):
        self.predict_proba(X) > 0.5
        
        