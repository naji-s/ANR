from sklearn.base import BaseEstimator
from scipy.special import logsumexp, softmax, expit
from numpy import random, multiply, arange, tile, log, squeeze, eye, diag, true_divide, inf, asarray, power, logical_and, exp, zeros, einsum, linalg, ndim, copy, hstack, ones, var, vstack, asarray, float64, std
from scipy.optimize import minimize as scipy_minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numexpr as ne
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph, csr_matrix
from numpy.linalg  import LinAlgError 
from scipy.sparse.linalg import eigsh
import numexpr as ne
from scipy.linalg.blas import sgemm
from sys import path
path.append('/home/scratch/nshajari/psych_model/utils')
from tensor_utils import k_nn_graph, convert_sparse_tensor_to_csr_matrix


def rbf(X, Y=None, lengthscale=1., var=1.):
    if Y is None:
        raise ValueError("Y value for kernel needs to be set. Otherwise use featurizer()")
    if ndim(X) == 1:
        X = X[None, ...]
    if ndim(Y) == 1:
        Y = Y[None, ...]
    gamma = 1. / (2 * lengthscale)
    """C := alpha*op( A )*op( B ) + beta*C,"""
    try:
        X_norm = -gamma*einsum('ij,ij->i',X,X)
        Y_norm = -gamma*einsum('ij,ij->i',Y,Y)
    except Exception as e:
        raise type(e)(str(e) + ' ' + str(X) + ',' + str(Y))
    return ne.evaluate('v * exp(A + B + C)', {\
        'A' : X_norm[:,None],\
        'B' : Y_norm[None,:],\
        'C' : sgemm(alpha=2.0*gamma, a=X, b=Y, trans_b=True),\
        'v' : var\
    })
def linear(X, Y=None, lengthscale=1., var=1.):
    if Y is None:
        raise ValueError("Y value for kernel needs to be set. Otherwise use featurizer()")
    if ndim(X) == 1:
        X = X[None, ...]
    if ndim(Y) == 1:
        Y = Y[None, ...]
    gamma = 1. / lengthscale * var
    """C := alpha*op( A )*op( B ) + beta*C,"""
#     return gamma*einsum('ij,ij->i',X,Y)
    return ne.evaluate('(A + B + C)', {\
        'A' : 0,\
        'B' : 0,\
        'C' : sgemm(alpha=gamma, a=X, b=Y, trans_b=True),\
    })
class PythonPsychM(BaseEstimator):
    def __init__(self, encoder=None, sig_a=None, sig_b=None, psych_alpha=None, psych_beta=None, psych_gamma=None, psych_lambda=None, g_prime=None, l_prime=None, kernel=None,
        loss_type=None,
                 num_of_ppsych_models=None,
        is_SPM=False,
        sig_reg_coeff=.0,
        sig_reg_penalty='l2',
        ssl_type='SSL_LBO', 
        psych_reg_coeff=.0,
        psych_reg_penalty='l2',
        gp_kernel_lengthscale=1.,                 
        gp_kernel_var=1.,
        lap_kernel_lengthscale=1.,
        lap_kernel_var=1.,
        lap_kernel_k = 5,
        lap_connectivity='distance',
        lap_kernel_type='rbf',
        gp_kernel_type='rbf',
        lbo_temperature=1.,
        lap_kernel_noise=.01,
        lap_kernel_power=5,
        lap_kernel_normed=False):
        super().__init__()
        self.encoder = encoder
        self.g_prime = g_prime
        self.l_prime = l_prime        
        self.sig_a = sig_a
        self.sig_b = sig_b
        self.psych_alpha = psych_alpha
        self.psych_beta = psych_beta
        self.psych_lambda = psych_lambda
        self.psych_gamma = psych_gamma
        self.kernel = kernel
        self.is_SPM = is_SPM
        self.gp_kernel_var = gp_kernel_var
        self.lap_kernel_var = lap_kernel_var
        self.lap_kernel_k = lap_kernel_k
        self.lap_connectivity = lap_connectivity
        self.lap_kernel_type = lap_kernel_type
        self.gp_kernel_type = gp_kernel_type
        self.gp_kernel_lengthscale = gp_kernel_lengthscale
        self.lbo_temperature = lbo_temperature
        self.lap_kernel_lengthscale = lap_kernel_lengthscale
        self.lap_kernel_noise = lap_kernel_noise
        self.lap_kernel_power = lap_kernel_power
        self.lap_kernel_normed = lap_kernel_normed
        self.sig_reg_coeff = sig_reg_coeff
        self.sig_reg_penalty =sig_reg_penalty
        self.psych_reg_coeff = psych_reg_coeff
        self.psych_reg_penalty = psych_reg_penalty
        self.ssl_type = ssl_type
        self.loss_type = loss_type
        self.num_of_ppsych_models = num_of_ppsych_models
    
    def partition_params(self, params=None):
        sig_a = params[:self.kernel_mat_dim]
        sig_b = params[self.kernel_mat_dim:self.kernel_mat_dim+1]
        psych_alpha = params[self.kernel_mat_dim + 1: self.kernel_mat_dim + self.input_dim + 1]
        psych_beta = params[self.input_dim + self.kernel_mat_dim + 1: self.kernel_mat_dim + self.input_dim + 2]
        if self.is_SPM:
            return sig_a, sig_b, psych_alpha, psych_beta 
        else:
            g_prime = squeeze(params[self.kernel_mat_dim + self.input_dim + 2: self.kernel_mat_dim + self.input_dim + 3])
            l_prime = squeeze(params[self.kernel_mat_dim + self.input_dim + 3 : self.kernel_mat_dim + self.input_dim + 4])
            return sig_a, sig_b, psych_alpha, psych_beta, g_prime, l_prime
    
    def regularization_loss(self, params=None):
        reg_loss = 0
        if self.is_SPM:
            sig_a, sig_b,\
            psych_alpha, psych_beta = self.partition_params(params)  
        else:
            sig_a, sig_b,\
            psych_alpha, psych_beta,\
            _, _ = self.partition_params(params)  
        reg_loss += linalg.norm(psych_alpha) ** 2 * self.psych_reg_coeff
        
        reg_loss += (sig_a.T.dot(self.kernel_mat).dot(sig_a)  + linalg.slogdet(self.kernel_mat)[1]) / (2 * self.kernel_mat_dim)
        L_t = self.kernel_mat.dot(sig_a)
        if 'lbo' in self.ssl_type.lower():
            Lap_reg = self.exp_Lap_mat
        elif 'laplacian' in self.ssl_type.lower():
            Lap_reg = self.Lap_mat
        reg_loss += (L_t.T.dot(Lap_reg).dot(L_t) - linalg.slogdet(Lap_reg)[1]) / (2 * self.kernel_mat_dim)
        return reg_loss
    
    
    
    def fit(self, X, y, minimization_method='L-BFGS-B', p=.05, warm_start=False):
        self.input_dim = X['sig_input'].shape[-1]
#         self.sig_input_ = hstack((X['sig_input'], 10. * ones((len(X['sig_input']), 1))))
#         self.psych_input_ = hstack((X['psych_input'], 10. * ones((len(X['psych_input']), 1))))
        self.sig_input_ = X['sig_input'].astype(float64)
        self.psych_input_ = X['psych_input'].astype(float64)
        
        l_y_decimal_cal = self.encoder.inverse_transform(y)        
        l = (l_y_decimal_cal / 2).astype(int)
#         k = int(p * len(l))
#         idx = random.choice(range(len(l)),k)
#         l[idx] = 1 - l[idx]
        
        self.l_ = l
        self.kernel_mat_dim = len(self.sig_input_)
        self.calculate_kernels(self.sig_input_)
        self.data_dim = self.sig_input_.shape[-1]
        if self.is_SPM:
            self.g_prime_ = -1e2
            self.l_prime_ = -1e2

        if warm_start:
            params_dict = self.get_params()
            if self.is_SPM:                
                x_0 = squeeze(hstack([params_dict['sig_a'].reshape((-1, )), asarray(params_dict['sig_b']).reshape((1,)), params_dict['psych_alpha'].reshape((-1,)), asarray(params_dict['psych_beta']).reshape((1,))]))
            else:
                x_0 = squeeze(hstack([params_dict['sig_a'].reshape((-1, )), asarray(params_dict['sig_b']).reshape((1,)), params_dict['psych_alpha'].reshape((-1, )), asarray(params_dict['psych_beta']).reshape((1,)),
                             asarray(params_dict['g_prime']).reshape((1,)), asarray(params_dict['l_prime']).reshape((1,))]))
        else:
            if self.is_SPM:
                x_0 = random.randn(self.data_dim + self.kernel_mat_dim + 2).astype(float64)
            else:
                x_0 = random.randn(self.data_dim + self.kernel_mat_dim + 4).astype(float64)
        total_loss = lambda x: self.softmax_loss(sig_input=self.sig_input_, psych_input=self.psych_input_, l=l, params=x) + self.regularization_loss(params=x)
#         print ("x_0:", x_0)
#         if minimization_method == 'BFGS':
        result = scipy_minimize(lambda x: total_loss(x), x_0, method=minimization_method, options={'maxiter':30000, 'maxfun':30000})
        x = result['x']
        if self.is_SPM:
            self.sig_a_, self.sig_b_,\
            self.psych_alpha_, self.psych_beta_ = self.partition_params(x)
        else:
            self.sig_a_, self.sig_b_,\
            self.psych_alpha_, self.psych_beta_,\
            self.g_prime_, self.l_prime_ = self.partition_params(x)
        self.psych_gamma_, self.psych_lambda_ = self.g_l_prime_to_gamma_lambda_transformer(self.g_prime_, self.l_prime_)
        self.final_loss = result.fun
        self.final_success = result.success
        
        return self

    def g_l_prime_to_gamma_lambda_transformer(self, g_prime, l_prime):#, reparametrization_style=3):       


        output = softmax([0., -g_prime, -l_prime])
        gamma_ = output[1]
        lambda_ = output[2]

        return gamma_, lambda_   
#     def gamma_lambda_to_g_l_prime_transformer(self, gamma_, lambda_):#, reparametrization_style=3):

#         constant = log(1. - gamma_ - lambda_)
#         g_prime = log(gamma_) - constant
#         l_prime = log(lambda_) - constant

#         return g_prime, l_prime                                            

    def softmax_loss(self, sig_input=None, psych_input=None, l=None, params=None, is_padded=False):
        """sig_input needs to be padded here"""
        if self.is_SPM:
            sig_a, sig_b,\
            psych_alpha, psych_beta = self.partition_params(params)
            g_prime, l_prime = self.g_prime_, self.l_prime_
            
        else:
            sig_a, sig_b,\
            psych_alpha, psych_beta,\
            g_prime, l_prime = self.partition_params(params)

        L_s = psych_input.dot(psych_alpha) + psych_beta
#         L_t = self.featurizer(sig_input, is_padded=is_padded).dot(sig_a) + sig_b
        L_t = self.kernel_mat.dot(sig_a) + sig_b

#             L_s = psych_linear
#             L_t = sig_linear
#             logsumexp = tf.math.reduce_logsumexp

        # NOTICE: 
        # C is not appearing because C = E - D + F
        # Also A = log(D_star) + G + H
        # Also C = E - D + F
        L_s = L_s.reshape((-1, 1))
        L_t = L_t.reshape((-1, 1))
        shape = L_s.shape
        zero = 0.

            
        


        A_leftover = multiply(l, logsumexp(hstack([tile(-g_prime, shape), - g_prime - L_s,  tile(0., shape)]), axis=1))        
        H_F_B_G =  - logsumexp(hstack([tile(zero, shape), -L_s]), axis=1) - logsumexp(hstack([tile(zero, shape), -L_t]), axis=1) \
        - logsumexp(hstack([tile(zero, shape), tile(-l_prime, shape), tile(- g_prime,shape)]), axis=1)
        E_D = multiply(1-l, logsumexp(hstack([tile(-l_prime, shape), - L_s, - L_s -l_prime, -L_t, -L_t - g_prime, -L_t - l_prime, -L_s - L_t, -L_s - L_t - g_prime, -L_s - L_t - l_prime]), axis=1))
        log_like = A_leftover + H_F_B_G + E_D
        return -log_like.mean()
    
    def calculate_kernels(self, features, is_padded=True):
        #######################################################
        ### BEGIN: CALCULATING THE LAPLACIAN CASE AND THE HEAT KERNEL
        #######################################################
        features = features.astype(float64)
        if self.lap_connectivity=='connectivity':
#             W = k_nn_graph(features, k=self.lap_kernel_k, mode='connectivity',include_self=False)
            W = kneighbors_graph(features, self.lap_kernel_k, mode='connectivity',include_self=False)
#             W = convert_sparse_tensor_to_csr_matrix(W)
            W = (((W + W.T) > 0) * 1.)
        elif self.lap_connectivity=='distance':
#             if type(features).__name__ != 'ndarray':
#                 if tf.keras.backend.shape(features)[0] >= 1:
#                     features = features.numpy()
#             W = k_nn_graph(features, k=self.lap_kernel_k, mode='distance',include_self=False)
#             W = convert_sparse_tensor_to_csr_matrix(W)
            W = kneighbors_graph(features, self.lap_kernel_k, mode='distance',include_self=False)
            W = W.maximum(W.T)
        # checking for floating point error
        largest_power = 200
        W_square = W.data ** 2
#         W_square = tf.math.square(W.data)
        W_square_good = W_square < 4 * largest_power * self.lap_kernel_lengthscale
        W_square = W_square / (4 * self.lap_kernel_lengthscale)


        if self.lap_kernel_type == 'rbf':
            W_square[W_square_good] = exp(-W_square[W_square_good])
            W_square[~W_square_good] = 0.
#             W_square = tf.math.exp(-W_square / (4. * self.lap_kernel_lengthscale))

        elif self.lap_kernel_type == 'linear':
            pass
            
        else:
            raise NotImplementedError("kernel type is:" + self.lap_kernel_type)
        W.data = W_square#.numpy()
        Lap_mat = csgraph.laplacian(W, normed=self.lap_kernel_normed)
        temp_Lap_mat = Lap_mat#.toarray()
        svd_func = lambda x: eigsh(A=x, k=self.kernel_mat_dim-1, maxiter=10000)
        D, U = svd_func(temp_Lap_mat)
        U_T = U.T

        if self.ssl_type is not None and 'laplacian' in self.ssl_type.lower():
            D = D.astype(float64)
            D[D<1e-100] = 1e-100
            log_noisy_D = D + self.lap_kernel_noise
            log_noisy_D = log(log_noisy_D)
            log_noisy_D_too_big_p = log_noisy_D > largest_power
            log_noisy_D_too_small_p = log_noisy_D < -largest_power

#             print ("TEST")
            noisy_D = D + self.lap_kernel_noise
            powered_D = copy(D)
            powered_D[log_noisy_D_too_big_p] = exp(self.lap_kernel_power)
            powered_D[log_noisy_D_too_small_p] = 0.
            good_log_noisy_D_idx = logical_and(~log_noisy_D_too_big_p, ~log_noisy_D_too_small_p)
            powered_D[good_log_noisy_D_idx] = power(noisy_D[good_log_noisy_D_idx], self.lap_kernel_power)
            self.Lap_mat = U.dot(diag(powered_D).dot(U_T))
            self.exp_Lap_mat = None

        elif self.ssl_type is not None and  'lbo' in self.ssl_type.lower():
            temperature = self.lbo_temperature
            D = -temperature * D
            D = exp(D)
            self.exp_Lap_mat = U @ diag(D) @ U_T * self.lap_kernel_var
            self.Lap_mat = None
#     #######################################################
#     ### END: CALCULATING THE LAPLACIAN CASE AND THE HEAT KERNEL
#     #######################################################

        if self.gp_kernel_type == 'linear':
            kernel_func = lambda x, y: linear(x, y, lengthscale=self.gp_kernel_lengthscale, var=self.gp_kernel_var)

        elif self.gp_kernel_type == 'rbf':
            kernel_func = lambda x, y: rbf(x, y, lengthscale=self.gp_kernel_lengthscale, var=self.gp_kernel_var)
#             kernel_func = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=tf.math.sqrt(tf.cast(self.gp_kernel_var, tf.float64)), length_scale=tf.math.sqrt(tf.cast(self.gp_kernel_lengthscale, tf.float64)), feature_ndims=1, validate_args=True,
#         name='PythonExponentiatedQuadratic')
        elif self.gp_kernel_type == 'laplacian':
            raise NotImplementedError("TODO: Laplacian kernel is not implemented yet")
        else:
            raise NotImplementedError("print self.gp_kernel_type is not assigned")

        if self.gp_kernel_type == 'laplacian':

            pass
        else:
#             self.kernel = lambda x, y: kernel_func.matrix(x, y).numpy()
            self.kernel = lambda x, y: kernel_func(x, y)

        if self.gp_kernel_type == 'laplacian':
            if self.manifold_kernel_type.lower() == 'lbo':
#                     kernel_mat = tf.linalg.inv(Lap_mat)
                pass
            elif self.manifold_kernel_type.lower() == 'laplacian':
#                     kernel_mat = tf.linalg.inv(exp_Lap_mat)
                pass
            else:
                raise NotImplementedError
        else:
            self.kernel_mat = self.featurizer(features, is_padded=is_padded)

    def featurizer(self, X, is_padded=False):
#         if not padded:
#             X = hstack((X, 10 * ones((X.shape[0], 1))))
        return self.kernel(X, self.sig_input_)
    
    def predict_prob_y_given_x(self, X):
        if self.is_SPM and (self.ssl_type is not None and 'SSL' in self.ssl_type) and linalg.norm(self.psych_alpha_) >= linalg.norm(self.sig_a_) and not self.SPM_switch_call:
            self.SPM_switch_call = True
            output = self.predict_prob_l_given_y_x(X)
            self.SPM_switch_call = False
            return output
        X_ = X['sig_input']

        
        
        if self.ssl_type is not None and ('LBO' in self.ssl_type or 'laplacian' in self.ssl_type):
            X_ = self.featurizer(X_, is_padded=False)
        linear_out = X_.dot(self.sig_a_) + self.sig_b_
        output = expit(linear_out)
        return output



    def predict_prob_l_given_y_x(self, X):       
        if self.is_SPM and (self.ssl_type is not None and 'SSL' in self.ssl_type) and linalg.norm(self.psych_alpha_) >= linalg.norm(self.sig_a_) and not self.SPM_switch_call:
            self.SPM_switch_call = True
            output = self.predict_prob_y_given_x(X)
            self.SPM_switch_call = False
            return output
        X_ = X['psych_input']
        linear_output = X_.dot(self.psych_alpha_) + self.psych_beta_
        output = expit(linear_output) * (1 - self.psych_gamma_ - self.psych_lambda_) + self.psych_gamma_
        
        return output



    def predict_l_given_y_x(self, X):
        return self.predict_prob_l_given_y_x(X)>=0.5


    def predict_y_given_x(self, X):
        return self.predict_prob_y_given_x(X)>=0.5


    def predict_proba(self, X):
        p_y_given_X_arr = self.predict_prob_y_given_x(X)
        p_l_given_x_y_arr = self.predict_prob_l_given_y_x(X)
        assert (p_y_given_X_arr.shape == p_l_given_x_y_arr.shape)
        return multiply(p_y_given_X_arr, p_l_given_x_y_arr)

    def predict(self, X, threshold = 0.5):
        return self.predict_proba(X)>=0.5    