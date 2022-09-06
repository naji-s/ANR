"""Rewritten psychometric layer (older one was psych_TF) for TF2"""
# import tensorflow as tf
import sys
from scipy.linalg import cholesky
import sys
sys.path.append('/home/scratch/nshajari/psych_model/')
sys.path.append('/home/scratch/nshajari/psych_model/utils/')
sys.path.append('/home/scratch/nshajari/psych_model/LPUModels/')
import logging
import os
EPSILON = 1e-16
DELTA = 1e-4
from scipy.linalg import qr_multiply, qr
from scipy.linalg.lapack import dtrtri

def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
# import gpflow
import tensorflow_probability as tfp
from sklearn.neighbors import kneighbors_graph
# from utils.func_lib import convert_sparse_matrix_to_sparse_tensor
from scipy import sparse, spatial
from scipy.sparse.linalg import expm#, fractional_matrix_power
tf.keras.backend.set_floatx('float64')
from scipy.sparse import csgraph, csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence as ArpackException
from scipy.linalg import svd as scipy_svd
from numpy.linalg import LinAlgError, matrix_power
from numpy.linalg import svd as numpy_svd
from numpy import copy, exp
import warnings
# from utils.math_utils import sparse_std
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import svds, eigsh
from tensor_utils import k_nn_graph, convert_sparse_tensor_to_csr_matrix

# from scipy.sparse.linalg.eigsh
def svd_decompose(Lap_mat, padding=False):
    try:
        svd_func = lambda x: eigsh(A=x, k=Lap_mat.shape[0]-1, maxiter=50000)
        D, U = svd_func(Lap_mat)
    except Exception as e:
        raise type(e)("svd_func in svd_decompose() raises this error:" + str(e))
    if padding:
        U = np.hstack((U, np.ones(D.shape[0] + 1).reshape((-1, 1)) / np.sqrt(D.shape[0]+1)))
        D = np.hstack((D, np.zeros(1))) 
    decomposition_dict = dict()
    decomposition_dict['D_vec'] = D
    decomposition_dict['U'] = U
    return decomposition_dict    

def invert_mat_with_cholesky(mat, with_scipy=False):
    counter = 0
    while True:
        eps_diag = np.diag(np.ones(mat.shape[0], dtype=np.float64) * DELTA)
        biased_mat = mat + (counter > 0) * eps_diag * 10 ** counter
        try:
            if with_scipy:
                mat_cholesky = cholesky(tf.cast(biased_mat, tf.double))
            else:
                mat_cholesky = tf.linalg.cholesky(biased_mat).numpy()
                if np.isnan(mat_cholesk).any():
                    print ("FUCK ME!")
                    raise

        except:
            counter += 1
            # logging = tf.get_logger()
            # logging.warning(str(e) + "inverting with cholesky is failing, trial num:" + str(counter))
            if counter > 5:
                unnorm_M_inv = np.eye(mat.shape[0])
                break
            continue
        if with_scipy:
            mat_cholesky_inv = dtrtri(mat_cholesky)[0]
        else:
            mat_cholesky_inv = tf.linalg.triangular_solve(tf.cast(mat_cholesky, tf.double), tf.linalg.eye(mat_cholesky.shape[0], dtype=tf.double), lower=True).numpy().T

        #multiplying by self.manifold_kernel_amplitude ** 2 since M is already normalized here                
        unnorm_M_inv = mat_cholesky_inv @ mat_cholesky_inv.T
        break
    return unnorm_M_inv
# def invert_mat_with_cholesky(mat):
#     counter = 0
#     while True:
#         eps_diag = np.diag(np.ones(mat.shape[0], dtype=np.float64) * DELTA)
#         biased_mat = mat + (counter > 0) * eps_diag * 10 ** counter
#         try:
#             mat_cholesky = cholesky(biased_mat)
#         except LinAlgError as e:
#             counter += 1
#             logging = tf.get_logger()
#             logging.warning(str(e) + "inverting with cholesky is failing, trial num:" + str(counter))
#             if counter > 10:
#                 unnorm_M_inv = np.eye(mat.shape[0])
#                 break
#             continue
#         mat_cholesky_inv = dtrtri(mat_cholesky)[0]
#         #multiplying by self.manifold_kernel_amplitude ** 2 since M is already normalized here
#         unnorm_M_inv = mat_cholesky_inv @ mat_cholesky_inv.T
#         break
#     return unnorm_M_inv


def build_W(features, k_neighbours, lengthscale, connectivity):
    try:
        if connectivity=='connectivity':
            W = kneighbors_graph(features, k_neighbours, mode='connectivity', include_self=False)
            W = (((W + W.T) > 0) * 1.)
            W.data = np.squeeze(W.data) / (2 * lengthscale ** 2)

        elif connectivity=='distance':
            if type(features).__name__ != 'ndarray':
                if tf.keras.backend.shape(features)[0] >= 1:
                    features = features.numpy()
            W = kneighbors_graph(features, k_neighbours, mode='distance',include_self=False)
            W = W.maximum(W.T)
            W.data = np.square(W.data).ravel() 
            W.data = np.squeeze(W.data) / (2 * lengthscale ** 2)
            W.data = tf.math.exp(-W.data).numpy()
            W.data = np.nan_to_num(W.data)

    except Exception as e:
        raise type(e)(str(e) + 'error is in kneighbors_graph')

    # THE SPARSE CASE USING TENSORFLOW. k_nn_graph IMPLEMENTATION CURRENTLY IS BROKEN
    # W = k_nn_graph(features, k=self.manifold_kernel_k, mode=self.opt['neighbor_mode'], include_self=False)
    # W = tf.sparse.maximum(W, tf.sparse.transpose(W))
    # W = convert_sparse_tensor_to_csr_matrix(W
    # checking for floating point error

    return W


def calculate_L_p_kernel(Lap_mat, manifold_kernel_noise, manifold_kernel_power, 
                         manifold_kernel_amplitude, use_eigsh=False, return_inverse=False):
    manifold_mat_inv = None

    if use_eigsh:
        # raise NotImplementedError("eigsh is implemented with a big mistake! needs correction: SVD cannot inverse that easily! the inverse of UDU^T is not U^T *1/D* U unfortunately!") 
        try:
            largest_power = np.inf
            decomposition_dict = svd_decompose(Lap_mat, padding=True)
            U = decomposition_dict['U']
            U_T = U.T
            noisy_D = decomposition_dict['D_vec'] + manifold_kernel_noise
            D_vec_powered = noisy_D ** manifold_kernel_power
            # log_noisy_D = tf.math.log(noisy_D).numpy()
            # log_noisy_D[log_noisy_D > largest_power] = largest_power
            # log_noisy_D_too_small_p = log_noisy_D < -largest_power
            # L_p_mat = tf.linalg.matmul(tf.linalg.matmul(U, D_vec_powered), U_T).numpy()\
            #                                         * self._manifold_kernel_amplitude ** 2

            manifold_mat = U @ np.diag(D_vec_powered) @ U_T * manifold_kernel_amplitude ** 2
            if return_inverse:
                manifold_mat_inv = U @ np.diag(1. / D_vec_powered) @ U_T / manifold_kernel_amplitude ** 2

            # print ("WTF???", manifold_mat @ manifold_mat_inv)
        except Exception as e:
            raise type(e)("caclulating L_r raises this error when relying on eigsh:" + str(e))
    else:
        decomposition_dict = None
        noisy_L = Lap_mat + manifold_kernel_noise * np.eye(Lap_mat.shape[0])
        noisy_L_p = matrix_power(noisy_L, manifold_kernel_power)
        # noisy_L_p_cholesky = cholesky(noisy_L_p)
        # noisy_L_p_cholesky_inv = dtrtri(noisy_L_p_cholesky)[0]
        from scipy.sparse import csr_matrix 
        manifold_mat = noisy_L_p * manifold_kernel_amplitude ** 2
        if return_inverse:
            manifold_mat_inv = invert_mat_with_cholesky(noisy_L_p)
            manifold_mat_inv = manifold_mat_inv / manifold_kernel_amplitude ** 2


    return manifold_mat, decomposition_dict, manifold_mat_inv

def calculate_heat_kernel(Lap_mat, lbo_temperature, manifold_kernel_amplitude, use_eigsh, return_inverse=False):
    if use_eigsh:
        # raise NotImplementedError("eigsh is implemented with a big mistake! needs correction: SVD cannot inverse that easily! the inverse of UDU^T is not U^T *1/D* U unfortunately!") 
        try:
            decomposition_dict = svd_decompose(Lap_mat)
            D_vec = decomposition_dict['D_vec']
            U = decomposition_dict['U']
            U_T = U.T
            D = tf.math.exp(-lbo_temperature  * D_vec)
        except Exception as e:
            raise type(e)(str(e)+'Exponentiating D failed')
        try:
            exp_Lap_mat = tf.linalg.matmul(tf.linalg.matmul(U, 
                                                            tf.linalg.diag(tf.experimental.numpy.ravel(D))), U_T) *\
                                                            manifold_kernel_amplitude ** 2
            if return_inverse:
                exp_Lap_mat_inv = tf.linalg.matmul(
                                                tf.linalg.matmul(U, 
                                                    tf.linalg.diag(1. / tf.experimental.numpy.ravel(D))), U_T) /\
                                                        manifold_kernel_amplitude ** 2
            else: 
                exp_Lap_mat_inv = None


        except Exception as e:
            raise type(e)(str(e)+"exp_lap_mat calculation is failing WTF?!")
    else:
        decomposition_dict = None
        try:
            exp_Lap_mat = expm(-lbo_temperature * Lap_mat) * manifold_kernel_amplitude ** 2
            if return_inverse:
                exp_Lap_mat_inv = expm(lbo_temperature * Lap_mat) / manifold_kernel_amplitude ** 2
            else:
                exp_Lap_mat_inv = None



        except Exception as e:
            raise type(e)("exponentation L using \
                                scipy.sparse.linalg.expm is raising this error:" +\
                          str(e) + " Lap_mat has type:"+str(type(Lap_mat)))            

    return exp_Lap_mat, decomposition_dict, exp_Lap_mat_inv

def build_manifold_mat(W, manifold_kernel_noise, manifold_kernel_type, manifold_kernel_power, manifold_kernel_amplitude, lbo_temperature, manifold_kernel_normed=False, use_eigsh=False, return_inverse=True):
#        if use_eigsh:
        # raise NotImplementedError("eigsh is implemented with a big mistake! needs correction: SVD cannot inverse that easily! the inverse of UDU^T is not U^T *1/D* U unfortunately!") 
    try:
        if hasattr(W.data, 'numpy'):
            tf.print ("we have nunmpy property")
            W.data = W.data.numpy()
        try:
            # tf.print ("YEAH FUCKER!!:",W.data.ndim, W.indices.ndim, W.indptr.ndim, output_stream=sys.stderr)
            Lap_mat = csgraph.laplacian(W, normed=manifold_kernel_normed).astype(np.float64).tocsc()
        except Exception as e:
            raise type(e)(str(e)+"you fucking CSRGRAPH!!!" + str(type(W)))



        if 'lbo' in manifold_kernel_type.lower():
            try:
                # manifold_mat, decomposition_dict, manifold_mat_inv =\
                            # self.calculate_heat_kernel(Lap_mat, use_eigsh=use_eigsh, return_inverse=return_inverse) 

                manifold_mat, decomposition_dict, manifold_mat_inv = \
                                    calculate_heat_kernel(Lap_mat, lbo_temperature=lbo_temperature, 
                                                          manifold_kernel_amplitude=manifold_kernel_amplitude, 
                                                          use_eigsh=use_eigsh, return_inverse=return_inverse)
            except Exception as e:
                raise type(e)("Calculating LBO raises this:" + str(e))

        elif 'laplacian' in manifold_kernel_type.lower():
            try:
                # manifold_mat, decomposition_dict, manifold_mat_inv = \
                #             self.calculate_L_p_kernel(Lap_mat, use_eigsh=use_eigsh, return_inverse=return_inverse)
                manifold_mat, decomposition_dict, manifold_mat_inv = \
                calculate_L_p_kernel(Lap_mat, manifold_kernel_noise=manifold_kernel_noise, 
                                     manifold_kernel_power=manifold_kernel_power, 
                                     manifold_kernel_amplitude=manifold_kernel_amplitude, 
                                     use_eigsh=use_eigsh, return_inverse=return_inverse)                    

            except Exception as e:
                raise type(e)("Calculating L^p raises this:" + str(e))

    except Exception as e:
        raise type(e)(str(e)+'FUCKKK!!!! 56' )
    return manifold_mat, Lap_mat, decomposition_dict, manifold_mat_inv

@tf.function
def f_k_norm_func(alpha, kernel_mat):
    alpha_kernel = tf.linalg.matmul(alpha, kernel_mat, transpose_a=True)
    return tf.linalg.matmul(alpha_kernel, alpha) / 2. #+ tf.linalg.logdet(kernel_mat) / 2. +  
    # tf.cast(tf.math.log(tf.cast(2 * np.pi, tf.float64)) *
    # tf.cast(tf.shape(self.sampled_gp_input_features[0]), tf.float64) / 2., tf.float64)
    # return tf.linalg.matmul(alpha_kernel, alpha) / 2. + tf.linalg.logdet(kernel_mat + 
    #                 tf.linalg.diag(tf.experimental.numpy.ravel(tf.abs(tf.random.normal(shape=[tf.shape(alpha)[0]], 
    #                 stddev=tf.math.reduce_std(kernel_mat)*1e-2, dtype=tf.float64))))) / 2. + 
    #                 tf.cast(tf.math.log(tf.cast(2 * np.pi, tf.float64)) * 
    #                         tf.cast(tf.shape(self.sampled_gp_input_features[0]), tf.float64) / 2., tf.float64)

@tf.function
def manifold_norm_func(linear_mat_mul, manifold_mat): 
    # return tf.linalg.matmul(tf.linalg.matmul(linear_mat_mul, manifold_mat, transpose_a=True),
    # linear_mat_mul)/2.  - (tf.math.reduce_sum(self.D_vec)) / 2.

    return tf.linalg.matmul(tf.linalg.matmul(linear_mat_mul, manifold_mat, transpose_a=True), linear_mat_mul)/2.
                # - tf.linalg.logdet(manifold_mat + 
                #                    tf.linalg.diag(tf.experimental.numpy.ravel(tf.abs(tf.random.normal(shape=
                #                     [tf.shape(alpha)[0]], stddev=1e-8, dtype=tf.float64))))) / 2. + 
                #                     tf.cast(tf.math.log(tf.cast(2 * np.pi, tf.float64)) * 
                #                     tf.cast(tf.shape(self.sampled_gp_input_features[0]), tf.float64) / 2.,
                #                     tf.float64)

@tf.function
def linear_out_func(linear_mat_mul, beta):
    return tf.add(linear_mat_mul, beta)  

@tf.function
def linear_mat_mul_func(kernel_mat, alpha): 
    return tf.linalg.matmul(kernel_mat, alpha)

class GPLayer(tf.keras.layers.Layer):
    """The idea of handling sparse tensorf is taken from 
    https://medium.com/dailymotion/how-to-design-deep-learning-models-with-sparse-inputs-in-tensorflow-keras-fd5e754abec1
    lot of thanks goes to Sharon!! """
    
    def check_input_float64(self, attrs):
        for attr in attrs:
            attr_val = getattr(self, attr)
            if 'ndarray' in type(attr_val).__name__:
                setattr(self, attr,  attr_val.astype(np.float64))
            elif attr_val is not None:
                if 'float' in type(attr_val).__name__.lower():
                    setattr(self, attr, np.float64(attr_val))

    @property            # first decorate the getter method
    def kernel_mat(self): # This getter method name is *the* name
        return self._kernel_mat.numpy()

    @property            # first decorate the getter method
    def kernel(self): # This getter method name is *the* name
        return self._kernel
    
    
    @property            # first decorate the getter method
    def manifold_mat(self): # This getter method name is *the* name
        if hasattr(self._manifold_mat, 'numpy'):
            return self._manifold_mat.numpy()
        else:
            return self._manifold_mat

    @property            # first decorate the getter method
    def order(self): # This getter method name is *the* name
        return self._order
        
    
    def __init__(self, reg_coeff= None, reg_penalty=None, layer_name=None,
                 sampled_input_features=None, input_type=None, parent=None, 
                unlabeled_data_inc_type=None,

                 kernel=None, gp_kernel_type=None, gp_kernel_lengthscale=None, gp_kernel_lengthscale_trainable=None, 
                 gp_kernel_amplitude=None, gp_kernel_params_pack=None, noise_sigma=None, 
                 
                 manifold_kernel_lengthscale=None, manifold_kernel_lengthscale_trainable=None, 
                 manifold_kernel_amplitude=None, manifold_kernel_amplitude_trainable=None, 
                 manifold_neighbor_mode=None, lbo_temperature=None, 
                 manifold_kernel_noise=None, manifold_kernel_power=None, manifold_kernel_normed=None, 
                 manifold_kernel_k=None,  manifold_kernel_type=None,
                 
                 do_I_plus_KM_inv=True, use_eigsh=False, I_plus_KM_inv_M_calc_method=None, 
                 I_plus_KM_inv_M_using_factorization=None, I_plus_KM_inv_M_using_expm_acting=None, 
                 I_plus_KM_inv_M_using_eigsh=None,  invert_manifold_mat_explicitly=None,
                 
                 alpha=None, beta=None, name='gplayer', training_size=None):
        super().__init__(name=name)
        try:
            self._kernel = kernel
            self._unlabeled_data_inc_type = unlabeled_data_inc_type
            self._gp_kernel_type = gp_kernel_type
            self._gp_kernel_lengthscale = gp_kernel_lengthscale
            self._gp_kernel_lengthscale_trainable = gp_kernel_lengthscale_trainable
            self._gp_kernel_amplitude = gp_kernel_amplitude
            self._noise_sigma = noise_sigma
            
            self._manifold_kernel_noise = manifold_kernel_noise
            self._manifold_kernel_power = manifold_kernel_power
            self._manifold_neighbor_mode = manifold_neighbor_mode
            self._manifold_kernel_type = manifold_kernel_type
            self._manifold_kernel_lengthscale = manifold_kernel_lengthscale
            self._manifold_kernel_lengthscale_trainable = manifold_kernel_lengthscale_trainable
            # self.check_manifold_kernel_lengthscale_type()
            self._manifold_kernel_amplitude = manifold_kernel_amplitude
            self._manifold_kernel_amplitude_trainable = manifold_kernel_amplitude_trainable
            self._manifold_kernel_normed = manifold_kernel_normed
            self._manifold_kernel_k = manifold_kernel_k            
            self._lbo_temperature = lbo_temperature
            

            
            self._do_I_plus_KM_inv = do_I_plus_KM_inv
            self._invert_manifold_mat_explicitly = invert_manifold_mat_explicitly
            self._I_plus_KM_inv_M_calc_method = I_plus_KM_inv_M_calc_method
            self._I_plus_KM_inv_M_using_factorization = I_plus_KM_inv_M_using_factorization
            self._I_plus_KM_inv_M_using_expm_acting = I_plus_KM_inv_M_using_expm_acting
            self._I_plus_KM_inv_M_using_eigsh = I_plus_KM_inv_M_using_eigsh        
            
            
            self._training_size = training_size

            self._alpha = alpha
            self._beta = beta
            
            # # setting the input type where setting it as index will make the kernel calculatiuons faster
            # # and feature input_type will need kernel calculation every time
            # if input_type in ['index', 'feature']:
            #     pass
            # elif input_type is None:
            #     input_type = 'index'
            # else:
            #     raise NotImplementedError("input_type is not set")
            # self.input_type = input_type

            # if sampled_input_features is None:
            #     raise NotImplementedError("sampled_input_features is not set")
                
                
            self._sampled_input_features = sampled_input_features 
            self._parent = parent
        
            self._layer_name = layer_name
            self._reg_penalty =  reg_penalty 
            self._reg_coeff = reg_coeff

            self.set_relevant_input_features()
            self.set_kernel_dim()
            self.check_input_float64(attrs=['_gp_kernel_lengthscale', 
                    '_gp_kernel_amplitude', 
                    '_noise_sigma', 
                    '_manifold_kernel_noise', 
                    '_manifold_kernel_power', 
                    '_manifold_kernel_lengthscale', 
                    '_manifold_kernel_amplitude', 
                    '_lbo_temperature', 
                    '_alpha', '_beta', 
                    '_reg_coeff'])
            
            
        except Exception as e:
            if hasattr(e, 'message'):
                raise type(e)(e.message+'Squaring W is causing error in GPLayer init' )
            else:
                raise type(e)(str(e)+'Squaring W is causing error in GPLayer init' )
            
    # def check_if_one_dim_var(self):
    #     if self._beta is not None and tf.rank(self._beta) < 2:
    #         raise NotImplementedError("self._beta initial value has rank" +\
    #                                   str(tf.rank(self._beta)) + " which is \
    #                                   smaller than 2. all variables need to have rank 2")
            
        
    # def check_manifold_kernel_lengthscale_type(self):
    #     if hasattr(self.manifold_kernel_lengthscale, 'numpy'):
    #         self.manifold_kernel_lengthscale = self.manifold_kernel_lengthscale.numpy()

                        
    
    def set_relevant_input_features(self):
        sampled_input_features = self._sampled_input_features['sig_input']
        if tf.rank(sampled_input_features) == 1:
            sampled_input_features = tf.expand_dims(sampled_input_features, axis=0)
        self._sampled_gp_input_features = sampled_input_features
        
    def set_kernel_dim(self):
        self._kernel_dim = tf.keras.backend.shape(self._sampled_gp_input_features)
        
        
    

    
    

    

        

    

    
    def create_gp_kernel_func(self):
        if self._gp_kernel_type == 'linear':
            kernel = tfp.math.psd_kernels.Linear(bias_variance=
                                                 tf.cast(tf.squeeze(self._gp_kernel_lengthscale), tf.float64), 
                                                 slope_variance=tf.cast(self._gp_kernel_amplitude, tf.float64), 
                                                 shift=None, feature_ndims=1, validate_args=False, name='Linear')
            print ("WE GOT IN LINEAR BABY!!!")
            kernel_func = lambda x, y: kernel.matrix(x, y)

        elif self._gp_kernel_type == 'rbf':
            kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=
                                    tf.cast(self._gp_kernel_amplitude, tf.float64),length_scale=
                                    tf.cast(tf.squeeze(self._gp_kernel_lengthscale),tf.float64), 
                                    feature_ndims=1, validate_args=True, name='ExponentiatedQuadratic')
            
            print ("WE GOT IN RBF BABY!!!")
            kernel_func = lambda x, y: kernel.matrix(x, y)
        elif self._gp_kernel_type == 'laplacian':
            raise NotImplementedError("TODO: Laplacian kernel is not implemented yet")
        else:
            raise NotImplementedError("print self._kernel_type is not assigned")
                    
            
        return kernel_func
    

    
    def calculate_I_plus_MK_inv_M(self, Lap_mat=None, kernel_mat=None, 
                                  manifold_mat=None, manifold_mat_inv=None, with_expm_acting=False, 
                                  I_plus_MK_inverter=None): 
        
        from scipy.sparse.linalg import expm_multiply
        
        if with_expm_acting:
            if hasattr(kernel_mat, 'numpy'):
                kernel_mat = kernel_mat.numpy()
            MK =  expm_multiply(-Lap_mat * self._lbo_temperature, kernel_mat) * self._manifold_kernel_amplitude ** 2
        else:
            MK =  manifold_mat @ kernel_mat

        I_plus_MK = np.eye(len(kernel_mat)) + MK

        if I_plus_MK_inverter == 'qr':
            counter = 0
            while True:
                eps_diag = np.diag(np.ones(manifold_mat.shape[0], dtype=np.float64) * DELTA)
                biased_I_plus_MK = I_plus_MK + (counter > 0) * eps_diag * 10 ** counter
                try:
                    if hasattr(biased_I_plus_MK, 'todense'):
                        biased_I_plus_MK = biased_I_plus_MK.todense()
                    I_plus_MK_Q, I_plus_MK_R = qr(biased_I_plus_MK)
                except LinAlgError as e:
                    counter += 1
                    logging = tf.get_logger()
                    logging.warning("counter increased to " + str(counter) + " for I_plus_MK QR decomopsition")
                    continue

                I_plus_MK_R_inv = dtrtri(I_plus_MK_R)[0]
                break
            I_plus_MK_inv = I_plus_MK_R_inv @ I_plus_MK_Q.T
        elif I_plus_MK_inverter == 'cholesky':
            # NOT THAT THIS DOES MAKE SENSE SINCE I + MK IS NOT SYMMETRIC 
            #(BUT IT ACTUALLY HAS POSITIVE AND 
            # REAL EIGENVALUES!!)
            counter = 0
            while True:
                eps_diag = np.diag(np.ones(manifold_mat.shape[0], dtype=np.float64) * DELTA)
                biased_I_plus_MK = I_plus_MK + (counter > 0) * eps_diag * 10 ** counter
                I_plus_MK_cholesky = tf.linalg.cholesky(biased_I_plus_MK).numpy().T
                if np.isnan(I_plus_MK_cholesky).any():
                    counter += 1
                    logging = tf.get_logger()
                    logging.warning("counter increased to " + str(counter) + " for I_plus_MK cholesky")
                    continue
                I_plus_MK_cholesky_inv = dtrtri(I_plus_MK_cholesky)[0]
                I_plus_MK_inv = I_plus_MK_cholesky_inv @ I_plus_MK_cholesky_inv.T

                break
        else:
            raise NotImplementedError("For method invert_I_plus_MK, I_plus_MK_inverter needs to be cholesky or qr")
        if with_expm_acting: 
            I_plus_MK_inv_times_M =\
            expm_multiply(-Lap_mat * self._lbo_temperature, I_plus_MK_inv.T).T * self._manifold_kernel_amplitude ** 2                    
        else:
            I_plus_MK_inv_times_M = I_plus_MK_inv @ manifold_mat

        M_inv_plus_K_inv = I_plus_MK_inv_times_M 
            
        return M_inv_plus_K_inv



    def calculate_M_inv_plus_K_inv(self, kernel_mat=None, manifold_mat=None, manifold_mat_inv=None):
        if not manifold_mat_inv:
            counter = 0
            try:    
                manifold_mat_inv = invert_mat_with_cholesky(manifold_mat)
            except LinAlgError as e:
                counter += 1
                logging = tf.get_logger()
                logging.warning("counter increased to " + str(counter) + " for manifold_mat inversion")
                        
        M_inv_plus_K = kernel_mat + manifold_mat_inv
        
        try:    
            M_inv_plus_K_inv = invert_mat_with_cholesky(M_inv_plus_K)
        except LinAlgError as e:
            logging = tf.get_logger()
            logging.warning("counter increased to " + str(counter) + " for M_inv_plus_K inversion")
            
        return M_inv_plus_K_inv
    
    def calculate_mixed_kernel(self, manifold_mat=None, manifold_mat_inv=None, 
                               Lap_mat=None, kernel_mat=None, decomposition_dict=None,
                                method='invert_I_plus_MK', with_expm_acting=None,
                               I_plus_MK_inverter = 'qr'):
        """ here we creat \tilde_K according to eq. (5) in https://www.ijcai.org/Proceedings/07/Papers/171.pdf
         as opposed just regularization offered in manifold regularization paper which the RKHS where f comes
         from has kernel k and not \tilde_k (which has L added to it)
         method: (string) either 'invert_M_first' or 'invert_I_plus_MK'         
         """
        
        # if invert_M_using_expm and 'laplacian' in self.manifold_kernel_type.lower():
        #     raise NotImplementedError("invert_M_using_expm cannot be true while the L^p is being used for SSL")
        
        # if with_expm_acting and 'laplacian' in self.manifold_kernel_type.lower():
        #     raise NotImplementedError("with_expm_acting cannot be true while the L^p is being used for SSL")
        
        
        if method == 'invert_I_plus_MK':
            try:
                M_inv_plus_K_inv = self.calculate_I_plus_MK_inv_M(kernel_mat=kernel_mat, 
                                                                  manifold_mat=manifold_mat, 
                                                                  manifold_mat_inv=manifold_mat_inv, 
                                                                  with_expm_acting=with_expm_acting, 
                                                                  I_plus_MK_inverter=I_plus_MK_inverter, 
                                                                  Lap_mat=Lap_mat)
            except Exception as e:
                raise type(e)("invert_I_plus_MK method is causing this failure:" +str(e))
            
        elif method == 'invert_M_first':
            try:
                M_inv_plus_K_inv = self.calculate_M_inv_plus_K_inv(kernel_mat=kernel_mat, 
                                                                   manifold_mat=manifold_mat,
                                                                   manifold_mat_inv=manifold_mat_inv)
            except Exception as e:
                if 'value' in type(e).__name__.lower():
                    try:
                        M_inv_plus_K_inv = self.calculate_M_inv_plus_K_inv(kernel_mat=kernel_mat, 
                                                                       manifold_mat=manifold_mat, manifold_mat_inv=False)
                    except Exception as e:
                        raise type(e)("M_inv_plus_K_inv method is causing this failure\
                                      even when manifold_mat_inv is explicitly set to False:" +str(e))
                else:
                    raise type(e)("M_inv_plus_K_inv method is causing this failureand \
                                        it is not a ValueError exception:" + str(e))                
        
        
        def mixed_kernel_func(x, z):
            k_x = self._original_kernel_func(self._sampled_gp_input_features, x)
            k_z = self._original_kernel_func(self._sampled_gp_input_features, z)
            return self._original_kernel_func(x, z) -\
                                    tf.linalg.matmul(tf.linalg.matmul(tf.transpose(k_x), M_inv_plus_K_inv), k_z)
        return mixed_kernel_func
    
    def calculate_main_kernels(self, features, use_eigsh=False, return_manifold_mat_inv=False):
#        if use_eigsh:
            # raise NotImplementedError("eigsh is implemented with a big mistake! needs correction: SVD cannot inverse that easily! the inverse of UDU^T is not U^T *1/D* U unfortunately!") 
        #######################################################
        ### BEGIN: CALCULATING THE LAPLACIAN CASE AND THE HEAT KERNEL
        #######################################################
        self._kernel_mat_dim = len(self._sampled_gp_input_features)
        if self._manifold_kernel_type is not None:
            # W = selfW(features)
            W = build_W(features, k_neighbours=self._manifold_kernel_k, lengthscale=self._manifold_kernel_lengthscale, connectivity=self._manifold_neighbor_mode)
        
        #######################################################
        ### END: CALCULATING THE LAPLACIAN CASE AND THE HEAT KERNEL
        #######################################################
        original_kernel_func = self.create_gp_kernel_func()
        kernel_mat = original_kernel_func(features, features).numpy()
        # manifold_mat, Lap_mat, decomposition_dict, manifold_mat_inv =\
                            # self.build_manifold_mat(W, use_eigsh=use_eigsh, return_inverse=return_manifold_mat_inv)
        manifold_mat, Lap_mat, decomposition_dict, manifold_mat_inv =\
                            build_manifold_mat(W, manifold_kernel_noise=self._manifold_kernel_noise, 
                                                    manifold_kernel_type=self._manifold_kernel_type, 
                                                    manifold_kernel_power=self._manifold_kernel_power, 
                                                    manifold_kernel_amplitude=self._manifold_kernel_amplitude, 
                                                    lbo_temperature=self._lbo_temperature, 
                                                    manifold_kernel_normed=self._manifold_kernel_normed, 
                                                    use_eigsh=use_eigsh, return_inverse=return_manifold_mat_inv)
                                                    
        return kernel_mat, manifold_mat, manifold_mat_inv, Lap_mat, decomposition_dict, original_kernel_func


    

    def build(self, input_shape):
#         @tf.function
        input_shape = input_shape['sig_input'].as_list()[0]
        print ("Setting the kernel...") 
        
        try:
            return_manifold_mat_inv = self._invert_manifold_mat_explicitly
            if 'lbo' in self._manifold_kernel_type.lower():
                method = self._I_plus_KM_inv_M_calc_method #  'invert_M_first' or invert_I_plus_MK
                I_plus_MK_inverter = self._I_plus_KM_inv_M_using_factorization
                with_expm_acting= self._I_plus_KM_inv_M_using_expm_acting
                use_eigsh = self._I_plus_KM_inv_M_using_eigsh
            elif 'laplacian' in self._manifold_kernel_type.lower():
                method = self._I_plus_KM_inv_M_calc_method # or 'invert_M_first'
                I_plus_MK_inverter = self._I_plus_KM_inv_M_using_factorization
                with_expm_acting=self._I_plus_KM_inv_M_using_expm_acting
                use_eigsh = self._I_plus_KM_inv_M_using_eigsh
            else:
                raise NotImplementedError("No other SSL method available")
        
            original_kernel_mat, self._manifold_mat, self._manifold_mat_inv,\
            self._Lap_mat, decomposition_dict, self._original_kernel_func =\
                                        self.calculate_main_kernels(self._sampled_gp_input_features, 
                                                                    use_eigsh=use_eigsh,
                                                                    return_manifold_mat_inv=return_manifold_mat_inv)
        except Exception as e:
            raise type(e)("calculating main kernels is raising this error:" + str(e))

        try:
            self._mixed_kernel_func = self.calculate_mixed_kernel(manifold_mat=self._manifold_mat, 
                                                                 manifold_mat_inv=self._manifold_mat_inv,
                                                                 kernel_mat=original_kernel_mat, 
                                                                 Lap_mat=self._Lap_mat, 
                                                                 decomposition_dict=decomposition_dict, 
                                                                 I_plus_MK_inverter=I_plus_MK_inverter, 
                                                                 with_expm_acting=with_expm_acting, 
                                                                 method=method)    
            
        except Exception as e:
            raise type(e)("calculating mixed kernel is raising error:" + \
                          str(e) +"when the I+MK inversion is calculated using "\
                          +str(self._I_plus_KM_inv_M_calc_method)+" method")
            
        counter = 0
        ####################################################
        # METHOD THROUGH CALCULATING NEW KERNEL AND L_inv_plus_K_D_D 
        ####################################################
        # with_cholesky = True 
        # with_expm_acting = True # ONLY APPLICABLE WHEN (I + MK)^{-1}@M is BEING USED 
        # invert_with_expm = False # ONLY APPLICABLE WHEN (M^{-1} + K)^{-1} IS BEING USED WHERE M^{-1}= exp(-tL)^{-1}=exp(tL)
        # use_qr = True
                        
        if self._do_I_plus_KM_inv:
            self._kernel = self._mixed_kernel_func
        else:
            self._kernel = self._original_kernel_func

        # UPDATING KERNEL_MAT BECAUSE THE NEW KERNEL MIGHT BE THE MIXED ONE AS OPPOSED TO THE RBF ONE
        self._kernel_mat = self._kernel(self._sampled_gp_input_features, self._sampled_gp_input_features)
            
        # BUILDING KERAS/TF VARIABLES
        if self._alpha is None:
            alpha_initializer=lambda x, dtype:\
                            tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=.1)\
                                                                (shape=[self._kernel_mat_dim, 1], dtype=tf.float64)
        else:
            alpha_initializer=lambda x, dtype: tf.constant(self._alpha, dtype=tf.float64)
            
        self._alpha = self.add_weight("alpha", shape=[self._kernel_mat_dim, 1], regularizer=None\
                                     if self._reg_penalty is None\
                                     else self._reg_penalty(l=self._reg_coeff),\
                                     trainable=True, initializer=alpha_initializer, dtype=tf.float64)
        
        if self._beta is None:
            beta_initializer=lambda x, dtype:tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=.1)\
                                                                                    (shape=[], dtype=tf.float64)
        else:            
            beta_initializer=lambda x, dtype:tf.constant(self._beta, dtype=tf.float64)
        
        self._beta = self.add_weight("beta",shape=[], trainable=True, initializer=beta_initializer, dtype=tf.float64)

        # if self.input_type == 'feature':
        #     pass
        # elif self.input_type == 'index':
        #     pass
        # else:
        #     raise NotImplementedError
            
            
        
#         self.KLK = tf.linalg.matmul(self.kernel_mat, tf.linalg.matmul(self.Lap_mat, self.kernel_mat))
                    
        if self._layer_name is None:
            self._layer_name = 'sigmoid_layer'
        
        self._order = [0] * len(self.variables)
        for i, variable in enumerate(self.variables):
            if 'alpha' in variable.name:
                self._order[0] = i
            elif 'beta' in variable.name:
                self._order[1] = i
            else:
                raise NotImplementedError("THE VARIABLE DOES NOT BELONG TO GP LAYER. WTF??")


    def call(self, input_tensor_dict=None, training=False):
        try:
            input_idx = input_tensor_dict['idx']
        except KeyError as e:
            logging = tf.get_logger()
            logging.warning("masker_cast is not set. defaulting to set the mask to be all indices...")            
            input_idx = input_tensor_dict['idx'] = tf.cast(np.arange(len(input_tensor_dict['psych_input'])), tf.int32)
        masker_cast = tf.cast(input_idx, tf.int32)
        
        
        layer_input = input_tensor_dict['sig_input']

        if tf.rank(layer_input) == 1:
            layer_input = tf.expand_dims(layer_input, axis=0)

        
        
#         @tf.function
#         def prepare_masker(x):
#             return tf.cast(tf.reshape(x, (-1, 1)), tf.int32)
    
    #     @tf.function
    #     def mix_norm_func(self, noise_sigma, alpha, manifold_mat_U, manifold_mat_D_vec):
    #         noise_square = tf.math.square(noise_sigma)
    #         alpha_U = tf.linalg.matmul(tf.transpose(alpha), manifold_mat_U)
    #         diag_sum = tf.linalg.diag(tf.math.reciprocal(tf.math.add(noise_sigma, manifold_mat_D_vec)))
    #         return tf.math.multiply(noise_square, tf.linalg.matmul(alpha_U, tf.linalg.matmul(diag_sum, tf.transpose(alpha_U))))

    #     @tf.function
    #     def alpha_norm_func(self, alpha, noise_sigma):
    #         return tf.linalg.matmul(tf.transpose(alpha), alpha) * noise_sigma


        
        #         if type(Lap_mat).__name__ in ['csr_matrix', 'coo_matrix']:
#             self.Lap_mat.assign(np.asarray(Lap_mat.todense()))
#         else:
            
#         if type(exp_Lap_mat).__name__ in ['csr_matrix', 'coo_matrix']:
#             self.exp_Lap_mat.assign(np.asarray(exp_Lap_mat.todense()))
#         else:
#         layer_input = input_tensor_dict['sig_input']

#         if tf.rank(layer_input) == 2:
#         for key in input_tensor_dict.keys():
#             input_tensor_dict[key] = tf.cond(tf.rank(input_tensor_dict['sig_input']) == 2,lambda: tf.expand_dims(input_tensor_dict[key], axis=0),  lambda: input_tensor_dict[key])
        
        
        # tf.print ("THE TYPE:", (layer_input))
        # logging = tf.get_logger()
        # logging.warning("layer_input dtype is:" + str(layer_input.dtype))
        # layer_input = tf.cast(layer_input, tf.float64)
        # linear_out = tf.add(tf.matmul(layer_input, self._alpha), self._beta)
        # if tf.dtype(layer_input) :
        #     logging = tf.get_logger()
        #     logging.warning(str(e) + "... so casting layer_input to tf.float64")
        #     layer_input = tf.cast(layer_input, tf.float64)
        #     linear_out = tf.add(tf.matmul(layer_input, self._alpha), self._beta)

        batch_size = tf.keras.backend.shape(layer_input)[0]
#         masking_vector, layer_input = mixed_input_tensor
        
#         diag_mat = tf.linalg.diag(tf.squeeze(masking_vector))
#         diag_mat_sparse = tf.sparse.from_dense(diag_mat)
#         GP_sub_tensor = tf.linalg.matmul(self.kernel_mat, diag_mat, b_is_sparse=True)
#         GP_sub_tensor = tf.sparse.sparse_dense_matmul(diag_mat_sparse, GP_sub_tensor)
#         if not training:
#             input_type = 'feature'
#         else:
#             input_type = self.input_type
#         if training:
#             kernel_mat = self.kernel_mat
#             Lap_mat = self.Lap_mat
#         else:
#             kernel_mat, Lap_mat = self.calculate_kernels(layer_input)
#             kernel_mat = tf.convert_to_tensor(kernel_mat)
#         Lap_mat = tf.convert_to_tensor(self.Lap_mat)
#         else:
#             raise NotImplementedError("self.manifold_kernel_type is not right!")
#         if tf.rank(masker_cast) == 1:
#             masker_cast = tf.expand_dims(masker_cast, axis=0)
        if training:    
            
# BEGIN: THIS BELOW PART WORKS WITH BATCHING TOO BUT IT MIGHT NOT BE COMPLETELY CORRECT. IT SEEMS CHALLENGING
# TO MAKE DUAL FORM FOR KERNEL METHODS WORK WITH BATCH GRADIENT DESCENT SO I WILL JUST WRITE THE CODE FOR FULL # DATASET BELOW AFTER THE COMMENTED SECTION
            try:
                try:
                    if hasattr(self._manifold_mat, 'todense'):
                        dense_manifold_mat = self._manifold_mat.todense()
                    else:
                        dense_manifold_mat = self._manifold_mat
                    active_alpha = tf.gather(self._alpha, masker_cast, axis=0)
                    manifold_mat_active_rows = tf.gather(dense_manifold_mat, masker_cast, axis=0)
                    active_manifold_mat = tf.gather(manifold_mat_active_rows, masker_cast, axis=1)

                    kernel_active_rows = tf.gather(self._kernel_mat, masker_cast, axis=0)
                    active_kernel = tf.gather(kernel_active_rows, masker_cast, axis=1)


#                     tf.print('input data shape for layer_input:', layer_input.shape, "masker_cast:", masker_cast, 'manifold_mat.shape:', manifold_mat.shape, ' manifold_mat_active_rows.shape:', manifold_mat_active_rows.shape, 'active_manifold_mat.shape:', active_manifold_mat.shape, output_stream=sys.stderr)

#                     tf.print('active_alpha.shape:', active_alpha.shape,  'kernel_mat.shape:', self.kernel_mat.shape, 'kernel_active_rows.shape:', kernel_active_rows.shape, 'active_kernel.shape:', active_kernel.shape, 'inner_matmul_shape:', tf.linalg.matmul(active_alpha, active_kernel, transpose_a=True).shape, output_stream=sys.stderr)

            
                    linear_mat_mul = tf.linalg.matmul(kernel_active_rows, active_alpha, transpose_a=True)
        #             linear_out = linear_mat_mul + self._beta
                    f_k_norm = f_k_norm_func(active_alpha, active_kernel)
                    if self._do_I_plus_KM_inv:
                        manifold_norm = tf.zeros_like(f_k_norm)
                    else:
                        manifold_norm = manifold_norm_func(linear_mat_mul, self._manifold_mat)
            
    #                 tf.print('f_k_norm.shape:', f_k_norm.shape, output_stream=sys.stderr)

        #             manifold_mat_active_rows = tf.squeeze(tf.gather(manifold_mat_mat, masker_cast, axis=0), axis=-2)
        #             active_manifold_mat = tf.squeeze(tf.gather(tf.transpose(manifold_mat_active_rows), masker_cast, axis=0, batch_dims=1), axis=-2)

        #             f_k_norm = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(active_alpha), active_kernel), active_alpha) + tf.linalg.matmul(tf.transpose(active_alpha), active_alpha) * self.noise_sigma
            #             Lap_norm = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(linear_mat_mul), active_Lap_mat), linear_mat_mul)
            #             tf.print( "TRACE IS:", tf.linalg.trace(Lap_norm))
            # THIS BELOW PART WORKS WITH BATCHING TOO BUT IT MIGHT NOT BE COMPLETELY CORRECT. IT SEEMS CHALLENGING
            # TO MAKE DUAL FORM FOR KERNEL METHODS WORK WITH BATCH GRADIENT DESCENT SO I WILL JUST WRITE THE CODE FOR FULL # DATASET BELOW : END
        #             else:
        #                 linear_mat_mul = linear_mat_mul_func(self.kernel_mat, alpha)
        #     #             self.normalization_std = tf.math.reduce_std(linear_mat_mul)
        #     #             self.normalization_mean = tf.math.reduce_mean(linear_mat_mul)
        #     #             linear_mat_mul = tf.divide(linear_mat_mul - self.normalization_mean, self.normalization_std)
        #                 f_k_norm = f_k_norm_func(self._alpha, self.kernel_mat)
        # #                 Lap_mat = self.Lap_mat
        #                 exp_Lap_mat = self.exp_Lap_mat

            #             mix_norm = self.mix_norm_func(self.noise_sigma, self._alpha, self.manifold_mat_U, self.manifold_mat_D_vec) - self.alpha_norm_func(self._alpha, self.noise_sigma)
        #             mix_norm = tf.zeros(shape=[batch_size, 1, 1], dtype=tf.float64)
                    # if self.manifold_kernel_type in ['SSL_LBO', 'SSL_laplacian']:
                    #     manifold_norm = manifold_norm_func(linear_mat_mul, self.manifold_mat)
                    # else:
                    #     raise NotImplementedError("self.manifold_kernel_type is not right!")
        #             alpha_norm = tf.zeros(shape=[batch_size, 1, 1], dtype=tf.float64)
        #             alpha_norm = self.alpha_norm_func(self._alpha, self.sigma_noise)

    #                 linear_mat_mul = tf.gather(linear_mat_mul, masker_cast, axis=1)
    #                 tf.print("linear_mat_mul shape:", linear_mat_mul.shape, "and manifold_mat shape:", manifold_mat.shape, 'linear_mat_mul afte gathering shape:', linear_mat_mul.shape)

                except Exception as e:
                    raise type(e)(str(e) + " here is the fucked up error"+ str(active_alpha.shape)+ 'manifold_mat_active_rows.shape')#+'active_kernel.shape:'+str(active_kernel.shape))
            except Exception as e:
                raise type(e)(str(e) + " second layer active_alpha:" + str(active_alpha.shape) + " second layer self.kernel_mat shape:" + str(self._kernel_mat.shape))
            linear_out = linear_out_func(linear_mat_mul, self._beta)
                

        else:
            # if tf.rank(layer_input)==1:
                # layer_input = tf.expand_dims(layer_input, axis=0)
                
#             active_alpha = tf.math.expand_dims(self._alpha, axis=0)
#             active_Lap_mat = tf.math,expand_dims(tf.zeros([batch_size, batch_size], dtype=tf.float64), axis=0)
#             stacked_input_tensor = tf.concat([self.sampled_input_features, layer_input], axis=0)
#             stacked_kernel_active_rows = self.kernel(stacked_input_tensor, self.sampled_input_features)
            kernel_active_rows = self._kernel(layer_input, self._sampled_gp_input_features)
#             tf.print("active rows:", kernel_active_rows, output_stream=sys.stderr)
#             kernel_active_rows = self.kernel.scaled_squared_euclid_dist(layer_input, tf.math,expand_dims(self.sampled_input_features, axis=0))
#             active_kernel = self.kernel_mat

            #APPLYING SCHUR COMPLEMENT
#             A = self.kernel_mat
#             A_inv = self.kernel_mat_inv
#             B = kernel_active_rows
#             C = tf.transpose(kernel_active_rows)
#             D = self.kernel(layer_input, layer_input)
#             S = D - tf.matmul(tf.matmul(tf.matmul(D, C), A_inv), B)
#             S_inv = tf.linalg.inv(S)
#             K_tilde_D_D_first_part_first_col = tf.concat([A_inv + tf.matmul(tf.matmul(tf.matmul(tf.matmul(A_inv, V), S_inv), C), A_inv), -tf.matmul(S_inv, tf.matmul(C, A_inv))], axis=1)
#             K_tilde_D_D_first_part_second_col = tf.concat([-tf.matmul(A_inv, tf.matmul(B, S_inv)), S_inv], axis=1)
#             K_tilde_D_D_first_part = tf.concat([K_tilde_D_D_first_part_first_col, K_tilde_D_D_first_part_second_col], axis=0)
            
            linear_mat_mul = tf.linalg.matmul(kernel_active_rows, self._alpha)
#             linear_mat_mul = tf.divide(linear_mat_mul - self.normalization_mean, self.normalization_std)
            linear_out = linear_mat_mul + self._beta
#             f_k_norm = tf.squeeze(tf.linalg.matmul(tf.transpose(linear_mat_mul), linear_mat_mul) + tf.linalg.matmul(tf.transpose(active_alpha, perm=[0, 2, 1]), active_alpha) * self.noise_sigma, axis=-1)
            f_k_norm = tf.zeros(shape=[1, 1], dtype=tf.float64)
            manifold_norm = tf.zeros(shape=[1, 1], dtype=tf.float64)


        output =  [self._gp_kernel_lengthscale, self._manifold_kernel_lengthscale, linear_out, self._alpha, self._beta, manifold_norm, f_k_norm]  
#         else:
#             raise NotImplementedError
            

        return output
#     def compute_output_shape(self,inputShape):
#         return [(None,1),(None,1)]

#     def compute_output_shape(self, input_shape):
#         return [(None,1), (None,1), (None,1), (None,1), (None,1), (None,1), (None,1)]
    
    # def get_output_shape(self, input_shape):
    #     input_shape = input_shape.get_shape().as_list()
    #     return input_shape[0], self.num_outputs
    
#     def predict_proba(self, X, **kwargs):
#         gamma_, lambda_, linear_out = self.call(X)
#         return gamma_ + (1 - gamma_ - lambda_) * tf.sigmoid(linear_out)
    
#     def predict(self, X, **kwargs):
#         return self.predict_proba(X) > 0.5
        
