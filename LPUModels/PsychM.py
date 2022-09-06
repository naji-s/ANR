# import tensorflow as tf
import logging
import os
from numpy.random import multivariate_normal
DELTA = 1e-4
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError, matrix_power
from utils.func_lib import find_optimal_threshold
from time import time
def set_tf_loglevel(level):
    if level>= logging.FATAL:
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
EPSILON = 1e-16

tf.keras.backend.set_floatx('float64')
from sklearn.base import BaseEstimator
from numpy import mod, zeros, hstack, ones, zeros_like
from numpy import diag
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)
import sys
sys.path.append('/home/scratch/nshajari/psych_model/')
sys.path.append('/home/scratch/nshajari/psych_model/utils/')
from scipy.linalg import pinvh as scipy_pinvh
from scipy import sparse
import dill as pickle
from scipy.stats import truncnorm
import numpy as np
import tensorflow_probability as tfp
from gc import collect as gc_collect
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
# from math_utils import tf.math.sigmoid, modified_rbf_kernel, modified_linear_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import roc_curve
# from sklearn.metrics.pairwise import rbf_kernel as modified_rbf_kernel
from scipy.special import expit 
import sys
sys.path.append('/home/scratch/nshajari/psych_model/puLearning')
sys.path.append('/home/scratch/nshajari/psych_model/utils')
sys.path.append('/home/scratch/nshajari/psych_model/LPUModels')
# sys.path.append('/home/scratch/nshajari/psych_model/LPUModels')
# from scipy.special import tf.math.sigmoid
# from PropensityEM import PropensityEM
# from naive_clf import NaiveLePU
from PUAdapterTF import PUAdapterTF
from PsychMKeras import PsychMKeras

from PythonPsychM import PythonPsychM

from sklearn.model_selection import StratifiedKFold
from dask_ml.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# from math_utils import tf.math.sigmoid

from FunctionFactory import FunctionFactory,assign_new_model_parameters

from func_lib import convert_sparse_matrix_to_sparse_tensor
# from TransformToCOO import TransformToCOO
# from LPUModels.PUAdapterTF import PUAdapterTF
# from KMEModel import KMEModel
from sklearn import metrics
from scorer_library import flexible_scorer
from utils.func_lib import gamma_lambda_to_g_l_prime_transformer, g_l_prime_to_gamma_lambda_transformer
from sklearn.neighbors import kneighbors_graph


# from lap_svm import rbf
# def f_g_prime_init_func(x):
#     return np.float64(x) / 2.
# def f_l_prime_init_func(x):
#     return np.float64(1e-5)
# def warm_f_g_prime_init_func(x):
#     return x / 2.

# class LossGradientNorm(tf.keras.metrics.Metric):
#     def __init__(self, name="lossgradient_norm", model=None, **kwargs):
#         super(LossGradientNorm, self).all(name=name, **kwargs)
#         self.model = model
    
#         self.gradient_norm = self.add_weight(name="lgn", initializer="zeros")

#     def update_state(self, y_true, y_pred, sample_weight=None):
# #         y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
# #         values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
# #         values = tf.cast(values, "float32")
# #         if sample_weight is not None:
# #             sample_weight = tf.cast(sample_weight, "float32")
# #             values = tf.multiply(values, sample_weight)
# #         self.true_positives.assign_add(tf.reduce_sum(values))
#         norm = 0.
        
# #         with tf.GradientTape() as tape:
# #             tape.watch(self.model.keras_model.trainable_variables)
# #             loss = self.model.keras_model.model_loss(y_true, y_pred)
# #         tf.print ("Trainable Variables Are:", self.model.keras_model.trainable_variables)
# #         tf.print ("Loss on the other hand is:", loss)
# #         gradients = tape.gradient(loss, self.model.keras_model.trainable_variables)
# #         tf.print ("Gradients are:", gradients)
#         gradients = tf.keras.backend.gradients(self.model.keras_model.model_loss(y_true, y_pred), self.model.keras_model.trainable_variables)
#         for item in gradients:
#             norm += tf.norm(item) ** 2
#         self.gradient_norm.assign(norm)

#     def result(self):
#         return self.gradient_norm

#     def reset_states(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.gradient_norm.assign(0.0)    

 
        
class PsychM(BaseEstimator): 
    def __init__(self, encoder=None,  metrics=[], child=None, sig_a=None, sig_b=None, psych_alpha=None, psych_beta=None, psych_gamma=None, psych_lambda=None, g_prime=None, l_prime=None, kernel=None, history=None, alternate_history_list=None, alternate_param_reporter_list=None, alternate_descend_epoch_list=None, parameter_reporter=None,
        min_rerun=1,        
        max_rerun=1,        
                 
        is_SPM=False,
        is_fitted=False,
        keras_model_initialized=False,
        sig_reg_coeff=None,
        sig_reg_penalty=tf.keras.regularizers.L2,
        optimization_results=None,
        epochs=5,
        psych_reg_coeff=None,
        psych_reg_penalty=tf.keras.regularizers.L2,
        batch_size=64,
        keras_model_weights=None,
        workers=1,
        fit_counts=0,
        verbose=0,
        loss_type=None,
        use_multiprocessing=True,
        calibrate=False,
        f_g_prime_init=None,
        f_l_prime_init=None,
        alternate_training=False,
        constrained_optimization=None,
        gp_kernel_lengthscale_trainable=None,
        gp_kernel_lengthscale=None,                 
        gp_kernel_amplitude=None,
        fresh_opt_initial_point=None,
        manifold_kernel_lengthscale=None,
        manifold_kernel_lengthscale_trainable=None,
        manifold_kernel_amplitude=None,
        manifold_kernel_amplitude_trainable=None,
        manifold_kernel_k = None,
        noise_sigma=None,
        lr_reduce_min_delta=None,
        end_training_min_delta=None,
        dropout_rate=None,
        max_iter=None,
        manifold_neighbor_mode=None,
        manifold_kernel_type=None,
        gp_kernel_type=None,
        lbo_temperature=None,
        manifold_kernel_noise=None,
        manifold_kernel_power=None,
        manifold_kernel_normed=None,
        warm_f_g_prime_init=None,
        warm_f_l_prime_init=None,
        optimizer_dict=None,
        barrier_initial_C=None,
        warm_cv_params=None,
        manifold_kernel_params_pack=None,
        gp_kernel_params_pack=None,
        sampled_input_features=None, 
        # little_input_dict = None,
                 name='psychm',
                number_of_successful_attempts=3,
        freeze_psychometric_original = False,
                 freeze_psych_alpha=False, 
                 training_size = None,
                 params_sample = None,
                 with_laplace_method=False,
                 ambient_to_intrinsic_amplitude_ratio=None,
                 prior_sample_size=100,
                 invert_manifold_mat_explicitly=None,
                 I_plus_KM_inv_M_calc_method=None,
                 I_plus_KM_inv_M_using_factorization=None,
                 I_plus_KM_inv_M_using_expm_acting=None,
                 I_plus_KM_inv_M_using_eigsh=None,
                 optimal_threshold=None,
                 unlabeled_data_inc_type=None
                 
                ):

        super().__init__()

#         self.as_super = super(PsychM, self)
#         self.as_super.__init__(*args, **kwargs)            
        self.with_laplace_method = with_laplace_method
        self.number_of_successful_attempts = number_of_successful_attempts
        self.metrics = metrics
        self.encoder = encoder
        self.child = child
        self.freeze_psych_alpha = freeze_psych_alpha
        self.g_prime = g_prime
        self.l_prime = l_prime        
        self.sig_a = sig_a
        self.sig_b = sig_b
        self.name = name
        self.max_rerun = max_rerun
        self.min_rerun = min_rerun
        self.is_fitted = is_fitted
        self.psych_alpha = psych_alpha
        self.psych_beta = psych_beta
        self.psych_lambda = psych_lambda
        self.psych_gamma = psych_gamma
        self.kernel = kernel
        self.history = history
        self.parameter_reporter = parameter_reporter
        self.is_SPM = is_SPM
        self.alternate_history_list = alternate_history_list
        self.alternate_param_reporter_list = alternate_param_reporter_list
        self.alternate_descend_epoch_list = alternate_descend_epoch_list
        self.loss_type = loss_type
        self.use_multiprocessing = use_multiprocessing 
        self.keras_model_initialized = keras_model_initialized
        self.calibrate = calibrate
        self.warm_f_g_prime_init = warm_f_g_prime_init
        self.warm_f_l_prime_init = warm_f_l_prime_init
        self.f_g_prime_init = f_g_prime_init
        self.f_l_prime_init = f_l_prime_init
        self.alternate_training = alternate_training
        self.gp_kernel_lengthscale_trainable = gp_kernel_lengthscale_trainable
        self.gp_kernel_amplitude = gp_kernel_amplitude
        self.fresh_opt_initial_point = fresh_opt_initial_point
        self.manifold_kernel_lengthscale_trainable = manifold_kernel_lengthscale_trainable
        self.manifold_kernel_amplitude = manifold_kernel_amplitude
        self.manifold_kernel_amplitude_trainable = manifold_kernel_amplitude_trainable
        self.noise_sigma = noise_sigma
        self.lr_reduce_min_delta = lr_reduce_min_delta
        self.end_training_min_delta = end_training_min_delta
        self.fit_counts = fit_counts
        self.manifold_kernel_k = manifold_kernel_k
        self.dropout_rate = dropout_rate
        self.manifold_neighbor_mode = manifold_neighbor_mode
        self.manifold_kernel_type = manifold_kernel_type
        self.gp_kernel_type = gp_kernel_type
        self.gp_kernel_lengthscale = gp_kernel_lengthscale
        self.gp_kernel_params_pack = gp_kernel_params_pack
        self.lbo_temperature = lbo_temperature
        self.optimization_results = False
        self.manifold_kernel_lengthscale = manifold_kernel_lengthscale
        self.manifold_kernel_params_pack = manifold_kernel_params_pack
        self.manifold_kernel_noise = manifold_kernel_noise
        self.manifold_kernel_power = manifold_kernel_power
        self.manifold_kernel_normed = manifold_kernel_normed
        self.warm_f_g_prime_init = warm_f_g_prime_init
        self.warm_f_l_prime_init = warm_f_l_prime_init
        self.constrained_optimization = constrained_optimization
        self.optimizer_dict = optimizer_dict
        self.warm_cv_params = warm_cv_params
        self.barrier_initial_C = barrier_initial_C 
        self.sig_reg_coeff = sig_reg_coeff
        self.sig_reg_penalty =sig_reg_penalty
        self.psych_reg_coeff = psych_reg_coeff
        self.psych_reg_penalty = psych_reg_penalty
        self.batch_size = batch_size
        self.verbose = verbose
        self.epochs = epochs
        self.sampled_input_features = sampled_input_features
        self.workers = workers
        # self.little_input_dict = little_input_dict
        self.freeze_psychometric_original = freeze_psychometric_original
        self.training_size = training_size
        self.keras_model_weights = keras_model_weights
        self.params_sample = params_sample
        self.ambient_to_intrinsic_amplitude_ratio = ambient_to_intrinsic_amplitude_ratio
        self.prior_sample_size = prior_sample_size
        self.control_amplitudes()
        self.max_iter = max_iter
        self.invert_manifold_mat_explicitly = invert_manifold_mat_explicitly
        self.I_plus_KM_inv_M_calc_method = I_plus_KM_inv_M_calc_method
        self.I_plus_KM_inv_M_using_factorization = I_plus_KM_inv_M_using_factorization
        self.I_plus_KM_inv_M_using_expm_acting = I_plus_KM_inv_M_using_expm_acting
        self.I_plus_KM_inv_M_using_eigsh = I_plus_KM_inv_M_using_eigsh
        self.set_I_plus_KM_inv_M_params()
        self.optimal_threshold = optimal_threshold
        self.unlabeled_data_inc_type = unlabeled_data_inc_type

#         self.import_libraries()
#     def import_libraries(self):

    def set_I_plus_KM_inv_M_params(self):
        pass
        self.invert_manifold_mat_explicitly
        self.I_plus_KM_inv_M_calc_method
        self.I_plus_KM_inv_M_using_factorization
        self.I_plus_KM_inv_M_using_expm_acting
        self.I_plus_KM_inv_M_using_eigsh

    def control_amplitudes(self):
        if self.ambient_to_intrinsic_amplitude_ratio is None:
            return
        else:
            self.manifold_kernel_amplitude = self.gp_kernel_amplitude / self.ambient_to_intrinsic_amplitude_ratio 
                
    def set_params(self, **parameters):
        try:
            for parameter, value in parameters.items():                    
                setattr(self, parameter, value)
#             if parameters['manifold_kernel_params_pack'] is not None:
#                 for i, param in enumerate(['lbo_temperature', 'manifold_kernel_lengthscale', 'manifold_kernel_noise', 'manifold_kernel_power', 'manifold_kernel_amplitude']):
#                     setattr(self, param, list(parameters['manifold_kernel_params_pack'])[i])   
                    
#             if parameters['gp_kernel_params_pack'] is not None:
#                 for i, param in enumerate(['gp_kernel_lengthscale', 'manifold_kernel_amplitude']):
#                     setattr(self, param, list(parameters['gp_kernel_params_pack'])[i])    
        except:
            TypeError("This is the NONE!!!")
        self.control_amplitudes()
        return self  
    
    def get_params(self, deep=True):
        
        return_dict = {
        'keras_model_weights': self.keras_model_weights,
        'freeze_psychometric_original': self.freeze_psychometric_original,
        'encoder': self.encoder,
        'is_SPM': self.is_SPM,
        # 'little_input_dict': self.little_input_dict, 
        'sig_reg_coeff': self.sig_reg_coeff,
        'sig_reg_penalty': self.sig_reg_penalty,
        'manifold_kernel_type' : self.manifold_kernel_type,
        'epochs' : self.epochs,
        'is_fitted': self.is_fitted,
        'optimization_results': self.optimization_results, 
        'psych_reg_coeff' : self.psych_reg_coeff,
        'psych_reg_penalty' : self.psych_reg_penalty,
        'metrics' : self.metrics,
        'batch_size' : self.batch_size,
        'workers' : self.workers,
        'verbose' : self.verbose,
        'child' : self.child,
        'history': self.history, 
        'freeze_psych_alpha': self.freeze_psych_alpha,
        'fit_counts': self.fit_counts,
        'loss_type' : self.loss_type,
        'use_multiprocessing' : self.use_multiprocessing,
        'keras_model_initialized' : self.keras_model_initialized,
        'calibrate' : self.calibrate,
        'warm_f_g_prime_init' : self.warm_f_g_prime_init,
        'warm_f_l_prime_init' : self.warm_f_l_prime_init,
        'f_g_prime_init' : self.f_g_prime_init,
        'f_l_prime_init' : self.f_l_prime_init,
        'alternate_training' : self.alternate_training,
        'constrained_optimization' : self.constrained_optimization,
        'barrier_initial_C' : self.barrier_initial_C,
        'optimizer_dict' : self.optimizer_dict,
        'warm_cv_params' : self.warm_cv_params,
        'gp_kernel_lengthscale' : self.gp_kernel_lengthscale,
        'gp_kernel_lengthscale_trainable' : self.gp_kernel_lengthscale_trainable,
        'gp_kernel_amplitude' : self.gp_kernel_amplitude,
        'manifold_kernel_lengthscale': self.manifold_kernel_lengthscale,
        'manifold_kernel_lengthscale_trainable': self.manifold_kernel_lengthscale_trainable,
        'manifold_kernel_amplitude': self.manifold_kernel_amplitude,
        'manifold_kernel_amplitude_trainable': self.manifold_kernel_amplitude_trainable,
        'manifold_kernel_k': self.manifold_kernel_k, 
        'noise_sigma' : self.noise_sigma,
        'lr_reduce_min_delta' : self.lr_reduce_min_delta,
        'end_training_min_delta' : self.end_training_min_delta,
        'dropout_rate' : self.dropout_rate,
        'manifold_neighbor_mode' : self.manifold_neighbor_mode,
        'gp_kernel_type' : self.gp_kernel_type,
        'lbo_temperature' : self.lbo_temperature,
        'manifold_kernel_noise' : self.manifold_kernel_noise,
        'manifold_kernel_power' : self.manifold_kernel_power,
        'manifold_kernel_normed' : self.manifold_kernel_normed,
        'g_prime' : self.g_prime,
        'l_prime' : self.l_prime,
        'psych_alpha' : self.psych_alpha,
        'psych_beta' : self.psych_beta,
        'sig_a' : self.sig_a,
        'sig_b' : self.sig_b,
        'psych_alpha' : self.psych_alpha,
        'psych_beta' : self.psych_beta,
        'psych_lambda' : self.psych_lambda,
        'psych_gamma' : self.psych_gamma,
        'alternate_history_list': self.alternate_history_list,
        'alternate_param_reporter_list': self.alternate_param_reporter_list,
        'alternate_descend_epoch_list': self.alternate_descend_epoch_list,
        'parameter_reporter': self.parameter_reporter,
        'gp_kernel_params_pack': self.gp_kernel_params_pack,
        'manifold_kernel_params_pack': self.manifold_kernel_params_pack,
        'number_of_successful_attempts': self.number_of_successful_attempts,
        'sampled_input_features': self.sampled_input_features,
        'kernel': self.kernel,
        'max_rerun': self.max_rerun, 
        'min_rerun': self.min_rerun, 
        'training_size': self.training_size,
        'params_sample': self.params_sample, 
        'fresh_opt_initial_point': self.fresh_opt_initial_point,
        'with_laplace_method': self.with_laplace_method,
        'ambient_to_intrinsic_amplitude_ratio': self.ambient_to_intrinsic_amplitude_ratio,
        'prior_sample_size': self.prior_sample_size,
        'max_iter': self.max_iter,
        'invert_manifold_mat_explicitly': self.invert_manifold_mat_explicitly,
        'I_plus_KM_inv_M_calc_method': self.I_plus_KM_inv_M_calc_method,
        'I_plus_KM_inv_M_using_factorization': self.I_plus_KM_inv_M_using_factorization,
        'I_plus_KM_inv_M_using_expm_acting': self.I_plus_KM_inv_M_using_expm_acting,
        'I_plus_KM_inv_M_using_eigsh': self.I_plus_KM_inv_M_using_eigsh,
        'optimal_threshold': self.optimal_threshold,
        'unlabeled_data_inc_type': self.unlabeled_data_inc_type
        }
#         try:
#             if self.manifold_kernel_params_pack is not None:
#                 for i, param in enumerate(['lbo_temperature', 'manifold_kernel_lengthscale', 'manifold_kernel_noise', 'manifold_kernel_power', 'manifold_kernel_amplitude']):
#                     return_dict[param] = list(self.manifold_kernel_params_pack.values)[i]

#             if self.gp_kernel_params_pack is not None:
#                 for i, param in enumerate(['gp_kernel_lengthscale','gp_manifold_kernel_amplitude']):
#                     return_dict[param] = list(self.gp_kernel_params_pack.values)[i]
#         except Exception as e:
#             if hasattr(e, 'message'):
#                 raise type(e)(e.message+'get_params has a problem' )
#             else:
#                 raise type(e)(str(e)+'get_params has a problem' )
                        
        return return_dict
                

#     def set_params(self, **parameters):
#         from glob import glob
#         from dill import dump
#         setattr(self, 'encoder', parameters['encoder'])
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
# #         if hasattr(self, 'history_'):
# #             beta_scores = self.history_['fbeta_score'][-1]
# #             if beta_scores[0]>.8 and beta_scores[1]>0.8:
# #                 file_name = 'test_psych_model'
# #                 file_name = file_name+'_'+str(len(glob('/home/scratch/nshajari/psych_model/test_psych_models/' + file_name+'*'))+1)
# #                 with open('/home/scratch/nshajari/psych_model/test_psych_models/' + file_name + '.pkl', 'wb') as f:
# #                     dump(self.get_params(), f)
#         setattr(self, 'encoder', parameters['encoder'])
#         return self        
        
        
        
#     def set_g_prime_l_prime_functions(self):
#         def identity(x):
#             return x
#         if self.g_prime_init is None:
#             if self.f_g_prime_init is None:
#                 print("No function assigned to f_g_prime_init... setting it to default f_g_prime_init_func")
#                 self.f_g_prime_init = f_g_prime_init_func
#             self.g_prime_init = self.f_g_prime_init(1.)
            
            
#         if self.l_prime_init is not None:
#             if self.f_l_prime_init is None:
#                 print("No function assigned to f_l_prime_init... setting it to default f_l_prime_init_func")
#                 self.f_l_prime_init = f_l_prime_init_func
#             self.l_prime_init = self.f_l_prime_init(1.)

#     def set_warm_gamma_lambda_functions(self):
#         if self.warm_f_g_prime_init is None:
#             self.warm_f_g_prime_init = warm_f_g_prime_init_func
#         else:
#             self.warm_f_g_prime_init = warm_f_g_prime_init_func

#         if self.warm_f_l_prime_init is None:
#             self.warm_f_l_prime_init = warm_f_g_prime_init_func
#         else:
#             self.warm_f_l_prime_init = warm_f_g_prime_init_func
            
            
            

            
    def compile(self, optimizer, model_loss, metrics=[]):
        self.keras_model.compile(loss=model_loss, optimizer=optimizer, metrics=metrics)

    def initialize_model(self, sampled_input_features=None, reinitialize=False, set_fit_params=True):
        if sampled_input_features is None:
            sampled_input_features = self.sampled_input_features
        tf.print ("initializing the model... g_prime is:", self.g_prime, "and l_prime is:", self.l_prime, output_stream=sys.stderr)
        self.SPM_switch_call = False
        if reinitialize:

            # This function keeps the learning rate at 1e-10 for the first ten epochs
            # and decreases it exponentially after that.
            import numpy as np
            def scheduler(epoch, lr):
                starting_epoch = 5
                breaking_num = 10
                if epoch < starting_epoch:
                    return lr
                else:
                    coeff = int((epoch - starting_epoch)  / breaking_num) + 1
                    return lr * tf.math.exp(.05 * (-coeff))

#             print ("is_SPM is:", self.is_SPM)
            # g_prime = None
            # l_prime = None
            if self.is_SPM or self.freeze_psychometric_original:
                g_prime=self.g_prime 
                l_prime=self.l_prime
            else:
                g_prime = None
                l_prime = None
                
            sig_a = None
            sig_b = None
            
            if self.freeze_psychometric_original:
                psych_alpha=self.psych_alpha
                psych_beta=self.psych_beta
            else:
                psych_alpha = None
                psych_beta = None
        else:
            g_prime=self.g_prime 
            l_prime=self.l_prime
            sig_a=self.sig_a 
            sig_b=self.sig_b
            psych_alpha=self.psych_alpha
            psych_beta=self.psych_beta
        try:
            self.keras_model = PsychMKeras(unlabeled_data_inc_type=self.unlabeled_data_inc_type, is_SPM=self.is_SPM, sig_reg_coeff=self.sig_reg_coeff, manifold_kernel_type = self.manifold_kernel_type, sig_reg_penalty=self.sig_reg_penalty, psych_reg_coeff = self.psych_reg_coeff, psych_reg_penalty=self.psych_reg_penalty, parent=self, loss_type=self.loss_type, constrained_optimization=self.constrained_optimization, barrier_initial_C=self.barrier_initial_C, GP_input_type='index', sampled_input_features=sampled_input_features, gp_kernel_lengthscale=self.gp_kernel_lengthscale, gp_kernel_lengthscale_trainable=self.gp_kernel_lengthscale_trainable, noise_sigma=self.noise_sigma, dropout_rate=self.dropout_rate, manifold_kernel_lengthscale=self.manifold_kernel_lengthscale, manifold_kernel_lengthscale_trainable=self.manifold_kernel_lengthscale_trainable, manifold_kernel_amplitude=self.manifold_kernel_amplitude, gp_kernel_amplitude=self.gp_kernel_amplitude, manifold_kernel_amplitude_trainable=self.manifold_kernel_amplitude_trainable,
    manifold_neighbor_mode=self.manifold_neighbor_mode, gp_kernel_type=self.gp_kernel_type, lbo_temperature=self.lbo_temperature, manifold_kernel_noise=self.manifold_kernel_noise, manifold_kernel_power=self.manifold_kernel_power, manifold_kernel_normed=self.manifold_kernel_normed, manifold_kernel_k=self.manifold_kernel_k, g_prime=g_prime, l_prime=l_prime, sig_a=sig_a, sig_b=sig_b, psych_alpha=psych_alpha, psych_beta=psych_beta, freeze_psychometric_original=self.freeze_psychometric_original, training_size=self.training_size, invert_manifold_mat_explicitly=self.invert_manifold_mat_explicitly,
                                           I_plus_KM_inv_M_calc_method=self.I_plus_KM_inv_M_calc_method,
                                           I_plus_KM_inv_M_using_factorization=self.I_plus_KM_inv_M_using_factorization,
                                           I_plus_KM_inv_M_using_expm_acting=self.I_plus_KM_inv_M_using_expm_acting,
I_plus_KM_inv_M_using_eigsh=self.I_plus_KM_inv_M_using_eigsh, freeze_psych_alpha=self.freeze_psych_alpha)
        except Exception as e:
            if hasattr(e, 'message'):
                raise type(e)(e.message+'Squaring W is causing error 21' )
            else:
                raise type(e)(str(e)+'Squaring W is causing error 21' )
#         if self.freeze_psychometric_original:
# #             pass
#             for var in self.keras_model.psych_layer.trainable_variables:
#                 param_name = var.name.split('/psychometric_layer/')[1].split(':')[0]
#                 param_value = getattr(param_name)
#                 var.assign(param_value)
        self.keras_model(sampled_input_features)
        # try:
            # if self.little_input_dict is None:
            #     self.set_little_input_dict(little_input_dict)
            # tf.print("little_input_dict:", self.little_input_dict, output_stream=sys.stderr)
            # self.keras_model(self.little_input_dict)
        # except Exception as e:
        #     if hasattr(e, 'message'):
        #         raise type(e)(e.message+'Squaring W is causing error 16' )
        #     else:
        #         raise type(e)(str(e)+'Squaring W is causing error 16' )
        try:
#         if sampled_input_features['sig_input'] is not None and hasattr(self, 'keras_model'):
#             sampled_input_features['sig_input'] = self.keras_model.sampled_input_features['sig_input']
    #         tf.print("Keras layer name:", type(self.keras_model.sig_layer).__name__)
            if type(self.keras_model.sig_layer).__name__ == 'GPLayer':
                self.GP_model_exists = True
                kernel_mat = self.keras_model.sig_layer.kernel_mat
                # manifold_mat = self.keras_model.sig_layer.manifold_mat
                if hasattr(kernel_mat, 'numpy'):
                    kernel_mat = kernel_mat.numpy()
                    
                # Lap_mat = self.keras_model.sig_layer.Lap_mat        
                # exp_Lap_mat = self.keras_model.sig_layer.exp_Lap_mat        

                
            # checking if we are in Laplacian-based SSL setting and if so 
            # to update the set of unlabeled points
            if self.manifold_kernel_type is not None:
                # and self.sampled_input_features['sig_input'] is not None:
                self.keras_model.sampled_input_features = sampled_input_features
                # if self.GP_model_exists:
                self.kernel_mat = kernel_mat
                # self.manifold_mat = manifold_mat
                # self.Lap_mat = Lap_mat
                # if 'LBO' in self.manifold_kernel_type:
                #     if hasattr(exp_Lap_mat, 'numpy'):
                #         self.manifold_mat = exp_Lap_mat.numpy()
                #     else:
                #         self.manifold_mat = exp_Lap_mat
                # else:
                #     if hasattr(Lap_mat, 'numpy'):
                #         self.manifold_mat = Lap_mat.numpy()
                #     else:
                #         self.manifold_mat = Lap_mat


            self.child = self.keras_model
        except Exception as e:
            if hasattr(e, 'message'):
                raise type(e)(e.message+'Squaring W is causing error 18' )
            else:
                raise type(e)(str(e)+'Squaring W is causing error 18' )
        var_list = []
        for var in self.keras_model.trainable_variables:            
            print (var.name)
            if 'sig_layer/alpha' in var.name:
                var_list += [ones(tf.shape(self.keras_model.sig_layer._alpha)[0])]
            if 'sig_layer/beta' in var.name:
                var_list += [[1e-2]]
            if 'psychometric_layer/alpha' in var.name:
                var_list += [ones(tf.shape(self.keras_model.psych_layer._alpha)[0])]                
            if 'psychometric_layer/beta' in var.name:
                var_list += [[1e-2]]                
            if 'psychometric_layer_gamma_prime' in var.name:
                var_list += [[1e-2]]                
            if 'psychometric_layer_lambda_prime' in var.name:
                var_list += [[1e-2]]                
                
        cov_diag = hstack(var_list)
        mean = zeros_like(cov_diag)
        # print ("MEAN AND COV:", mean, cov_diag)
        self.params_sample = multivariate_normal(mean=mean, cov=diag(cov_diag), size=self.prior_sample_size)
            
        self.refresh_params()
        print ("intializing done")
#     def training_failed():
#         history_, parameter_reporter, lookback_reduce_lr
        
    def keras_fit(self, X_train=None, l_y_decimal_train=None, validation_data=None, lr=None, optimizer=None, tol=None):
        from MyCallbacks import ParameterReporter,  ReduceLROnRelativePlateauWithLookback
        from scorer_library import LPUF1ScoreForY, LPUF1ScoreForL, LPUBrierScoreForL, LPUBrierScoreForY
        try:
            try:
                parameter_reporter = ParameterReporter()
                lookback_reduce_lr = ReduceLROnRelativePlateauWithLookback(monitor='loss', mode='min', factor=.5, k=5,  patience=1,  verbose=self.verbose, restore_best_weights=True, min_lr=self.end_training_min_delta, min_delta=self.lr_reduce_min_delta)     
        #         tf.print("Model in keras_fit is:", self.keras_model,  output_stream=sys.stderr)

                lpu_f1_scorer_for_y = LPUF1ScoreForY(num_classes=2, model=self, average=None, threshold=0.5)
                lpu_f1_scorer_for_l = LPUF1ScoreForL(num_classes=2, model=self, average=None, threshold=0.5)
            except Exception as e:
                raise type(e)(str(e)+"it is scorer_library that causes errors")
        except Exception as e:
            raise type(e)(str(e)+"exterior level of first try in keras_fit")
        N_dataset_size = l_y_decimal_train.shape[0]
#         if 'int' in type(self.batch_size).__name__:
#             self.batch_size = min(N_dataset_size, self.batch_size)
#         else:
#             self.batch_size = int(N_dataset_size * self.batch_size)
        # the following loop is intended to address the problem of failed fitting which might 
        # run into nan/inf values. the loop will only break when none of the parameters after fitting are nan.
        # while (True): LOOP IS NOT HERE ANYMORE FOR NOW
        dataset_generator = self.create_keras_dataset(X_train, l_y_decimal_train, batch_size=self.batch_size)
        
        optimizer_dict = self.optimizer_dict.copy()
        if lr is not None:
            optimizer_dict['learning_rate'] = lr 
        if optimizer is not None:
            pass
#             if 'keras' in optimizer_dict['optimizer_name']:

#             else:
#                 raise NotImpelementedError("You can only use Keras optimizers at this point")
        else:
            pass
#             try:
# #                 optimizer = self.optimizer
#             except Exception as e:
#                 tf.print ("problem is here")
#                 if hasattr(e, 'message'):
#                     tf.print ("problem is here:", e.message)
#                     raise type(e)(e.message + ' problem is here')                    
#                 else:
#                     raise type(e)(str(e)+'problem is here' )
            
#         with tf.GradientTape() as tape:
#             loss = self.keras_model.model_loss
#         tf.print ("OPTIMIZER IS:", self.optimizif 'keras' in selfer_name)
    
        if 'keras' in optimizer_dict['optimizer_name']:
            real_optimizer_name = optimizer_dict['optimizer_name'].split('_')[1].capitalize()
            optimizer = eval('tf.keras.optimizers.' + real_optimizer_name) 
            tf.print ("Compiling loss...")
            self.compile(optimizer(**{'learning_rate':optimizer_dict['learning_rate']}), metrics=self.metrics+[lpu_f1_scorer_for_y, lpu_f1_scorer_for_l, LPUBrierScoreForL(model=self), LPUBrierScoreForY(model=self)], model_loss=self.keras_model.model_loss)
    # [lpu_f1_scorer, LossGradientNorm(model=self)], model_loss=self.keras_model.model_loss)
            tf.print ("Compiling loss done...")
    #         self.compile(self.optimizer_name(**{'learning_rate':lr}) if optimizer is None else optimizer(**{'learning_rate':lr}), metrics=self.metrics, model_loss={'output_1':self.keras_model.model_loss, 'output_2':self.keras_model.model_loss, 'output_3': self.keras_model.model_loss, 'output_4': self.keras_model.model_loss})
            try:
#                 tf.print ("TYPES:", type(dataset_generator), type(parameter_reporter), type(lookback_reduce_lr), type(self.epochs), type(self.verbose), type(self.batch_size), type(self.workers), type(validation_data), output_stream=sys.stderr)
                history_ = self.keras_model.fit(dataset_generator, epochs=self.epochs, use_multiprocessing=self.use_multiprocessing, batch_size=self.batch_size, verbose=self.verbose, callbacks=[parameter_reporter, lookback_reduce_lr], workers=self.workers, validation_data=validation_data).history
            except Exception as e:
                raise type(e)(str(e) + " the error happens when calling keras_model.fit" + str(self.keras_model.get_weights()))
            self.optimization_results = {'converged':False, 'failed':False}
        else:
            try:
                try:
                    # for item in self.keras_model.trainable_variables:
                    var_list = []
                    # for item in self.keras_model.trainable_variables:
                        # if tf.rank(item) == 0:
                        #     new_item = tf.expand_dims(tf.expand_dims(item, axis=0), axis=0)
                        # if tf.rank(item) == 1:
                        #     new_item = tf.expand_dims(item, axis=0)
                        # else:
                        #     new_item = item
                        # var_list.append(new_item)
                    init_params = tf.reshape(tf.concat([tf.expand_dims(tf.expand_dims(item, axis=0), axis=0) if tf.rank(item)==0 else item for item in self.keras_model.trainable_variables], axis=0), [1, -1])[0]
                    tf.print("Using Tensorflow-probability Optimizer")
                    func_and_grad_generator = FunctionFactory(self.keras_model, self.keras_model.model_loss, X_train, l_y_decimal_train, with_gradient=True, verbose=False)
                except Exception as e:
                    tf.print ("TFP is fucking me!!! :'( '")
                    if hasattr(e, 'message'):
                        tf.print ("THE POOPY message is:", e.message)
                        raise type(e)(e.message + ' happens at fucking here')                    
                    else:
                        raise type(e)(str(e)+'TFP you MOFO' )
                try:
                    try:
                        t = time()
                        results=tfp.optimizer.lbfgs_minimize(value_and_gradients_function=lambda x: func_and_grad_generator(x)[:2], initial_position=init_params, max_iterations=self.max_iter, tolerance=tol, parallel_iterations=1)
                        print ("TFP LBFGS took", time() - t, "seconds...")
                        # results=tfp.optimizer.lbfgs_minimize(value_and_gradients_function=lambda x: func_and_grad_generator(x)[:2], initial_position=init_params, tolerance=1e-6, max_line_search_iterations=100, num_correction_pairs=100,  max_iterations=100)
                        # results=tfp.optimizer.lbfgs_minimize(lambda x: func_and_grad_generator(x)[:2], initial_position=init_params, max_iterations=1000, tolerance=1e-8, max_line_search_iterations=500, num_correction_pairs=500, parallel_iterations=1)
                    except Exception as e:
                        raise type(e)(str(e) + 'lbfgs_minimize fuck you!!')
    #                         value_and_gradients_function=func_and_grad_generator, initial_position=init_params, previous_optimizer_results=None,
    #                         num_correction_pairs=100, tolerance=1e-08, x_tolerance=0, f_relative_tolerance=0,
    #                         initial_inverse_hessian_estimate=None, max_iterations=self.epochs, parallel_iterations=50,
    #                         stopping_condition=None, max_line_search_iterations=100, name=None)
                    for key, value in func_and_grad_generator.history.items():
                        func_and_grad_generator.history[key] = [item.numpy() if hasattr(item, 'numpy') else item for item in value]
                    history_ = func_and_grad_generator.history
                    lookback_reduce_lr.best = np.min(func_and_grad_generator.history['loss'])
                    
                    assign_new_model_parameters(self.keras_model, results.position, 
                                                func_and_grad_generator.partition_idx, 
                                                func_and_grad_generator.n_tensors, func_and_grad_generator.shapes)
                    
                    # func_and_grad_generator.assign_new_model_parameters(results.position)
                    self.final_pos = results.position.numpy()
                    
                    

                    counter = 0
                        
                    # tf.print ("eigenvalues are:", np.linalg.eig(temp_hessian.numpy()))
                    # tf.print ("final_hessian:", tf.math.reduce_all((tf.squeeze(final_hessian) - tf.linalg.adjoint(tf.squeeze(final_hessian)))<1e-5), output_stream=sys.stderr)
                    # tf.print("element " + str(counter) + ":", element, element.shape, output_stream=sys.stderr)
                    
                    lookback_reduce_lr.best_weights = self.keras_model.get_weights()
                    self.optimization_results = {'converged':results.converged, 'failed':results.failed}
                except Exception as e:
                    tf.print ("sampled_input_features :'( '2", init_params.shape, X_train['sig_input'].shape, "func_and_grad_generator.history.items():", func_and_grad_generator.history.items(), 'results:')
                    if hasattr(e, 'message'):
                        tf.print ("THE POOPY message is 2:", e.message)
                        raise type(e)(e.message + ' happens at fucking here 2')               
                    else:
                        raise type(e)(str(e)+'TFP you MOFO 2')
            except Exception as e:
                raise type(e)(str(e)+'LBFGS is the culprit!!')
            print ("fitting is done. starting to calculate the hessian")
            t = time()
            _, _, final_hessian = func_and_grad_generator.return_hessian(results.position)
            print ("calculating the hessian took", time()-t, "seconds...")
            # print ("final_hessian:", final_hessian, self.keras_model.trainable_variables)
            # final_hessian = tf.concat(final_hessian, axis=1)
            # final_hessian = tf.squeeze(final_hessian)
            # final_hessian = tf.where(tf.math.is_nan(final_hessian), tf.ones_like(final_hessian) * 0., final_hessian)
            # final_hessian = ((final_hessian + tf.transpose(final_hessian)) / 2.).numpy()
            # final_hessian_std = np.std(final_hessian)
            counter = 0
            while True:
                try:
                    # eps_diag = tf.experimental.numpy.ravel(tf.abs(tf.random.normal(shape=[tf.shape(final_hessian)[0]], stddev=1e-4, dtype=tf.float64))).numpy()                    
                    # eps_diag[eps_diag < EPSILON] = EPSILON
                    eps_diag = tf.linalg.diag(tf.ones(tf.shape(final_hessian)[0], dtype=tf.float64) * DELTA)
                    temp_hessian = final_hessian + (counter > 0) * eps_diag * 10 ** counter
                    try:
                        cholesky_L = cholesky(temp_hessian) # notice that for scipy if L = cholesky(A) L.T @ L = A and NOT L @ L.T = A
                    except LinAlgError as e:
                        counter += 1                        
                        logging = tf.get_logger()
                        logging.warning("counter increased to " + str(counter) + " for hessian inversion")
                        continue
                    break
                except Exception as e:
                    print (str(e), counter)
                    
                # if counter > 1:
                    # cholesky_L = np.eye(final_hessian.shape[0]) * 1e4
                    # self.final_hessian_inv = tf.eye(temp_hessian.shape[0])
            
                # if counter > 100:
                # #     temp_hessian = tf.eye(final_hessian.shape[0])
                    # break
            try:
                if hasattr(self, 'with_laplace_method') and self.with_laplace_method:
                    from scipy.linalg.lapack import dtrtri
                    inv_cholesky_L = dtrtri(cholesky_L)[0]
                    id_samples = np.random.randn(temp_hessian.shape[0], self.prior_sample_size)
                    self.params_sample = (inv_cholesky_L @ id_samples).T + self.final_pos
                    # params_sample = params_sample + self.final_pos
                    # from scipy.stats import multivariate_normal 
                    # self.params_sample = multivariate_normal.rvs(mean=self.final_pos, cov=self.final_hessian_inv, size = 1000)
                else:
                    self.params_sample = np.asarray([self.final_pos] * self.prior_sample_size)
            except Exception as e:
                raise type(e)(str(e)+"HERE WE ARE WITH multivariate_normal fucking things up!!")
                
            
#             parameter_reporter = None
#             lookback_reduce_lr
#             if lr < 1e-5:
#                 self.training_failed()
                
            
#         print ("Training ended due to early stopping?", training_ender.ending_activated)
#         print ("Training ended due to lookback early stopping?", lookback_training_ender.ending_activated)
        self.optimization_results_ = self.optimization_results
        return history_, parameter_reporter, lookback_reduce_lr
        
        
    def partial_fit(self, X, y, validation_data=None, c_estimation_method='elkan', weights=None, reinitialize=True):
        pass
        
        
        
    def refresh_params(self):
        psych_layer_weights = self.keras_model.psych_layer.get_weights()
        gp_layer_weights = self.keras_model.sig_layer.get_weights()
        # psych_alpha_, psych_beta_, g_prime_, l_prime_ =  [psych_layer_weights[item] for item in self.keras_model.psych_layer.order]
        psych_alpha_, psych_beta_, g_prime_, l_prime_ =  psych_layer_weights
        if self.manifold_kernel_type is not None:
            # if self.gp_kernel_lengthscale_trainable:
            #     gp_kernel_lengthscale_, manifold_kernel_lengthscale_, sig_a_, sig_b_ = [gp_layer_weights[item] for item in self.keras_model.sig_layer.order]
            # else:
            # sig_a_, sig_b_ = [gp_layer_weights[item] for item in self.keras_model.sig_layer.order]
            gp_kernel_lengthscale_, manifold_kernel_lengthscale_ = self.keras_model.sig_layer.gp_kernel_lengthscale, self.keras_model.sig_layer.manifold_kernel_lengthscale
        else:
            gp_kernel_lengthscale_= manifold_kernel_lengthscale_ = 1.
            # sig_a_, sig_b_, _, _ = [gp_layer_weights[item] for item in self.keras_model.sig_layer.order]
            sig_a_, sig_b_, _, _ = gp_layer_weights


        
        try:
            psych_gamma_, psych_lambda_ = g_l_prime_to_gamma_lambda_transformer(g_prime_, l_prime_)
        except Exception as e:
            raise type(e)(str(e)+'trouble is in converting g_prime and l_prime to gamma and lambda, g_prime:'+str(g_prime_)+" and l_prime:"+str(l_prime_)) 

        try:
            self.sig_a = sig_a_
            self.sig_b = sig_b_
            self.psych_alpha = psych_alpha_
            self.psych_beta = psych_beta_
            self.g_prime = g_prime_
            self.l_prime = l_prime_
            self.psych_gamma = psych_gamma_
            self.psych_lambda = psych_lambda_
            # self.gp_kernel_lengthscale = gp_kernel_lengthscale_
            # self.manifold_kernel_lengthscale = manifold_kernel_lengthscale_

            self.sig_a_ = sig_a_
            self.sig_b_ = sig_b_
            self.psych_alpha_ = psych_alpha_
            self.psych_beta_ = psych_beta_
            self.g_prime_ = g_prime_
            self.l_prime_ = l_prime_
            self.psych_gamma_ = psych_gamma_
            self.psych_lambda_ = psych_lambda_
            # self.gp_kernel_lengthscale_ = gp_kernel_lengthscale_
            # self.manifold_kernel_lengthscale = manifold_kernel_lengthscale_
            
            
            # if self.manifold_kernel_type is not None and ('laplacian' in self.manifold_kernel_type or 'LBO' in self.manifold_kernel_type):
            
            self.kernel = self.keras_model.sig_layer.kernel
                
            # else:
                # raise ValueError("+!FUCK!+")
                # self.kernel = None
#             beta_scores = self.history['fbeta_score'][-1]
        except Exception as e:
            if hasattr(e, 'message'):
                raise type(e)(e.message+'trouble is in setting the chosen values after fitting' )
            else:
                raise type(e)(str(e)+'trouble is in setting the chosen values after fitting' )
        
#     def set_little_input_dict(self, little_input_dict=None):
#         if little_input_dict is None:
#             if hasattr(self, 'X_train_'):
#                 input_dict = self.X_train_
#             else:
#                 raise ValueError("little_input_dict cannot be set. You need to initialize model with giving a small input_dict or call the fit() method")
#         else:
#             input_dict = little_input_dict 
            
#         if type(input_dict['psych_input']).__name__ == 'csr_matrix':
#             self.little_input_dict = {'psych_input': tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(input_dict['psych_input'][0])), 'sig_input': tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(input_dict['sig_input'][0])), 'idx':[input_dict['idx'][0]]}
#         else:
#             self.little_input_dict = {'psych_input': tf.convert_to_tensor([input_dict['psych_input'][0]], dtype=tf.float64), 'sig_input': tf.convert_to_tensor([input_dict['sig_input'][0]], dtype=tf.float64), 'idx':[input_dict['idx'][0]]}
            
            

        
    def fit(self, X, y, validation_data=None, c_estimation_method='elkan', weights=None, reinitialize=True, gp_kernel_setting=None, fresh_opt_initial_point=False, tol=1e-6):    
        # tf.config.threading.set_inter_op_parallelism_threads(1)
        # tf.config.threading.set_intra_op_parallelism_threads(1)
        if gp_kernel_setting == 'auto':
            self.gp_kernel_lengthscale = 1 * (X['sig_input'].std() * np.sqrt(X['sig_input'].shape[-1]))
            self.gp_kernel_lengthscale_ = self.gp_kernel_lengthscale
        try:
            if hasattr(self, 'optimizer_dict') and 'keras' in self.optimizer_dict['optimizer_name']:
                real_optimizer_name = self.optimizer_dict['optimizer_name'].split('_')[1].capitalize()
                optimizer = eval('tf.keras.optimizers.' + real_optimizer_name) 
            else:
                tf.print ("The optimizer is not set. Setting the optimizer to Adam by default")
                optimizer = tf.keras.optimizers.Adam
            X['idx'] = np.arange(len(X['sig_input']))
            self.X_train_ = X 
            self.l_y_cat_transformed_train_ = y

            self.sampled_input_features = self.X_train_
            self.sampled_input_features_ = self.X_train_
            
            self.initialize_model()

#             if self.f_g_prime_init is None and self.g_prime is None:
#                 raise NotImplementedError("either of f_g_prime_init or g_prime need to be set for fitting to work.")
#             if self.f_l_prime_init is None and self.l_prime is None:
#                 raise NotImplementedError("either of f_l_prime_init or l_prime init need to be set for fitting to work.")

            self.fitting_failed = False
            if self.manifold_kernel_type:
                self.GP_model_exists = True
            else:
                self.GP_model_exists = False

            if self.warm_cv_params is None:
                warm_start = False
            else:
                warm_start = True
            self.lr = self.optimizer_dict['learning_rate']
    #         self.set_g_prime_l_prime_functions()
    #         self.set_warm_gamma_lambda_functions()
            self.fitting = True
            import sys
#             from LPUModels.PUAdapterTF import PUAdapterTF
            # just converting names
            self.multiple_representations = True
        except Exception as e:
            raise type(e)(str(e)+'Squaring W is causing error  10' )
        try:
            
    #         temp_dict_ = dict()
    #         temp_dict_['psych_input'] = X_train[self.psych_vec]\
    #                                     [tuple(sorted((k, v) for k, v in self.psych_vec_params.items()))]
    #         temp_dict_['sig_input'] = X_train[self.sig_vec]\
    #                                     [tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]

#             if self.calibrate:
#                 temp_dict_train_ = dict()
#                 temp_dict_cal_ = dict()
#                 from sklearn.model_selection import train_test_split

#     # #             X_cal_[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))] = temp_dict_cal_[self.sig_vec]

#     # #             X_cal_[self.psych_vec][tuple(sorted((k, v) for k, v in self.psych_vec_params.items()))] = temp_dict_cal_[self.psych_vec]



#                 l_y_decimal_cal = self.encoder.inverse_transform(y)
#                 l = (l_y_decimal_cal / 2).astype(int)

#                 temp_dict_train_['psych_input'], temp_dict_cal_['psych_input'], temp_dict_train_['sig_input'], temp_dict_cal_['sig_input'], y, y_cal = train_test_split(self.X_train_['psych_input'],self.X_train_['sig_input'], y, stratify=l, test_size=0.1)
                # X_train_ = temp_dict_train_
    # #             X_cal_ = dict()
    # #             X_cal_[self.sig_vec] = dict()
                # X_cal_ = temp_dict_cal_
                
                
            print ("The inverse transform is running")
            self.l_y_decimal_train_ = self.encoder.inverse_transform(self.l_y_cat_transformed_train_)
            print ("The inverse transform is done")




    ####################
    ####### VALIDATION NEEDS UPDATE FOR SURE TO LOOK LIKE CALIBRATION. I DON'T THINK IT WORK AS IS!
    ####################
            if validation_data is not None:
                X_val, _, _, l_y_cat_transformed_val_ = validation_data
                l_y_decimal_val_ = self.encoder.inverse_transform(l_y_cat_transformed_val_)
                validation_data_ = self.create_keras_dataset(X_val, l_y_decimal_val_, batch_size=self.batch_size)
            else:
                validation_data_ = None

        except Exception as e:
            raise type(e)(str(e)+'Squaring W is causing error  12' )
#         try:
#         except Exception as e:
#             if hasattr(e, 'message'):
#                 tf.print ("THE POOPY message is:", e.message)
#                 raise type(e)(e.message + 'Squaring W is causing error  7')                    
#             else:
#                 raise type(e)(str(e)+'Squaring W is causing error  7' )
#         try:

#             self.little_input_dict['idx'] = tf.convert_to_tensor([np.arange(1).astype(np.float64)], dtype=tf.float64)
#         except Exception as e:
#             if hasattr(e, 'message'):
#                 tf.print ("THE POOPY message is:", e.message)
#                 raise type(e)(e.message + 'Squaring W is causing error  12')                    
#             else:
#                 raise type(e)(str(e)+'Squaring W is causing error  12' )
#         try:
#             tf.print ("initialize_model is running...")
        #         tf.print("little input dict shape is:", [tf.keras.backend.shape(self.little_input_dict[item]) for item in self.little_input_dict.keys()])
        #         tf.print("And trainable variables are:", self.keras_model.trainable_variables)
        #         if not self.is_SPM:
        #         self.keras_model.psych_layer.g_prime.assign([[self.f_g_prime_init(2.)]])
        #         self.keras_model.psych_layer.l_prime.assign([[self.f_l_prime_init(.1)]])

        except Exception as e:
            if hasattr(e, 'message'):
                tf.print ("THE POOPY message is:", e.message)
                raise type(e)(e.message + 'Squaring W is causing error  11')                    
            else:
                raise type(e)(str(e)+'Squaring W is causing error  11' )
        try:
        
            self.alternating_best_weights = self.keras_model.get_weights()
            self.initial_weights = self.keras_model.get_weights()
            #       self.alternating_psych_layer_best_weights = self.keras_model.psych_layer.get_weights()
            #       self.alternating_sig_layer_best_weights = self.keras_model.sig_layer.get_weights()
            self.alternating_best_loss = np.inf#self.keras_model.evaluate(dataset_generator, return_dict=True)['loss']
            self.alternate_history_list = []
            self.alternate_param_reporter_list = []
            self.alternate_descend_epoch_list = []
            if warm_start:
                if weights is None:
                    if c_estimation_method == 'elkan':
                        warm_start_model = LapPUAdapterTF(encoder=self.encoder,
                                                          preprocessing_type=self.preprocessing_type, 
                                                          maxiter=self.max_iter, 
                                                          tol=self.tol, 
                                                          kernel_type=self.gp_kernel_type, 
                                                          manifold_regularize=self.unlabeled_data_inc_type, 
                                                          sampled_input_features=self.sampled_input_features['sig_input'], 
                                                          manifold_kernel_amplitude=self.manifold_kernel_amplitude,
                                                          manifold_kernel_lengthscale=self.manifold_kernel_lengthscale,
                                                          gp_kernel_amplitude=self.gp_kernel_amplitude,
                                                          gp_kernel_lengthscale=self.gp_kernel_lengthscale,
                                                          manifold_neighbor_mode=self.manifold_neighbor_mode,
                                                          manifold_kernel_k=self.manifold_kernel_k, 
                                                          manifold_kernel_power=self.manifold_kernel_power, 
                                                          manifold_kernel_noise=self.manifold_kernel_noise, 
                                                          optimal_threshold=True, 
                                                          svc_cache=1000,
                                                          estimator_type='logistic')
                        if  self.warm_cv_params is not None:
                            warm_start_model.fit({'sig_input':self.keras_model.sig_layer.kernel_mat, 'psych_intput':self.keras_model.sig_layer.kernel_mat}, self.l_y_cat_transformed_train_, cv=self.warm_cv_params['cv'], Cs=1., random_cv=self.warm_cv_params['random'], sample_size=self.warm_cv_params['sample_size'])
                            warm_start_c = warm_start_model.c
                        else: 
                            warm_start_model.fit(self.keras_model.sig_layer.kernel_mat, self.l_y_decimal_train_)

                        if self.unlabeled_data_inc_type is not None:
                            warm_a_prime = warm_start_model.estimator.coef_.reshape((-1, 1))
                            warm_a = np.zeros(X_.shape[0]).reshape((-1, 1))
                            warm_a[~warm_start_model.hold_out_idx] = warm_a_prime
                            
                        else:
                            warm_a = warm_start_model.estimator.coef_.reshape((-1, 1))
                        warm_b = warm_start_model.estimator.intercept_
                        warm_alpha = tf.random.truncated_normal(
                                    shape=[self.X_train_['psych_input'].shape[-1], 1], mean=0.0, stddev=.01, dtype=tf.dtypes.float64, seed=2019)
                        warm_beta = np.log(warm_start_c / (1 - warm_start_c))                            
        #                     warm_beta = np.log(c / (1 - c))    

                else:
                    self.keras_model.set_weights(weights)

        #             cold_optimization_count = 1
        #         else:
        #             cold_optimization_count = 1
        #             self.initialize_model(self.little_input_dict=self.little_input_dict, X_train_['sig_input'])

            cold_optimization_count = 1
            failing_budget = 10
            lr = self.lr
            fail_count = 0
    #         reset_psych_layer=True
        except Exception as e:
            raise type(e)(str(e)+'Squaring W is causing error  9111' )
        num_of_attemps = 0
        failed_counter = 0

#         if self.is_fitted:
#             self.number_of_successful_attempts = 10
#         try:
#             try: 
#                 check_is_fitted(self)
#                 self.number_of_successful_attempts = 10
#             except NotFittedError as e:
#                 tf.print("MADE IT!!!!", output_stream=sys.stderr)
#         except Exception as e:
#             raise type(e)(str(e)+'Another shitty thing happende dbeside not gfitting!!' )

#         if self.fit_counts == 5:
#             self.number_of_successful_attempts = 10
        try:
            rerun_counter = 0
            while num_of_attemps - failed_counter < self.max_rerun and (num_of_attemps + failed_counter < self.max_rerun):
                try:
                    if failed_counter > 1:
                        print ("------!!!!!!!!!!!!!!!!!!!!!------------")
                        print ("------!!!!!!!!!!!!!!!!!!!!!------------")
                        print ("--!!!!!--FAILURE MATTERS--!!!!---------")
                        print ("------!!!!!!!!!!!!!!!!!!!!!------------")
                        print ("------!!!!!!!!!!!!!!!!!!!!!------------")
                        print ("--------------------------------------")
                    if failed_counter > 3:
                        print ("------?????????????????????------------")
                        print ("------?????????????????????------------")
                        print ("--????--EVEN THIS MUCH FAILURE--?????---------")
                        print ("------?????????????????????------------")
                        print ("------?????????????????????------------")
                        print ("--------------------------------------")

                    if rerun_counter > 1:
                        print ("------&&&&&&&&&&&&&&&&&&&&------------")
                        print ("------&&&&&&&&&&&&&&&&&&&&------------")
                        print ("--&&&&--EVEN THIS MUCH FAILURE--&&&&--")
                        print ("------&&&&&&&&&&&&&&&&&&&&------------")
                        print ("------&&&&&&&&&&&&&&&&&&&&------------")
                        print ("--------------------------------------")
                        
                    print ("--------------------------------------")
                    print ("--------------------------------------")
                    print ("--------------------------------------")
                    print ("--------------------------------------")
                    print ("--------------------------------------")
                    if hasattr(self, 'keras_model'):
                        del (self.keras_model.sig_layer)
                        del (self.keras_model.psych_layer)
                        del (self.keras_model)   
                        tf.keras.backend.clear_session()
                        self.fresh_opt_initial_point = True
                    if self.fresh_opt_initial_point:
                        self.initialize_model(reinitialize=True)
                        
                    if warm_start:
                        tf.print ("Warmstart assigining...", self.keras_model.sig_layer, output_stream=sys.stderr)
                        self.keras_model.sig_layer.alpha.assign(warm_a)
                        self.keras_model.sig_layer.beta.assign(warm_b[0])
                        if self.psych_alpha is None:
                            self.keras_model.psych_layer.alpha.assign(warm_alpha)
                        if self.psych_beta is None:
                            self.keras_model.psych_layer.beta.assign(warm_beta)
                        if self.g_prime is None and not self.is_SPM:
                            self.keras_model.psych_layer.g_prime.assign(-5)
                        if self.l_prime is None and not self.is_SPM:
                            self.keras_model.psych_layer.l_prime.assign(-5)
    
                        #                     try:
#                         self.keras_model(self.little_input_dict)
#                     except Exception as e:
#                         raise type(e)(str(e) + "the error is caused by feeding self.little_input_dict")
                        

#                     print ("The best weights are:", self.alternating_best_weights)
        #             self.keras_model.sig_layer.set_weights(self.alternating_sig_layer_best_weights)
        #             if fail_count > 1 or reset_psych_layer:        
        #                 self.keras_model.sig_layer.set_weights(self.alternating_sig_layer_best_weights)
        #                 if not reset_psych_layer:
        #                     fail_count = 0
        #                 reset_psych_layer = True
        #             else:
#                     self.keras_model.set_weights(self.initial_weights)
                    

                    if self.constrained_optimization == 'barrier':
                        self.keras_model.barrier_C.assign(self.barrier_initial_C * shrinking_factor ** (i + 1))
                    elif self.constrained_optimization == 'MoM':
                        self.keras_model.c.assign(self.keras_model.c * shrinking_factor)

                    # here we set sig_a_init_ and sig_b_init_ and psych_alpha_init_ and psych_beta_nit_ to the initialization values 
                    # to make sure the next time fitting will run from the best parameters! 
#                     self.set_params(**{'sig_a_init': self.sig_ainit_})
#                     self.set_params(**{'sig_b_init': self.sig_binit_})
#                     self.set_params(**{'psych_alpha_init': self.psych_alphainit_})
#                     self.set_params(**{'psych_beta_init': self.psych_betainit_})
                    rerun_counter += 1
                except Exception as e:
                    if hasattr(e, 'message'):
                        tf.print ("THE POOPY message is:", e.message)
                        raise type(e)(e.message + 'problem in initialization of keras_model')                    
                    else:
                        raise type(e)(str(e)+'problem in initialization of keras_model' )
                try:

                    history, parameter_reporter, lookback_reduce_lr = self.keras_fit(X_train=self.X_train_, l_y_decimal_train=self.l_y_decimal_train_, validation_data=validation_data_, lr=lr, tol=tol)


                except Exception as e:
                    if hasattr(e, 'message'):
                        tf.print ("THE POOPY message is:", e.message)
                        with open('/home/scratch/nshajari/psych_model/tmp/'+str(np.random.randn())+'.txt', 'w') as f:
                            f.writeline(str(self.get_params()))
                            
                        raise type(e)(e.message + 'keras_fit causes trouble')                    
                    else:
                        raise type(e)(str(e)+'keras_fit causes trouble' )


                try:
                    if lookback_reduce_lr.best is not None and self.alternating_best_loss > lookback_reduce_lr.best: 
                        tf.print("The best weights are updating... prior lowest loss:", self.alternating_best_loss, "current lowest loss:", lookback_reduce_lr.best, output_stream=sys.stderr)
                        self.parameter_reporter = parameter_reporter
                        self.history = history
                        self.descend_epochs = lookback_reduce_lr.descend_epochs
                        self.alternating_best_weights = lookback_reduce_lr.best_weights
                #            self.alternating_sig_layer_best_weights = self.keras_model.sig_layer.get_weights()
                        self.alternating_best_loss = lookback_reduce_lr.best
                        self.keras_model.set_weights(self.alternating_best_weights)
                        
                    # FOLLOWING LINE IS ADDDED TO RESTART MINIMIZATION WHEN THE LOSS GOES BELOW ZERO AND DEFIES THE
                    # LOGIC OF THE OLDER MULTIPLE FITTING PARTIALLY. THIS IS A TEMPORARY FIX!!!! BECAUSE
                    # IT WILL BREAK THE LOOP IF ONLY MINIMIZATION AT FIRST ATTEMPT RETURNS A POSITIVE VALUE
                    if self.alternating_best_loss > 0:
                        break
                except Exception as e:
                    if hasattr(e, 'message'):
                        tf.print ("THE POOPY message is:", e.message)
                        raise type(e)(e.message + 'trouble is after keras_fit')                    
                    else:
                        raise type(e)(str(e)+'trouble is after keras_fit' )

                try:
                    num_of_attemps += 1

#                     self.set_params(**{'sig_a': None, 
#                                      'sig_b': None, 
#                                      'psych_alpha': None,
#                                      'psych_beta': None,
#                                      })
#                     self.sig_a =  None
#                     self.sig_b = None
#                     self.psych_alpha =  None
#                     self.psych_beta = None

#     #                 if not self.is_SPM:
#                     self.set_params(**{'g_prime': None,
#                                  'l_prime': None,
#                                  'psych_gamma': None,
#                                  'psych_lambda': None,
#                                       })
#                     self.g_prime = None
#                     self.l_prime =  None
#                     self.psych_gamma = None
#                     self.psych_lambda = None

#                     if self.gp_kernel_lengthscale_trainable: 
#                         self.set_params(gp_kernel_lengthscale=None)
#                         self.gp_kernel_lengthscale=None
#                     if self.manifold_kernel_lengthscale_trainable: 
#                         self.set_params(manifold_kernel_lengthscale=None)
#                         self.manifold_kernel_lengthscale=None

                    if self.keras_model.reset_training or ('converged' in self.optimization_results.keys() and not   self.optimization_results['converged']):
                        failed_counter += 1
                except Exception as e:
                    raise type(e)(str(e) + 'setting the params of PsychM to none is making trouble')
                    


                    
                    
                try:
                    self.alternate_history_list += [history]
                    self.alternate_param_reporter_list += [parameter_reporter]
                    self.alternate_descend_epoch_list += [lookback_reduce_lr.descend_epochs]
                except Exception as e:
                    raise type(e)(str(e)+'trouble is after the while loop in setting the fitted parameters' )
                    if hasattr(e, 'message'):
                        raise type(e)(e.message + 'trouble is after the while loop in setting the fitted parameters' )   
                    else:
                        raise type(e)(str(e)+'trouble is after the while loop in setting the fitted parameters' ) 
                        
                if self.optimization_results['converged'] and num_of_attemps > self.min_rerun:
                    break
#             try:
#             if failed_counter == 0:
            self.keras_model_weights = self.keras_model.get_weights()
            self.refresh_params()
                

#             except Exception as e:
#                 raise type(e)(str(e) + 'refreshing params not working')
    #         try:
    #             param_dict = {'encoder':self.encoder,
    #             'is_SPM': self.is_SPM,
    #             'kernel': self.kernel,
    #             'sig_reg_coeff': self.sig_reg_coeff,
    #             'sig_reg_penalty': self.sig_reg_penalty,
    #             'manifold_kernel_type' : self.manifold_kernel_type,
    #             'epochs' : self.epochs,
    #             'psych_reg_coeff' : self.psych_reg_coeff,
    #             'psych_reg_penalty' : self.psych_reg_penalty,
    # #             'metrics' : self.metrics,
    #             'batch_size' : self.batch_size,
    #             'workers' : self.workers,
    #             'verbose' : self.verbose,
    #             'child' : self.child,
    #             'optimizer' : self.optimizer_name,
    #             'loss_type' : self.loss_type,
    #             'use_multiprocessing' : self.use_multiprocessing,
    #             'keras_model_initialized' : self.keras_model_initialized,
    #             'calibrate' : self.calibrate,
    #             'warm_f_g_prime_init' : self.warm_f_g_prime_init,
    #             'warm_f_l_prime_init' : self.warm_f_l_prime_init,
    #             'f_g_prime_init' : self.f_g_prime_init,
    #             'f_l_prime_init' : self.f_l_prime_init,
    #             'alternate_training' : self.alternate_training,
    #             'constrained_optimization' : self.constrained_optimization,
    #             'barrier_initial_C' : self.barrier_initial_C,
    #             'optimizer_params_dict' : self.optimizer_dict,
    #             'warm_cv_params' : self.warm_cv_params,
    #             'gp_kernel_lengthscale' : self.gp_kernel_lengthscale,
    #             'gp_kernel_lengthscale_trainable' : self.gp_kernel_lengthscale_trainable,
    #             'gp_kernel_amplitude' : self.gp_kernel_amplitude,
    #             'manifold_kernel_lengthscale': self.manifold_kernel_lengthscale,
    #             'manifold_kernel_lengthscale_trainable': self.manifold_kernel_lengthscale_trainable,
    #             'manifold_kernel_amplitude': self.manifold_kernel_amplitude,
    #             'manifold_kernel_amplitude_trainable': self.manifold_kernel_amplitude_trainable,
    #             'noise_sigma' : self.noise_sigma,
    #             'lr_reduce_min_delta' : self.lr_reduce_min_delta,
    #             'end_training_min_delta' : self.end_training_min_delta,
    #             'dropout_rate' : self.dropout_rate,
    #             'manifold_connectivity' : self.manifold_connectivity,
    #             'manifold_kernel_type' : self.manifold_kernel_type,
    #             'gp_kernel_type' : self.gp_kernel_type,
    #             'lbo_temperature' : self.lbo_temperature,
    #             'manifold_kernel_noise' : self.manifold_kernel_noise,
    #             'manifold_kernel_power' : self.manifold_kernel_power,
    #             'manifold_kernel_normed' : self.manifold_kernel_normed,

    #             'g_prime' : self.g_prime,
    #             'l_prime' : self.l_prime,
    #             'psych_alpha' : self.psych_alpha,
    #             'psych_beta' : self.psych_beta,
    #             'sig_a' : self.sig_a,
    #             'sig_b' : self.sig_b,
    #             'psych_lambda' : self.psych_lambda,
    #             'psych_gamma' : self.psych_gamma,
    #             'alternate_history_list': self.alternate_history_list, 
    #             'alternate_param_reporter_list': self.alternate_param_reporter_list,
    #             'alternate_descend_epoch_list': self.alternate_descend_epoch_list,
    #             'parameter_reporter': self.parameter_reporter,
    #             'sampled_input_features['sig_input']': self.sampled_input_features['sig_input']}
    #     #         self.set_params(**param_dict)

    #         except:
    #             raise TypeError("WE ARE FUCKED RIGHT HERE! 4")


        #         if beta_scores[0]>.8 and beta_scores[1]>0.8:
        #             file_name = 'test_psych_model'
        #             file_name = file_name+'_'+str(len(glob('/home/scratch/nshajari/psych_model/test_psych_models/' + file_name+'*'))+1)
        #             self.keras_model.save('/home/scratch/nshajari/psych_model/test_psych_models/' + file_name +'_model.pkl')
        #             with open('/home/scratch/nshajari/psych_model/test_psych_models/' + file_name + '.pkl', 'wb') as f:
        #                 pickle.dump(param_dict, f)
            try:
#                 if hasattr(self.keras_model, 'kernel'):
#                     del(self.keras_model.kernel)
                from pathlib import Path
#                 beta_score = logs.get('fbeta_score')
#                 if beta_score is not None:
#                     if beta_score[0] > .8 and beta_score[1] > 0.8:
#                 if self.history.get('lpu_f1_score_for_y') is not None and np.max(self.history['lpu_f1_score_for_y'])> 1.9:
#                     import dill as pickle
#                     path = "/home/scratch/nshajari/psych_model/test/"+str(id(self))+'_model'
#                     Path(path).mkdir(parents=True, exist_ok=True)
#                     self.keras_model.save(path+'/keras_model')
#                     with open (path+'/model_info.pkl', 'wb') as f:
#                         pickle.dump([self.X_train_, self.l_y_decimal_train_, self.history, self.parameter_reporter, self.get_params()], f)
#                     tf.print ('***********************------------------------------********************')
                
#                 del(self.kernel)
                
                # linear_out = self.kernel_mat @ self.sig_a_ + self.sig_b_
            
                # output = tf.math.sigmoid(linear_out)
                # W = np.diag(output * (1 - output))
                # self.final_kernel = self.kernel_mat @ self.I_plus_L_K_D_D_inv + 1./ W
                # self.final_kernel_inv = np.linalg.inv(self.final_kernel)
        
                del(self.keras_model)
                tf.keras.backend.clear_session()
                del(lookback_reduce_lr)
#                 self.func_and_grad_generator.model = None
#                 del(self.func_and_grad_generator)
#                 self.func_and_grad_generator = None
                self.fitting = False
                self.keras_model = None
                self.child = None
            except Exception as e:
                if hasattr(e, 'message'):
                    raise type(e)(e.message+'trouble is really in the final step??' )
                else:
                    raise type(e)(str(e)+'trouble is really in the final step??' )
        except Exception as e:
            raise type(e)(str(e)+'Squaring W is causing error 777' )
#         self.lpu_f1_scorer.model = None
        self.fit_counts += 1
        self.is_fitted = True
#         self.set_params(is_fitted=True)
        self.optimization_results_ = self.optimization_results
#         print("KERNEL IS:", self.kernel)
        double_optimize = False
        best_model = self
        if double_optimize:
            params_dict = self.get_params()
            python_psychm = PythonPsychM()
    #         python_psychm_2 = PythonPsychM()

            python_psychm_dict = python_psychm.get_params()
            key_list = []
            for key in params_dict.keys():
                if key not in python_psychm_dict.keys():
                    key_list.append(key)
            for key in key_list:
                params_dict.pop(key)

            python_psychm.set_params(**params_dict)
    #         python_psychm_2.set_params(**params_dict)
            loss = -1
            python_psychm.fit(X=X, y=y, warm_start=warm_start)
    #         python_psychm_2.fit(X=X, y=y, warm_start=False)

            if python_psychm.final_loss < self.alternating_best_loss:
                best_model = python_psychm
    #         elif python_psychm_2.final_loss > 0 and python_psychm_2.final_loss < self.alternating_best_loss:
    #             best_model = python_psychm_2

#     #         if self.alternating_best_loss < 0:
#     #             with open('/home/scratch/nshajari/kernels/kernel_'+str(random_num)+'.pkl', 'wb') as f:
#     #                 pickle.dump({'gp_kernel_1': python_psychm.kernel_mat, 'manifold_kernel_1': python_psychm.Lap_mat,}, f)

#     #             raise ValueError("The values of minimization cannot be negative. python_psychm_2.final_loss:" + 'python_psychm.final_loss:'+str(python_psychm.final_loss) + 'with PsychM loss:'+str(self.alternating_best_loss))

            if best_model == self:
                self.final_loss = self.alternating_best_loss
                self.final_success = None
            else:
                self.final_loss = best_model.final_loss
                self.final_success = best_model.final_success

            print ("!!!WE ARE LITERALLY AT THE END OF FIT FUNCTION!!! with final losses:", python_psychm.final_loss, "and main loss:", self.alternating_best_loss)
        else:
            print ("!!!WE ARE LITERALLY AT THE END OF FIT FUNCTION!!! with final losses and main loss:", self.alternating_best_loss)

            
        
        self.set_params(**best_model.get_params())
        if self.calibrate:
            X_cal_image_ = dict()
            X_cal_image_['sig_input'] = self.predict_prob_y_given_x(X_cal_)
            X_cal_image_['psych_input'] = self.predict_prob_l_given_y_x(X_cal_)
            X_cal_image_['idx'] = 0
            self.calib_psychm = PsychM(
                    psych_reg_coeff = 0.,
                    loss_type='expanded_using_softmax_mathematica', 
                    optimizer_dict = {'optimizer_name':'lbfgs', 'learning_rate': .001},
                    encoder=self.encoder,
                is_SPM=True, manifold_kernel_type=None, sampled_input_features=X_cal_image_, gp_kernel_amplitude=1., manifold_kernel_amplitude=0., gp_kernel_type='linear', gp_kernel_lengthscale=1.)
            self.calib_psychm.fit(X_cal_image_, y_cal)
        if self.optimal_threshold:
            self.optimal_threshold = find_optimal_threshold((self.l_y_decimal_train_/2).astype(int).ravel(), self.predict_prob_y_given_x(X).ravel())
        else:
            self.optimal_threshold = 0.5
        
        return self
    
    def pack_dataset(self, psych_input, sig_input, l_y_decimal, idx):
        return {'psych_input': psych_input, 'sig_input':sig_input, 'idx': idx}, l_y_decimal
    
    def create_keras_dataset(self, X, l_y_decimal, batch_size=64, shuffle_size=1000):
        seed_value = 2021
        l_y_decimal_transposed = l_y_decimal.reshape((-1, 1))
        if batch_size == -1:
            batch_size = len(l_y_decimal)
#             tf_dataset_idx = tf.data.Dataset.from_tensors(np.arange(tf.keras.backend.shape(l_y_decimal_transposed.astype(np.float64))[0]).astype(np.float64).reshape((1, -1)).squeeze())
#             tf_dataset_l_y_decimal= tf.data.Dataset.from_tensors(l_y_decimal_transposed)
#         else:
#             tf_dataset_idx = tf.data.Dataset.from_tensor_slices(np.arange(tf.keras.backend.shape(l_y_decimal_transposed.astype(np.float64))[0]).astype(np.float64).reshape((1, -1)).squeeze())\
#             .shuffle(shuffle_size, seed=seed_value).batch(batch_size)
#             tf_dataset_l_y_decimal = tf.data.Dataset.from_tensor_slices(l_y_decimal_transposed.astype(np.float64)).shuffle(shuffle_size, seed=seed_value).batch(batch_size)

#         tf_dataset_idx = tf.data.Dataset.from_tensor_slices(np.arange(tf.keras.backend.shape(l_y_cat_transposed)[0]).astype(np.float64).reshape((1, -1)).squeeze()).batch(batch_size)   
#         tf_dataset_l_y_cat = tf.data.Dataset.from_tensor_slices(l_y_cat_transposed).batch(batch_size)   


        psych_input = X['psych_input'].astype(np.float64)
    
        sig_input = X['sig_input'].astype(np.float64)
        dataset_idx = X['idx']

#         if batch_size == -1:
#             batch_size = 
#             tf_dataset_idx = tf.data.Dataset.from_tensors(dataset_idx)
#             tf_dataset_l_y_decimal= tf.data.Dataset.from_tensors(l_y_decimal_transposed)
#         else:
        tf_dataset_idx = tf.data.Dataset\
                                        .from_tensor_slices(dataset_idx)\
                                        .shuffle(shuffle_size, seed=seed_value)\
                                        .batch(batch_size)
        
        tf_dataset_l_y_decimal = tf.data.Dataset\
                                                .from_tensor_slices(l_y_decimal_transposed.astype(np.float64))\
                                                .shuffle(shuffle_size, seed=seed_value).batch(batch_size)

        self.psych_dim = tf.keras.backend.shape(psych_input)[-1]
        self.sig_dim = tf.keras.backend.shape(sig_input)[-1]
#         print ("sig_input shape:", sig_input.shape)
        if type(psych_input).__name__ == 'csr_matrix':
            if batch_size == -1:                
                tf_dataset_psych_input = tf.data.Dataset.\
                                from_tensors(tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(psych_input)))
            else:
                tf_dataset_psych_input = tf.data.Dataset.\
                                from_tensor_slices(tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(psych_input)))\
                                .shuffle(shuffle_size, seed=seed_value).batch(batch_size)
        else:
            if batch_size == -1:
                tf_dataset_psych_input = tf.data.Dataset.\
                                        from_tensors(psych_input)
            else: 
                tf_dataset_psych_input = tf.data.Dataset.\
                                        from_tensor_slices(psych_input).\
                                        shuffle(shuffle_size, seed=seed_value).batch(batch_size)            

        if type(sig_input).__name__ == 'csr_matrix':
            if batch_size == -1:
                tf_dataset_sig_input = tf.data.Dataset.\
                                from_tensors(tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(sig_input)))
            else:
                tf_dataset_sig_input = tf.data.Dataset.\
                                from_tensor_slices(tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(sig_input)))\
                                .shuffle(shuffle_size, seed=seed_value).batch(batch_size)
                
        else:
            if batch_size == -1:
                tf_dataset_sig_input = tf.data.Dataset\
                                    .from_tensors(sig_input)
            else:
                tf_dataset_sig_input = tf.data.Dataset\
                                    .from_tensor_slices(sig_input)\
                                    .shuffle(shuffle_size, seed=seed_value).batch(batch_size)            
                
        tf_dataset_iterable = tf.data.Dataset\
                            .zip((tf_dataset_psych_input, tf_dataset_sig_input, tf_dataset_l_y_decimal, tf_dataset_idx))
        
        return tf_dataset_iterable\
                .map(self.pack_dataset)
                                                                      
#     def unify_as_dict(self, X_psych, X_sig):
#         return {'psych_input': X_psych, 'sig_input': X_sig}
    
#     def __copy__(self):
#         from tf.keras.models import clone_model
#         copied_model = PsychM(**self.get_params())
#         copied_model.keras_model = clone_model(self.keras_model) 
#         return new

    
#     def psych_F(self, X):
#         return tf.math.sigmoid(X.dot(self.psych_alpha) + self.psych_beta)

#     def psychometric_func(self, X):
#         return self.psych_gamma + (1. - self.psych_gamma - self.psych_lambda) * self.psych_F(X)


    def correct_sign(self, psych_alpha):
        final_dot_product = np.einsum('ij, ij->', psych_alpha, self.sig_a)
        return psych_alpha * np.sign(final_dot_product)
        
    def calculate_W(self, psych_out, sig_out):
        """ THE REAL CALCULATION OF W, i.e. 
        -\Delta\Delta p(l=1|X) very similar to 
        "Gaussian Processes for Machine Learning" by Rassmussen et. al. 2006, p. 42
        
        W_{ii} := \frac{\partial^2p(y_i|X_i)}{\partial f_i^2}  - \frac{\left[ \frac{\partial^2p(y_i|X_i)}{\partial f_i^2}p(l_i=1|\theta, X_i)\left(1-p(l_i=1|X_i, \theta)p(y_i=1|X_i)\right) + \left(\frac{\partial p(y_i=1|X_i)}{\partial f_i}\right)^2 p(l_i=1|X_i, \theta)^2\right]}{(1-p(l_i=1|X_i, \theta)p(y_i=1|X_i))^2}
        
        
        part_1 = \frac{\partial^2p(y_i|X_i)}{\partial f_i^2}
        part_2 = -\frac{\partial^2p(y_i|X_i)}{\partial f_i^2} p(l_i=1|\theta, X_i) \left(1-p(l_i=1|X_i, \theta)p(y_i=1|X_i)\right) 
        part_3 = \left(\frac{\partial p(y_i=1|X_i)}{\partial f_i}\right)^2 p(l_i=1|X_i, \theta)^2
        part_4 =(1-p(l_i=1|X_i, \theta)p(y_i=1|X_i))
        
        W_{ii} = part_1 - [part_2 + part_3] / part_4
                
                """

        
        
        partial_2_p_y_x_partial_f_2 = sig_out * (1 - sig_out) * (1 - 2 * sig_out)
        h_x = 1 - sig_out * psych_out 
        partial_p_y_x_partial_f_sq = sig_out ** 2 * (1 - sig_out) ** 2
        
        part_1 = partial_2_p_y_x_partial_f_2 
        part_2 = -partial_2_p_y_x_partial_f_2 * psych_out * h_x
        part_3 = partial_p_y_x_partial_f_sq * psych_out ** 2
        part_4 = h_x ** 2
        
        W_ii = part_1 - [part_2 + part_3] / part_4
        
        
        
    def predict_prob_y_given_x(self, X, laplacian=False, raw_output=False, return_calibrated_scores=False):
        if 'idx' not in X.keys():
            X['idx'] = np.arange(len(X['sig_input']))
        
#         if self.fitting:
#             psych_alpha_, psych_beta_, g_prime_, l_prime_ =   [item for item in self.keras_model.psych_layer.get_weights()]
#             gp_kernel_lengthscale_, sig_a_, sig_b_, _, _=  [item for item in self.keras_model.sig_layer.get_weights()]
#         else:
#         sig_a_ = self.sig_a
#         sig_b_ = self.sig_b
#         psych_alpha_ = self.psych_alpha
#         psych_beta_ = self.psych_beta
        if self.is_SPM and (self.manifold_kernel_type is None or (self.gp_kernel_type == 'linear' and self.g_prime < 0.)) and np.linalg.norm(self.psych_alpha) >= np.linalg.norm(self.sig_a) and not self.SPM_switch_call:
            self.SPM_switch_call = True
            output = self.predict_prob_l_given_y_x(X, return_calibrated_scores=return_calibrated_scores)
            self.SPM_switch_call = False
            return output
        X_ = X['sig_input']

        
        
        if self.unlabeled_data_inc_type is not None and ('LBO' in self.manifold_kernel_type.lower() or 'laplacian' in self.manifold_kernel_type.lower()):
            try:
                kernel_mat = self.kernel(X_, self.sampled_input_features['sig_input'])
                if hasattr(kernel_mat, 'numpy'):
                    X_ = kernel_mat.numpy()
                else:
                    X_ = kernel_mat
                    
            except TypeError as e:
                raise type(e)(str(e) + " the kernel is:")
        # linear_out = X_.dot(self.sig_a) + self.sig_b
        # sigmoid_out = tf.math.sigmoid(linear_out).numpy()
            
        # if hasattr(sigmoid_out, 'numpy'):
            # sigmoid_out = tf.math.sigmoid(sigmoid_out).numpy()
            
        # k_x_star_x_star = self.kernel(X['sig_input'], X['sig_input'])
        # k_tilde_x_star_x_star = k_x_star_x_star - X_ @ self.final_kernel_inv @ self.manifold_mat @ X_.T
        
        if hasattr(self, 'with_laplace_method') and self.with_laplace_method:
            sig_a_samples = self.params_sample[:, -self.sig_a.shape[0]-1:-1]
            sig_b_samples = self.params_sample[:, -1]
        else:
            sig_a_samples = np.tile(self.sig_a.ravel(),  (self.params_sample.shape[0], 1))
            sig_b_samples = np.tile(self.sig_b.ravel(), (self.params_sample.shape[0], 1))
        
        
        # k_x_x = k_x_x @ (np.eye(k_x_x.shape[0]) + k_x_x @ self.manifold_mat)
        
        # V_q = np.diag(np.diag(k_x_x - X_ @ np.linalg.inv(k_tilde_x_star_x_star +  @ X_.T))
        linear_out_samples = X_.dot(sig_a_samples.T) + sig_b_samples.ravel()
        sig_out_samples = tf.math.sigmoid(linear_out_samples).numpy()
        sig_out_samples[sig_out_samples > 1-EPSILON] = 1 - EPSILON
        sig_out_samples[sig_out_samples < EPSILON] = EPSILON
        if not raw_output:
            sig_out_samples = tf.math.sigmoid(linear_out_samples).numpy().mean(axis=1).reshape((-1, 1))
        np.nan_to_num(sig_out_samples, copy=False, nan=EPSILON)
        # sigmoid_out = tf.math.sigmoid(linear_out_samples).numpy().mean(axis=1).reshape((-1, 1))
        
        if return_calibrated_scores:
            X_cal_ = {'sig_input': sig_out_samples}
            sig_out_samples = self.calib_psychm.predict_prob_y_given_x(X_cal_)
            
            
        return sig_out_samples
        
        
        # return expit(f_hat_sample).mean(axis=0).reshape((-1, 1))
        # return sigmoid_out
        
        
### THIS COMMENTED PART HAS TO DO WITH INCORPORATING SIGN FUCNTION TO MAKE SURE DOT PRODUC BETWEEN VECTORS IS POSITIVE!
#         if not self.is_SPM:
#             if self.SPM_switch_call:
#                 sig_a_ = self.correct_sign(sig_a_)
#             linear_out = X.dot(sig_a_) + sig_b_              
#         else:
#             linear_out = X.dot(sig_a_) + sig_b_    
    #     test_keras_model = PsychMKeras(is_SPM=self.is_SPM, sig_reg_coeff=self.sig_reg_coeff, manifold_kernel_type = self.manifold_kernel_type, sig_reg_penalty=self.sig_reg_penalty, psych_reg_coeff = self.psych_reg_coeff, psych_reg_penalty=self.psych_reg_penalty, parent=self, loss_type=self.loss_type, constrained_optimization=self.constrained_optimization, barrier_initial_C=self.barrier_initial_C, GP_input_type='index', sampled_input_features['sig_input']=self.sampled_input_features['sig_input'], gp_kernel_lengthscale=self.gp_kernel_lengthscale, gp_kernel_lengthscale_trainable=self.gp_kernel_lengthscale_trainable, noise_sigma=self.noise_sigma, dropout_rate=self.dropout_rate, manifold_kernel_lengthscale=self.manifold_kernel_lengthscale, manifold_kernel_lengthscale_trainable=self.manifold_kernel_lengthscale_trainable, manifold_kernel_amplitude=self.manifold_kernel_amplitude, gp_kernel_amplitude=self.gp_kernel_amplitude, manifold_kernel_amplitude_trainable=self.manifold_kernel_amplitude_trainable,
    # manifold_connectivity=self.manifold_connectivity, manifold_kernel_type=self.manifold_kernel_type, gp_kernel_type=self.gp_kernel_type, lbo_temperature=self.lbo_temperature, manifold_kernel_noise=self.manifold_kernel_noise, manifold_kernel_power=self.manifold_kernel_power, manifold_kernel_normed=self.manifold_kernel_normed, manifold_kernel_k=self.manifold_kernel_k, g_prime=self.g_prime, l_prime=self.l_prime, sig_a=self.sig_a, sig_b=self.sig_b, psych_alpha=self.psych_alpha, psych_beta=self.psych_beta, freeze_psychometric_original=self.freeze_psychometric_original, training_size=self.training_size)
    
#         test_keras_model(self.little_input_dict)#), "SSL TYPE:", self.manifold_kernel_type)

#         y_pred = test_keras_model(X)
#         test_keras_model.set_weights(self.keras_model_weights)
# #         tf.print ("THE FUCKED UP OUTPUT IS:", test_keras_model(X))
#         y_pred = test_keras_model(X)
#         batch_size = X_.shape[0]
#         psych_layer_input_shape = test_keras_model.psych_layer_input_shape
#         sig_layer_input_shape = test_keras_model.sig_layer_input_shape
#         g_prime, l_prime, psych_linear, psych_alpha, psych_beta,\
#                     sig_linear, sig_a, sig_b,\
#                     manifold_norm, f_k_norm = tf.cast(y_pred[0, 0], tf.float64), tf.cast(y_pred[1, 0], tf.float64), tf.cast(y_pred[2: batch_size + 2, :], tf.float64), tf.cast(y_pred[batch_size + 2:psych_layer_input_shape + batch_size + 2, :], tf.float64), tf.cast(y_pred[psych_layer_input_shape + batch_size + 2, 0], tf.float64),\
#                     tf.cast(y_pred[psych_layer_input_shape + batch_size + 3:psych_layer_input_shape + batch_size * 2 + 3, :], tf.float64), tf.cast(y_pred[psych_layer_input_shape + batch_size * 2 + 3: psych_layer_input_shape + sig_layer_input_shape + batch_size * 2 + 3, :], tf.float64), tf.cast(y_pred[psych_layer_input_shape + sig_layer_input_shape + batch_size * 2 + 3, :], tf.float64), tf.cast(y_pred[psych_layer_input_shape + sig_layer_input_shape + batch_size * 2 + 4, :], tf.float64), tf.cast(y_pred[psych_layer_input_shape + sig_layer_input_shape + batch_size * 2 + 5, :], tf.float64)             
#         linear_out = sig_linear
                
#         if f_k_norm > 10:
#             f_k_norm = 10
#         if f_k_norm < 0:
#             f_k_norm = 0
#         else:
#         f_k_norm =  tf.math.exp(-f_k_norm)
        # nominator = -linear_out - f_k_norm
        # denominator = -linear_out
#         if nominator > 30:
#             nominator = 30
#         if demonimator > 30:
#             demonimator = 30
#         if nominator < -30:
#             nominator = -30
#         if demonimator < -30:
#             demonimator = -30
            
        # output = np.true_divide(np.exp(nominator) , np.exp(denominator) + 1)
        # output = tf.math.divide(tf.math.exp(nominator), tf.math.exp(denominator) + 1).numpy()
    
        # output = np.nan_to_num(output, nan=1e-10)
        # output[output>1. - 1e-10] = 1-1e-10
        # output = tf.math.sigmoid(linear_out)
        # if hasattr(output, 'numpy'):
            # output = output.numpy()
#         tf.print ("SH***************************T:", tf.shape(f_k_norm), output_stream=sys.stderr)
        
        
        # return sigmoid_out.mean(axis=1).reshape((-1, 1))



    def predict_prob_l_given_y_x(self, X, raw_output=False, return_calibrated_scores=False):       
        # IMPLEMENTATION WITHOUT TENSORFLOW
#         if self.fitting:pow
#             psych_alpha_, psych_beta_, g_prime_, l_prime_ =   [item for item in self.keras_model.psych_layer.get_weights()]
#             gp_kernel_lengthscale_, sig_a_, sig_b_, _, _=  [item for item in self.keras_model.sig_layer.get_weights()]
#             psych_gamma_, psych_lambda_ = self.keras_model.g_l_prime_to_gamma_lambda_transformer(g_prime_init, l_prime_init)
# #             psych_gamma_, psych_lambda_ = [item.numpy() for item  in self.keras_model.psych_layer.get_gamma_lambda()]
#         else:
#         sig_a_ = self.sig_a
#         sig_b_ = self.sig_b
#         psych_alpha_ = self.psych_alpha
#         psych_beta_ = self.psych_beta
#         psych_gamma_ = self.psych_gamma
#         psych_lambda_ = self.psych_lambda
        if self.is_SPM and (self.manifold_kernel_type is None or (self.gp_kernel_type == 'linear' and self.g_prime < 0.)) and np.linalg.norm(self.psych_alpha) >= np.linalg.norm(self.sig_a) and not self.SPM_switch_call:
            self.SPM_switch_call = True
            output = self.predict_prob_y_given_x(X, return_calibrated_scores=return_calibrated_scores)
            self.SPM_switch_call = False
            return output
#         X = X[self.psych_vec][tuple(sorted((k, v) for k, v in self.psych_vec_params.items()))]
        X_ = X['psych_input']
        if hasattr(X_, 'values'):
            X_ = X_.values
        
#         psych_alpha_ = self.correct_sign(psych_alpha_)
        
#         linear_output = X_.dot(self.psych_alpha) + self.psych_beta
#         if hasattr(self.psych_gamma, 'numpy'):
# #             output = tf.sigmoid(linear_output) * (1 - self.psych_gamma.numpy() - self.psych_lambda.numpy()) + self.psych_gamma.numpy()
#             output = tf.math.sigmoid(linear_output) * (1 - self.psych_gamma.numpy() - self.psych_lambda.numpy()) + self.psych_gamma.numpy()
#         else:
# #             output = tf.sigmoid(linear_output) * (1 - self.psych_gamma - self.psych_lambda) + self.psych_gamma
#             output = tf.math.sigmoid(linear_output) * (1 - self.psych_gamma - self.psych_lambda) + self.psych_gamma
        if self.freeze_psych_alpha:
            psych_alpha_samples =  np.tile(self.psych_alpha.ravel(),  (self.params_sample.shape[0], 1))
            pointer = 0
        else:
            pointer = self.psych_alpha.shape[0]
            psych_alpha_samples = self.params_sample[:, :pointer]        
        if hasattr(self, 'with_laplace_method') and self.with_laplace_method:
            psych_beta_samples = self.params_sample[:, pointer]
            g_prime_samples = self.params_sample[:,pointer+1]            
            l_prime_samples = self.params_sample[:, pointer+2]
        else:
            psych_alpha_samples = np.tile(self.psych_alpha.ravel(),  (self.params_sample.shape[0], 1))
            psych_beta_samples = np.tile(self.psych_beta.ravel(), (self.params_sample.shape[0], 1))
            g_prime_samples = np.ones_like(self.params_sample[:, pointer+1]) * self.g_prime
            l_prime_samples = np.ones_like(self.params_sample[:, pointer+1]) * self.l_prime
    
        if self.is_SPM:
            g_prime_samples = np.ones_like(self.params_sample[:, pointer+1]) * self.g_prime
            l_prime_samples = np.ones_like(self.params_sample[:, pointer+1]) * self.l_prime

        if self.freeze_psychometric_original:
            psych_alpha_samples = np.tile(self.psych_alpha.ravel(),  (self.params_sample.shape[0], 1))
            psych_beta_samples = np.tile(self.psych_beta.ravel(), (self.params_sample.shape[0], 1))

        psych_linear_out_samples = X_.dot(psych_alpha_samples.T) + psych_beta_samples.ravel()
        gamma_samples, lambda_samples = g_l_prime_to_gamma_lambda_transformer(g_prime_samples, l_prime_samples)
        try:
            psych_linear_out_samples[psych_linear_out_samples > 200] = 200
            psych_linear_out_samples[psych_linear_out_samples <-200] = -200
            
            psych_out_samples = tf.math.sigmoid(psych_linear_out_samples).numpy()
        except LinAlgError:
            print ("BAD THINGS ARE HAPPENNIG")
            np.nan_to_num(psych_out_samples, copy=Fals)
        # psych_out_samples
        psych_out_samples = (1. - gamma_samples - lambda_samples).numpy().ravel() * psych_out_samples + gamma_samples.numpy().ravel()
        # sigmoid_out = tf.math.sigmoid(linear_out_samples).numpy().mean(axis=1).reshape((-1, 1))
        
        np.nan_to_num(psych_out_samples, copy=False, nan=EPSILON)

        psych_out_samples[psych_out_samples > 1-EPSILON] = 1 - EPSILON
        psych_out_samples[psych_out_samples < EPSILON] = EPSILON
        
        if not raw_output:
            psych_out_samples =  psych_out_samples.mean(axis=1).reshape((-1, 1))
        if return_calibrated_scores:
            X_cal_ = {'sig_input': psych_out_samples}
            psych_out_samples = self.calib_psychm.predict_prob_l_given_y_x(X_cal_)
        return psych_out_samples
        

        
    
    
### THIS COMMENTED PART HAS TO DO WITH INCORPORATING SIGN FUCNTION TO MAKE SURE DOT PRODUC BETWEEN VECTORS IS POSITIVE!
#         if not self.is_SPM:
#             psych_alpha_ = self.correct_sign(psych_alpha_)
#             linear_output = X.dot(psych_alpha_) + psych_beta_
#             output = tf.math.sigmoid(linear_output) * (1 - psych_gamma_ - psych_lambda_) + psych_gamma_
#         else:
#             if not self.SPM_switch_call:
#                 psych_alpha_ = self.correct_sign(psych_alpha_)
#             linear_output = X.dot(psych_alpha_) + psych_beta_
#             output = tf.math.sigmoid(linear_output)
        if hasattr(output, 'numpy'):
            output = output.numpy()
        return output



    def predict_l_given_y_x(self, X, threshold=0.5):
        return self.predict_prob_l_given_y_x(X)>=threshold


    def predict_y_given_x(self, X):
        return self.predict_prob_y_given_x(X)>=self.optimal_threshold


    def predict_proba(self, X, return_calibrated_scores=False):
        try: 
            p_y_given_X_arr = self.predict_prob_y_given_x(X, raw_output=True, return_calibrated_scores=return_calibrated_scores)
        except Exception as e:            
            raise type(e)(str(e) + 'predict_prob_y_given_x is the ass') 
        try:
            p_l_given_x_y_arr = self.predict_prob_l_given_y_x(X, raw_output=True, return_calibrated_scores=return_calibrated_scores)
        except Exception as e:            
            raise type(e)(str(e) + 'predict_prob_l_given_y_x is the ass psych_alpha shape:' + str(self.psych_alpha.shape) + 'and the shape of sig_a:'+ str(self.sig_a.shape)) 
        output = np.multiply(p_y_given_X_arr, p_l_given_x_y_arr).mean(axis=1).reshape((-1, 1))
        np.nan_to_num(output, copy=False, nan=EPSILON)
        return output

#         X_ = X['sig_input']
#         if hasattr(X_, 'values'):
#             X_ = X_.values
#         sig_a_samples = self.params_sample[:, -self.sig_a.shape[0]-1:-1]
#         sig_b_samples = self.params_sample[:, -1]
#         if self.manifold_kernel_type is not None and ('LBO' in self.manifold_kernel_type or 'laplacian' in self.manifold_kernel_type):
#             try:
#                 kernel_mat = self.kernel(X_, self.sampled_input_features['sig_input'])
#                 if hasattr(kernel_mat, 'numpy'):
#                     X_ = kernel_mat.numpy()
#                 else:
#                     X_ = kernel_mat
                    
#             except TypeError as e:
#                 raise type(e)(str(e) + " the kernel is:")

#         linear_out_samples = X_.dot(sig_a_samples.T) + sig_b_samples.ravel()
#         # sigmoid_out = tf.math.sigmoid(linear_out_samples).numpy().mean(axis=1).reshape((-1, 1))
#         sig_out_samples = tf.math.sigmoid(linear_out_samples).numpy()

#         X_ = X['psych_input']
        
#         if hasattr(X_, 'values'):
#             X_ = X_.values
#         try: 
#             psych_alpha_samples = self.params_sample[:, :self.psych_alpha.shape[0]]        
#             psych_beta_samples = self.params_sample[:, self.psych_alpha.shape[0]]
#             g_prime_samples = self.params_sample[:, self.psych_alpha.shape[0]+1]
#             l_prime_samples = self.params_sample[:, self.psych_alpha.shape[0]+2]

#             psych_linear_out_samples = X_.dot(psych_alpha_samples.T) + psych_beta_samples.ravel()
#             gamma_samples, lambda_samples = g_l_prime_to_gamma_lambda_transformer(g_prime_samples, l_prime_samples)

#             psych_out_samples = (1. - gamma_samples - lambda_samples).numpy().ravel() * tf.math.sigmoid(psych_linear_out_samples).numpy() + gamma_samples.numpy().ravel()

#             # sigmoid_out = tf.math.sigmoid(linear_out_samples).numpy().mean(axis=1).reshape((-1, 1))
#             # sig_out_samples[sig_out_samples < EPSILON] = EPSILON
#             # sig_out_samples[sig_out_samples > 1 - EPSILON] = 1 - EPSILON
#             # log_prod = np.log(sig_out_samples) + np.log(psych_out_samples)
#             # prod = np.exp(log_prod)
#             sig_out_samples[sig_out_samples > 1-EPSILON] = 1 - EPSILON
#             psych_out_samples[psych_out_samples > 1-EPSILON] = 1 - EPSILON
#             prod = tf.math.multiply(sig_out_samples, psych_out_samples).numpy()
#             # prod[prod > 1-EPSILON] = 1 - EPSILON
#             # prod[prod < EPSILON] = EPSILON
            
        #     output =  prod.mean(axis=1).reshape((-1, 1))
        # except Exception as e:
        #     raise type(e)(str(e)+"the sampling part from Bayesian analysis is fucking up things")
        # return output
        
        # try:
        #     output = np.multiply(p_y_given_X_arr, p_l_given_x_y_arr)
        #     assert (p_y_given_X_arr.shape == p_l_given_x_y_arr.shape)
        # except Exception as e:            
        #     raise type(e)(str(e) + 'fuck multiplicaytion is the ass') 
            
            
        # return output

    def predict(self, X, threshold = 0.5):
        return self.predict_proba(X)>=threshold
    
#     def transform(self, X, y=None):
#         return self.predict_proba(X)
    
    


########################################################################
####### CURRENTLY NOT WORKING @tf.function DECORATION OF THE ABOVE FUNCTIONS TO 
#######  SPEED UP THINGS. I NEED TO READ MORE ABOUT THIS
##############################################################################
#     @tf.function
#     def get_all_params_(self):
#         psych_alpha_= self.keras_model.psych_layer.alpha
#         psych_beta_= self.keras_model.psych_layer.beta
#         psych_gamma_, psych_lambda_ = self.keras_model.psych_layer.get_gamma_lambda()
#         sig_a_ = self.keras_model.sig_layer.alpha
#         sig_b_ = self.keras_model.sig.layer.beta
#         return psych_alpha_, psych_beta_, g_prime_, l_prime_, sig_a_, sig_b_
#     @tf.function
#     def predict_prob_y_given_x_(self, X):
#         psych_alpha, psych_beta, g_prime_, l_prime_, sig_a_, sig_b_ = get_all_params_()
#         if self.is_SPM and np.linalg.norm(psych_alpha_) >= np.linalg.norm(sig_a_) and not self.SPM_switch_call:
#             self.SPM_switch_call = True
#             output = self.predict_prob_l_given_y_x_(X)
#             self.SPM_switch_call = False
#             return output
#         X = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#         linear_out = X.dot(sig_a_) + sig_b_
#         return tf.math.sigmoid(linear_out)

#     @tf.function
#     def predict_prob_l_given_y_x_(self, X):       
#         psych_alpha, psych_beta, g_prime_, l_prime_, sig_a_, sig_b_ = get_all_params_()
#         if self.is_SPM and np.linalg.norm(psych_alpha_) >= np.linalg.norm(sig_a_) and not self.SPM_switch_call:
#             self.SPM_switch_call = True
#             output = self.predict_prob_y_given_x_(X)
#             self.SPM_switch_call = False
#             return output
#         X = X[self.psych_vec][tuple(sorted((k, v) for k, v in self.psych_vec_params.items()))]
#         output = tf.math.sigmoid(X.dot(psych_alpha_) + psych_beta_) * (1 - psych_gamma_ - psych_lambda_) + psych_gamma_
#         return output


    
#     @tf.function
#     def predict_l_given_y_x_(self, X):
#         return self.predict_prob_l_given_y_x_(X)>=0.5

#     @tf.function
#     def predict_y_given_x_(self, X):
#         return self.predict_prob_y_given_x_(X)>=0.5

#     @tf.function
#     def predict_proba_(self, X):
#         p_y_given_X_arr = self.predict_prob_y_given_x_(X)
#         p_l_given_x_y_arr = self.predict_prob_l_given_y_x_(X)
#         assert (p_y_given_X_arr.shape == p_l_given_x_y_arr.shape)
#         return np.multiply(p_y_given_X_arr, p_l_given_x_y_arr)

#     @tf.function
#     def predict_(self, X, threshold = 0.5):
#         return self.predict_proba_(X)>=threshold
        