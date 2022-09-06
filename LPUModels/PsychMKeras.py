# import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, accuracy_score
# tf.keras.backend.set_floatx('float64')
# from func_lib import convert_sparse_matrix_to_sparse_tensor
import numpy as np
import sys
sys.path.append('/home/scratch/nshajari/psych_model/')
sys.path.append('/home/scratch/nshajari/psych_model/utils/')
sys.path.append('/home/scratch/nshajari/psych_model/LPUModels/')

import logging
import sys
import logging
import os
from utils.func_lib import expanded_using_softmax_mathematica
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
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

import tensorflow_probability as tfp
from GPLayer import GPLayer
from PsychometricLayer import PsychometricLayer
#



# from PsychM import gamma_lambda_to_g_l_prime_transformer, g_l_prime_to_gamma_lambda_transformer



@tf.function
def scatter_update(idx, shape):
    return tf.scatter_nd(tf.reshape(tf.cast(idx, tf.int32), shape=(-1, 1)), tf.ones(shape=tf.keras.backend.shape(idx), dtype=tf.float64), shape=shape)

class PsychMKeras(tf.keras.Model):    
    def __init__(self, is_SPM=None, name='psych_model_keras', sig_reg_penalty=None,sig_reg_coeff=None, 
                 psych_reg_penalty=None, psych_reg_coeff= None, parent=None, epochs=None, loss_type=None, calibrate=None, 
                 constrained_optimization=None, barrier_initial_C=None, manifold_kernel_type=None, GP_SSL_L=None, 
                 sampled_input_features=None, GP_input_type=None, gp_kernel_lengthscale=None, 
                 gp_kernel_lengthscale_trainable=None, gp_kernel_amplitude=None, manifold_kernel_lengthscale=None, 
                 manifold_kernel_lengthscale_trainable=None, manifold_kernel_amplitude=None, 
                 manifold_kernel_amplitude_trainable=None, noise_sigma=None, l_prime=None, g_prime=None, 
                 dropout_rate=None, manifold_neighbor_mode=None,  gp_kernel_type=None, 
                 lbo_temperature=None, manifold_kernel_noise=None, manifold_kernel_power=None, 
                 manifold_kernel_normed=None, psych_alpha=None, psych_beta=None, sig_a=None, sig_b=None, 
                 manifold_kernel_k=None, freeze_psychometric_original=None, 
                 training_size=None, invert_manifold_mat_explicitly=None,
                 I_plus_KM_inv_M_calc_method=None, I_plus_KM_inv_M_using_factorization=None, 
                 I_plus_KM_inv_M_using_expm_acting=None, I_plus_KM_inv_M_using_eigsh=None, freeze_psych_alpha=None,
                 unlabeled_data_inc_type=None,
                 *args, **kwargs):
        super().__init__(name, *args, **kwargs)        
        try:
#             super(PsychMKeras, self).__init__(name=name, *args, **kwargs)        
            self.is_SPM = is_SPM
            self.freeze_psych_alpha = freeze_psych_alpha
            self.training_size = training_size
            self.noise_sigma = noise_sigma
            self.manifold_kernel_noise = manifold_kernel_noise 
            self.manifold_kernel_power = manifold_kernel_power
            self.manifold_kernel_k = manifold_kernel_k
            self.manifold_kernel_normed = manifold_kernel_normed
            self.lbo_temperature = lbo_temperature
            self.dropout_rate = dropout_rate
            self.psych_alpha = psych_alpha
            self.psych_beta = psych_beta
            self.sig_a = sig_a
            self.sig_b = sig_b
            self.g_prime = g_prime
            self.l_prime = l_prime
            self.freeze_psychometric_original = freeze_psychometric_original
            self.unlabeled_data_inc_type = unlabeled_data_inc_type

            self.GP_input_type = GP_input_type
            self.manifold_neighbor_mode = manifold_neighbor_mode
            self.manifold_kernel_type = manifold_kernel_type
            if self.unlabeled_data_inc_type is not None:
                if 'laplacian' in self.manifold_kernel_type.lower():
                    self.manifold_kernel_type = 'laplacian'
                elif 'lbo' in self.manifold_kernel_type.lower():
                    self.manifold_kernel_type = 'LBO'
                else:
                    raise NotImplementedError("Kernel for Laplacian needs to be either the heat kernel or Laplacian matrix itself")
            self.manifold_kernel_type = manifold_kernel_type
            self.gp_kernel_type = gp_kernel_type
            self.manifold_kernel_lengthscale = manifold_kernel_lengthscale
            self.manifold_kernel_lengthscale_trainable = manifold_kernel_lengthscale_trainable
            self.manifold_kernel_amplitude = manifold_kernel_amplitude
            self.manifold_kernel_amplitude_trainable = manifold_kernel_amplitude_trainable
            self.parent = parent
            self.loss_type = loss_type
            self.sampled_input_features = sampled_input_features
            self.sig_reg_coeff = sig_reg_coeff
            self.psych_reg_coeff = psych_reg_coeff
            self.sig_reg_penalty = sig_reg_penalty
            self.psych_reg_penalty = psych_reg_penalty
            self.calibrate = calibrate
            self.constrained_optimization = constrained_optimization
            self.barrier_initial_C = barrier_initial_C
            self.gp_kernel_lengthscale = gp_kernel_lengthscale
            self.gp_kernel_lengthscale_trainable = gp_kernel_lengthscale_trainable
            self.gp_kernel_amplitude = gp_kernel_amplitude
            self.reset_training = False
            self.invert_manifold_mat_explicitly = invert_manifold_mat_explicitly
            self.I_plus_KM_inv_M_calc_method = I_plus_KM_inv_M_calc_method
            self.I_plus_KM_inv_M_using_factorization = I_plus_KM_inv_M_using_factorization
            self.I_plus_KM_inv_M_using_expm_acting = I_plus_KM_inv_M_using_expm_acting
            self.I_plus_KM_inv_M_using_eigsh = I_plus_KM_inv_M_using_eigsh  
            self.unlabeled_data_inc_type = unlabeled_data_inc_type
            
        except Exception as e:
            if hasattr(e, 'message'):
                raise type(e)(e.message+'Squaring W is causing error 20' )
            else:
                raise type(e)(str(e)+'Squaring W is causing error 20' )
            
#         def get_config(self):
#             return {'model_loss': self.model_loss,
#                    'history':self.history,
#                    'kernel':self.sig_layer.kernel}

#         # There's actually no need to define `from_config` here, since returning
#         # `cls(**config)` is the default behavior.
#         @classmethod
#         def from_config(cls, config):
#             return cls(**config)



     
        


    def build(self, input_shape):
        self.softmax_layer = tf.keras.layers.Softmax()
        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate, dtype=tf.float64, name='dropout')

        self.psych_layer = PsychometricLayer(is_sigmoid=self.is_SPM, reg_coeff=0., reg_penalty=None, g_prime=self.g_prime, l_prime=self.l_prime, alpha=self.psych_alpha, beta=self.psych_beta, freeze_psychometric_original=self.freeze_psychometric_original, freeze_psych_alpha=self.freeze_psych_alpha)
        
        # if  self.manifold_kernel_type is None or 'SSL' not in self.manifold_kernel_type:
        #     self.sig_layer = PsychometricLayer(1,  reg_coeff = 0., reg_penalty=None, g_prime=self.g_prime, l_prime=self.l_prime, alpha=self.sig_a, beta=self.sig_b)
        # else:
            # reg_penalty is passed as none because regularization is not directly appleid to alpha but rather to alpha^T L^{-1} alpha
        self.sig_layer = GPLayer(reg_coeff=0., reg_penalty=None, sampled_input_features=self.sampled_input_features, input_type=self.GP_input_type, gp_kernel_lengthscale=self.gp_kernel_lengthscale, gp_kernel_amplitude=self.gp_kernel_amplitude, gp_kernel_lengthscale_trainable=self.gp_kernel_lengthscale_trainable,  noise_sigma=self.noise_sigma, manifold_kernel_lengthscale=self.manifold_kernel_lengthscale, manifold_kernel_lengthscale_trainable=self.manifold_kernel_lengthscale_trainable, manifold_kernel_amplitude=self.manifold_kernel_amplitude, manifold_kernel_amplitude_trainable=self.manifold_kernel_amplitude_trainable, manifold_neighbor_mode=self.manifold_neighbor_mode, manifold_kernel_type=self.manifold_kernel_type, gp_kernel_type=self.gp_kernel_type, lbo_temperature=self.lbo_temperature, manifold_kernel_noise=self.manifold_kernel_noise, manifold_kernel_power=self.manifold_kernel_power, manifold_kernel_k=self.manifold_kernel_k, manifold_kernel_normed=self.manifold_kernel_normed, alpha=self.sig_a, beta=self.sig_b, training_size=self.training_size, 
                                 invert_manifold_mat_explicitly=self.invert_manifold_mat_explicitly,
                                 I_plus_KM_inv_M_calc_method=self.I_plus_KM_inv_M_calc_method,
                                 I_plus_KM_inv_M_using_factorization=self.I_plus_KM_inv_M_using_factorization,
                                 I_plus_KM_inv_M_using_expm_acting=self.I_plus_KM_inv_M_using_expm_acting,
                                 I_plus_KM_inv_M_using_eigsh=self.I_plus_KM_inv_M_using_eigsh,unlabeled_data_inc_type=self.unlabeled_data_inc_type)
        dtype = tf.float64
        
        if self.constrained_optimization == 'barrier':
            dtype = tf.float64
            barrier_C_initializer = lambda x, dtype: tf.constant(self.barrier_initial_C, dtype=dtype)            
            self.barrier_C = self.add_weight("barrier_C", shape=[1, 1], trainable=False, initializer=barrier_C_initializer)
            
        elif self.constrained_optimization == 'MoM':
            mu_initializer = lambda x, dtype: tf.constant(1., dtype=dtype)            
            c_initializer = lambda x, dtype: tf.constant(10., dtype=dtype)            
            self.c = self.add_weight("c", shape=[1, 1], trainable=False, initializer=c_initializer)            
            self.mu_gamma = self.add_weight("mu_gamma", shape=[1, 1], trainable=True, initializer=mu_initializer)
            self.mu_lambda= self.add_weight("mu_lambda", shape=[1, 1], trainable=True, initializer=mu_initializer)
            self.mu_gamma_p_lambda = self.add_weight("mu_gamma_p_alpha", shape=[1, 1], trainable=True, initializer=mu_initializer)
            self.mu_alpha_a = self.add_weight("mu_alpha_a", shape=[1, 1], trainable=True, initializer=mu_initializer)
                
        if self.manifold_kernel_type == 'min_entropy':
            lambda_c_initializer = lambda x, dtype: tf.constant(1., dtype=dtype)            
            self.lambda_c = self.add_weight("lambda_c", shape=[1, 1], trainable=True, initializer=lambda_c_initializer, constraint=tf.keras.constraints.NonNeg())

            
        
    def call(self, input_tensor_dict, training=False):
#         print ("PsychMKeras is seeing input_tensor_dict:", input_tensor_dict)
        try:
            try:
                sig_input = dict()
                psych_input = dict()
    #             psych_input['psych_input'] = self.dropout_layer(input_tensor_dict['psych_input'], training=training)
    #             psych_input['idx'] = input_tensor_dict['idx']
    #             sig_input['sig_input'] = self.dropout_layer(input_tensor_dict['sig_input'], training=training)
                psych_input['psych_input'] = input_tensor_dict['psych_input']
                sig_input['sig_input'] = input_tensor_dict['sig_input']
            except Exception as e:
                raise type(e)(str(e) + " setting up input valeus faisl!")

            try:
                psych_input['idx'] = input_tensor_dict['idx']
                sig_input['idx'] = input_tensor_dict['idx']
                if len(input_tensor_dict['idx']) < len(input_tensor_dict['sig_input']):
                    psych_input['idx'] = input_tensor_dict['idx'] = tf.cast(np.arange(len(input_tensor_dict['psych_input'])), tf.int32) 
                    sig_input['idx'] = input_tensor_dict['idx'] = tf.cast(np.arange(len(input_tensor_dict['sig_input'])), tf.int32)
                
            except Exception as e:
                logging = tf.get_logger()
                # logging.warning("masker_cast is not set. defaulting to set the mask to be all indices...")            
                psych_input['idx'] = input_tensor_dict['idx'] = tf.cast(np.arange(len(input_tensor_dict['psych_input'])), tf.int32)
                sig_input['idx'] = input_tensor_dict['idx'] = tf.cast(np.arange(len(input_tensor_dict['sig_input'])), tf.int32)
                # raise type(e)(str(e) + "key error is having a problem!!!")
                
                
    

            num_of_batch = sig_input['sig_input'].shape[0]
        
        
            # psych_linear, g_prime, l_prime, psych_alpha, psych_beta = self.psych_layer(psych_input, training=training)
            psych_linear = self.psych_layer(psych_input, training=training)
            if self.unlabeled_data_inc_type is None:
                # sig_linear, _, _, sig_a, sig_b, manifold_norm, f_k_norm = self.sig_layer(sig_input, training=training)
                sig_linear = self.sig_layer(sig_input, training=training)
                f_k_norm = tf.cast(tf.zeros([psych_linear.shape[0], 1]), tf.float64)
            else:
                try:
                    sig_linear = self.sig_layer(sig_input, training)
                    if training:
                        f_k_norm = self.sig_layer.f_k_norm(self.sig_layer._alpha, self.kernel_mat)
                    else:
                        f_k_norm = tf.cast(tf.zeros([psych_linear.shape[0], 1]), tf.float64)
                        
                    # gp_kernel_lengthscale, manifold_kernel_lengthscale, sig_linear, sig_a, sig_b, manifold_norm, f_k_norm = self.sig_layer(sig_input, training=training)
                except Exception as e:
                    raise type(e)(str(e) + "calling GPLayer in the call of of PsychMKeras fails")
            # manifold_norm = tf.cast(tf.zeros_like(manifold_norm), tf.float64)
        except Exception as e:
            raise type(e)(str(e) + 'Squaring W is causing error 26')

        
        return tf.concat([psych_linear, sig_linear, f_k_norm], axis=0)

#     def compute_output_shape(self, input_shape):
#         input_shape = input_shape.shape().as_list()
#         return input_shape[0], self.num_units

        
        
        

    @tf.function
    def form_loss(self, l, g_prime, l_prime, psych_linear,sig_linear):
        """Implemented based on https://github.com/tensorflow/tensorflow/blob/9cd34cf019801aef9fc79dc2c6c02a204f4e86dd/tensorflow/python/ops/nn_impl.py#L127"""
        g_prime = tf.cast(g_prime, tf.float64)
        l_prime = tf.cast(l_prime, tf.float64)
        sig_linear = tf.cast(sig_linear, tf.float64)
        psych_linear = tf.cast(psych_linear, tf.float64)
        abs_g_prime = tf.math.abs(g_prime)
        abs_l_prime = tf.math.abs(l_prime)
        abs_psych_linear = tf.math.abs(psych_linear)
        abs_sig_linear = tf.math.abs(sig_linear)
        
        
        return tf.math.multiply(l, tf.math.log(abs_g_prime * (1.+ tf.exp(-abs_psych_linear)) + tf.exp(-tf.nn.relu(-psych_linear)))) -\
                    tf.math.multiply(l, tf.nn.relu(-sig_linear)) +\
                    tf.math.multiply(1. - l, tf.math.log(tf.math.multiply((1.+abs_g_prime+abs_l_prime)*(1. + tf.exp(-abs_sig_linear)) - tf.exp(-tf.nn.relu(-sig_linear)) * abs_g_prime, (1. + tf.exp(-abs_psych_linear))) - 1. / (1. + abs_g_prime + abs_l_prime) * \
                                                         tf.exp(-tf.nn.relu(-sig_linear) - tf.nn.relu(-psych_linear)))) - \
                                                         tf.math.log(1. + tf.exp(-abs_psych_linear)) - tf.math.log(1. + tf.exp(-abs_sig_linear)) - tf.math.log(1. + abs_g_prime + abs_l_prime) 

    @tf.function
    def form_loss_2(self, l, g_prime, l_prime, psych_linear, sig_linear):
        """Implemented based on https://github.com/tensorflow/tensorflow/blob/9cd34cf019801aef9fc79dc2c6c02a204f4e86dd/tensorflow/python/ops/nn_impl.py#L127"""
        g_prime = tf.cast(g_prime, tf.float64)
        l_prime = tf.cast(l_prime, tf.float64)
        sig_linear = tf.cast(sig_linear, tf.float64)
        psych_linear = tf.cast(psych_linear, tf.float64)
        abs_g_prime = tf.math.abs(g_prime)
        abs_l_prime = tf.math.abs(l_prime)
        abs_psych_linear = tf.math.abs(psych_linear)
        abs_sig_linear = tf.math.abs(sig_linear)
        
        
        log_1_input = abs_g_prime * (1.+ tf.exp(-abs_psych_linear)) + tf.exp(-tf.nn.relu(-psych_linear))
        log_1 = tf.math.log(log_1_input)

        log_2_1_inner = tf.where(abs_sig_linear > 1000, tf.cast(1., tf.float64), (1. + tf.exp(-abs_sig_linear)))
        log_2_2_inner = tf.where(abs_sig_linear > 1000, tf.cast(0., tf.float64), tf.exp(-tf.nn.relu(-sig_linear)))
        log_2_3_inner = tf.where(abs_psych_linear > 1000, tf.cast(1., tf.float64), (1. + tf.exp(-abs_psych_linear)))
        log_2_4_inner = tf.where(abs_psych_linear > 1000, tf.cast(0., tf.float64), tf.exp(-tf.nn.relu(-psych_linear)))

        
        log_2_input = tf.math.multiply((1.+abs_g_prime+abs_l_prime) * log_2_1_inner - log_2_2_inner * abs_g_prime, log_2_3_inner) - 1. / (1. + abs_g_prime + abs_l_prime) * \
                                                         log_2_2_inner * log_2_4_inner
        
        log_2 = tf.math.log(log_2_input)
        
        log_3_input = 1. + tf.exp(-abs_psych_linear)
        log_3 = tf.where(abs_psych_linear>1000, tf.cast(0., tf.float64), tf.math.log(log_3_input))

        log_4_input = 1. + tf.exp(-abs_sig_linear)
        log_4 = tf.where(abs_sig_linear>1000, tf.cast(0., tf.float64), tf.math.log(log_4_input))

        log_5_input = 1. + abs_g_prime + abs_l_prime
        log_5 = tf.math.log(log_5_input)

        return tf.math.multiply(l, log_1) -\
                    tf.math.multiply(l, tf.nn.relu(-sig_linear)) +\
                    tf.math.multiply(1. - l, log_2) - log_3 - log_4 - log_5
    @tf.function
    def SPM_form_loss(self, l, g_prime, l_prime, psych_linear,sig_linear):
        abs_g_prime = tf.math.abs(g_prime)
        abs_l_prime = tf.math.abs(l_prime)
        abs_psych_linear = tf.math.abs(psych_linear)
        abs_sig_linear = tf.math.abs(sig_linear)
        return tf.math.multiply(l, -tf.nn.relu(-psych_linear)) -\
                                    tf.math.multiply(l, tf.nn.relu(-sig_linear)) +\
        tf.math.multiply(1 - l, tf.math.log(tf.abs(tf.math.multiply((1 + tf.exp(-abs_sig_linear)),
                                             (1 + tf.exp(-abs_psych_linear))) -\
                                             tf.exp(-tf.nn.relu(-sig_linear) - tf.nn.relu(-psych_linear))))) -\
                                             tf.math.log(tf.abs(1 + tf.exp(-abs_psych_linear))) - tf.math.log(tf.abs(1 + tf.exp(-abs_sig_linear)))
        
    
    
    def final_loss(self, l_y_cat_true, y_pred):
        return self.model_loss(l_y_cat_true, y_pred)
    
    def lbfgs_model_loss(overall_theta, X, l_y_cat_true):
        pass
    
    @tf.function
    def model_loss(self, l_y_decimal_true, y_pred):
        try:
            if tf.rank(y_pred) == 1:
                y_pred = tf.reshape(y_pred, (-1, 1))
            batch_size = tf.keras.backend.shape(l_y_decimal_true)[0]

            g_prime = self.psych_layer._g_prime
            l_prime = self.psych_layer._l_prime
            psych_linear = tf.cast(y_pred[:batch_size, :], tf.float64) 
            sig_linear = tf.cast(y_pred[batch_size: batch_size * 2, :], tf.float64) 
            f_k_norm = tf.cast(y_pred[batch_size * 2, 0], tf.float64) 
            sig_a = self.sig_layer._alpha
            sig_b = self.sig_layer._beta
            psych_alpha = self.psych_layer._alpha
            psych_beta = self.psych_layer._beta
                                   
        except Exception as e:
            raise type(e)(str(e) + "calling call() in keras_model casues error" + str(y_pred.shape) + ', ' +str(y_pred))
        try:
            if hasattr(l_y_decimal_true, 'values'):
                l_y_decimal_true = l_y_decimal_true.values
#             l_y_decimal_true = l_y_decimal_true.reshape((1, -1))[0]
            l_true, _ = tf.cast(tf.cast(l_y_decimal_true / 2, tf.int32), tf.float64), tf.math.mod(l_y_decimal_true, 2)
    #         y = tf.cast(l_true, tf.float64)


            if g_prime is None:
                raise NotImplementedError(" FUCK ALL THE NONES!")
            if psych_linear is None:
                raise NotImplementedError(" FUCK ALL THE NONES!")
            try:
                if self.loss_type == 'standard':
                    psych_gamma, psych_lambda = g_l_prime_to_gamma_lambda_transformer(g_prime, l_prime)
                    abs_psych_linear = tf.abs(psych_linear)
                    abs_sig_linear = tf.abs(sig_linear)
                    neg_relu_psych_linear = -tf.nn.relu(-psych_linear)
                    neg_relu_sig_linear = -tf.nn.relu(-sig_linear)
                    main_loss = tf.math.multiply(l_true, tf.math.log(psych_gamma * (1 + tf.exp(-abs_psych_linear)) +\
                                                                (1 - psych_gamma - psych_lambda) * tf.exp(-tf.nn.relu(-psych_linear)))) -\
                                                                tf.math.multiply(l_true, tf.nn.relu(-sig_linear)) +\
                    tf.math.multiply(1 - l_true, tf.math.log(tf.math.multiply(1 + tf.exp(-abs_sig_linear) - tf.exp(-tf.nn.relu(-sig_linear)) * psych_gamma,
                                                         (1 + tf.exp(-abs_psych_linear))) -
                                                         (1 - psych_gamma - psych_lambda) * \
                                                         tf.exp(-tf.nn.relu(-sig_linear) - tf.nn.relu(-psych_linear)))) - \
                                                         tf.math.log(1 + tf.exp(-abs_psych_linear)) - tf.math.log(1 + tf.exp(-abs_sig_linear))



                elif self.loss_type == 'basic':
                    main_loss = tf.math.multiply(l_true, tf.math.log(psych_gamma+(1-psych_gamma-psych_lambda) * tf.math.sigmoid(psych_linear)) + tf.math.log(tf.math.sigmoid(sig_linear))) + tf.math.multiply(1 - l_true, tf.math.log(1 - tf.math.multiply(psych_gamma+(1-psych_gamma-psych_lambda) * tf.math.sigmoid(psych_linear), tf.math.sigmoid(sig_linear))))

                elif self.loss_type == 'log_sum_exp':
                    M = 1.01 * (tf.math.reduce_max([-psych_linear,  -sig_linear], axis=0) - tf.math.log(psych_lambda))
                    main_loss = tf.math.multiply(l_true, tf.math.log((1 + tf.exp(-abs_psych_linear)) * psych_gamma + (1 - psych_gamma - psych_lambda) * tf.exp(-tf.nn.relu(-psych_linear)))) + tf.math.multiply(1 - l_true, M + tf.math.log(tf.exp(-psych_linear - M) * (1 - psych_gamma) + tf.exp(-sig_linear - M) + tf.exp(-psych_linear - sig_linear - M) + tf.exp(tf.math.log(psych_lambda) - M))) - \
                    tf.math.log(1 + tf.exp(-abs_psych_linear)) - tf.math.log(1 + tf.exp(-abs_sig_linear)) - \
                    tf.math.multiply(1 - l_true, tf.nn.relu(-psych_linear)) - tf.nn.relu(-sig_linear)# end of seventh term                             

                elif self.loss_type == 'classic':
                    g_prime = tf.cast(g_prime, tf.float64)
                    l_prime = tf.cast(l_prime, tf.float64)
                    sig_linear = tf.cast(sig_linear, tf.float64)
                    psych_linear = tf.cast(psych_linear, tf.float64)
                    abs_g_prime = tf.math.abs(g_prime)
                    abs_l_prime = tf.math.abs(l_prime)
                    sq_sum = abs_g_prime + abs_l_prime
                    abs_psych_linear = tf.math.abs(psych_linear)
                    abs_sig_linear = tf.math.abs(sig_linear)
                    M = tf.math.reduce_max([-psych_linear,  -sig_linear, -psych_linear  -sig_linear])
                    M = tf.expand_dims(tf.math.reduce_max([M,  (tf.math.log(abs_l_prime) - tf.math.log(sq_sum + 1))[0]]), axis=0)
                    M_2 = tf.math.reduce_max(-psych_linear) 
                    M_2 = tf.expand_dims(tf.math.reduce_max([M_2, (tf.math.log(1. + abs_g_prime) - tf.math.log(1. + sq_sum))[0]]), axis=0)

                    main_loss = tf.math.multiply(l_true, M_2 + tf.math.log(tf.exp(-psych_linear - M_2) * abs_g_prime + tf.exp(tf.math.log(1. + abs_g_prime) - tf.math.log(1. + sq_sum) - M_2))) +\
                        tf.math.multiply(1 - l_true, M + tf.math.log(tf.exp(-psych_linear - M) * (1 - abs_g_prime) + tf.exp(-sig_linear - M) + tf.exp(-psych_linear - sig_linear - M) + tf.exp(tf.math.log(abs_l_prime) - tf.math.log(sq_sum + 1) - M))) - \
                        tf.math.log(1 + tf.exp(-abs_psych_linear)) - tf.math.log(1 + tf.exp(-abs_sig_linear)) \
                    -tf.nn.relu(-psych_linear) -\
                        tf.nn.relu(-sig_linear)

                elif self.loss_type == 'form':
                    main_loss = self.form_loss(l_true, g_prime, l_prime, psych_linear, sig_linear)

                elif self.loss_type == 'SPM_form':
                    main_loss = self.SPM_form_loss(l_true, g_prime, l_prime, psych_linear, sig_linear
                                                           )
                elif self.loss_type == 'form_2':
                    main_loss = self.form_loss_2(l_true, g_prime, l_prime, psych_linear, sig_linear)
                elif self.loss_type == 'expanded_using_softmax':
                    main_loss = expanded_using_softmax(l_true, g_prime, l_prime, psych_linear, sig_linear)
                elif self.loss_type == 'expanded_using_softmax_mathematica':
                    try:
                        # main_loss = expanded_using_softmax_mathematica(l_true, g_prime, l_prime, psych_linear, sig_linear)
                        main_loss = expanded_using_softmax_mathematica(l_true, self.psych_layer._g_prime, self.psych_layer._l_prime, psych_linear, sig_linear)
                    except Exception as e:
                        raise type(e)(str(e) + "softmax_mathematica is raising error")


                if self.manifold_kernel_type == 'min_entropy':
                    ratios = (1 - tf.math.sigmoid(sig_linear)) / (1 - tf.math.sigmoid(sig_linear) * (psych_gamma + (1 - psych_gamma - psych_lambda) * tf.math.sigmoid(psych_linear)))
                    ratios = ratios[l_true==0]
                    ratios = ratios[1e-12 < ratios]
                    ratios = ratios[ratios < 1-1e-12]
                    main_loss += tf.reduce_sum((ratios * tf.math.log(ratios) + (1 - ratios) * tf.math.log(1 - ratios))) * self.lambda_c
                    nll = -tf.reduce_sum(main_loss)

                ############################################################
                # doing laplacian regularization
                ############################################################
                elif self.manifold_kernel_type is not None:

                    nll = -tf.reduce_sum(main_loss, axis=0)
                    nll += f_k_norm #/ self.sig_layer_input_shape # + 1. / tf.math.reduce_std(sig_linear)+ 1. #                 tf.print('nll shape:', tf.shape(nll), output_stream=sys.stderr)
    # / tf.math.reduce_std(psych_linear) #+ 4 * tf.math.log(self.sig_layer.kernel_lengthscale) + 1. / (self.sig_layer.kernel_lengthscale) ** 2

                else:
                    nll = -tf.reduce_sum(main_loss)

            except Exception as e:
                raise type(e)(str(e) + "calculating nll is raising error")
    #         print ("ACTIVATED LOSS TYPE IS:", self.loss_type)
    #         nll = -tf.reduce_sum(main_loss)


            if self.constrained_optimization == 'MoM':
                total_loss = nll + 1. / (2 * self.c) * (tf.square(tf.nn.relu(self.mu_gamma - self.c * psych_gamma)) - tf.square(self.mu_gamma) + 
                                                        tf.square(tf.nn.relu(self.mu_lambda - self.c * psych_lambda)) - tf.square(self.mu_lambda) + 
                                                        tf.square(tf.nn.relu(self.mu_gamma_p_lambda - self.c * (1 - psych_gamma - psych_lambda))) - tf.square(self.mu_gamma_p_lambda)
                                                        + tf.square(tf.nn.relu(self.mu_alpha_a - self.c * tf.tensordot(psych_alpha, sig_a, axes=(0, 0)))) 
                                                        - tf.square(self.mu_alpha_a))

            elif self.constrained_optimization == 'barrier':
                alpha_norm = tf.norm(psych_alpha)
                a_norm = tf.norm(sig_a)
                total_loss = nll - 1. / (self.barrier_C) * tf.math.log(tf.tensordot(psych_alpha, sig_a, axes=(0, 0))/ ((tf.norm(psych_alpha)) * tf.norm(sig_a)))

            else:
                total_loss = nll
    #         gradient = tf.keras.backend.gradients(total_loss, self.trainable_variables)
        except Exception as e:
            raise type(e)(str(e)+'Squaring W is causing error fuck' )

        total_loss = total_loss + tf.linalg.norm(self.psych_layer._alpha, ord=2) ** 2 * self.psych_reg_coeff ** 2 / 2. + self.psych_layer._beta ** 2 / 2.  * 1e-4 + self.psych_layer._g_prime ** 2 / 2.  + self.psych_layer._l_prime ** 2 / 2.  + sig_b ** 2 / 2. * 1e-4
        
        return tf.squeeze(total_loss)