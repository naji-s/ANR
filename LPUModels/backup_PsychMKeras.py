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
                logging.warning("masker_cast is not set. defaulting to set the mask to be all indices...")            
                psych_input['idx'] = input_tensor_dict['idx'] = tf.cast(np.arange(len(input_tensor_dict['psych_input'])), tf.int32)
                sig_input['idx'] = input_tensor_dict['idx'] = tf.cast(np.arange(len(input_tensor_dict['sig_input'])), tf.int32)
                # raise type(e)(str(e) + "key error is having a problem!!!")
                
                
    

            num_of_batch = sig_input['sig_input'].shape[0]
        
        
            psych_linear, g_prime, l_prime, psych_alpha, psych_beta = self.psych_layer(psych_input, training=training)
            if self.manifold_kernel_type is None:
                sig_linear, _, _, sig_a, sig_b, manifold_norm, f_k_norm = self.sig_layer(sig_input, training=training)
                f_k_norm = tf.cast(tf.zeros_like(f_k_norm), tf.float64)
            else:
                try:
                    gp_kernel_lengthscale, manifold_kernel_lengthscale, sig_linear, sig_a, sig_b, manifold_norm, f_k_norm = self.sig_layer(sig_input, training=training)
                except Exception as e:
                    raise type(e)(str(e) + "calling GPLayer in the call of of PsychMKeras fails")
            manifold_norm = tf.cast(tf.zeros_like(manifold_norm), tf.float64)
        except Exception as e:
            raise type(e)(str(e) + 'Squaring W is causing error 26')
        try:
    # 8#             # this is to swithc to feature mode during prediction to actually recalculate 
    #             # kernel matrix
    #             if not training:
    #                 GP_input_type = 'feature'
    #             else:
    #                 GP_input_type = self.GP_input_type
    #             if GP_input_type == 'index':
    #                 sig_linear, sig_a, sig_b, Lap_mat, kernel_mat = self.sig_layer(idx)
    #             elif GP_input_type == 'feature':
    #                 sig_linear, sig_a, sig_b, Lap_mat, Kernel_mat  = self.sig_layer(sig_input)
    #             else:
    #                 raise NotImplementedError

    #             if tf.rank(self.masking_var) == 1:
    #                 sig_linear, sig_a, sig_b = self.sig_layer(tf.reshape(self.masking_var, shape=(1, -1)))
    #             else:
    #                 sig_linear, sig_a, sig_b = self.sig_layer(self.masking_var)


    #         try:
#             psych_gamma, psych_lambda = g_l_prime_to_gamma_lambda_transformer(g_prime, l_prime)#, reparametrization_style=self.reparametrization_style)
    #         except Exception as e:
    #             e.args += ('problem in transforming g_prime l_prime to gamma and lambda')
    #             raise                 

    #         psych_gamma = abs(g_prime) / (abs(g_prime) + abs(l_prime) + 1)
    #         psych_lambda = abs(l_prime) / (abs(g_prime) + abs(l_prime) + 1)
            try:
                self.features_dim = psych_alpha.shape[0]
                if psych_linear is None:
                    raise NotImplementedError(" FUCK psych_linear")
                if sig_linear is None:
                    raise NotImplementedError(" FUCK sig_linear")
            except Exception as e:
                raise type(e)(str(e)+'SERIOYSLY!?')

    #         dot_product = tf.squeeze(tf.tensordot(suspicious_psych_alpha, self.sig_a, axes=(0, 0)))
    #         assert tf.keras.backend.ndim(dot_product) == 0
    #         self.psych_alpha = suspicious_psych_alpha

    #         self.psych_alpha = suspicious_psych_alpha * tf.math.sign(dot_product)
    #         self.psych_linear = self.psych_linear * tf.math.sign(dot_product) +  (tf.math.sign(-dot_product) + 1)  * self.psych_beta

    #         self.psych_linear = tf.identity(self.psych_linear,name="psych_linear")
    #         self.sig_linear = tf.identity(self.sig_linear,name="sig_linear")
    #         self.sig_a = tf.identity(sig_a,name="sig_a")
    #         self.sig_b = tf.identity(self.sig_layer.beta,name="sig_b")
    #         self.psych_alpha = tf.identity(psych_alpha,name="psych_alpha")
    #         self.psych_beta = tf.identity(self.psych_layer.beta,name="psych_beta")
    #         psych_gamma = tf.identity(self.psych_gamma,name="psych_gamma")
    #         psych_lambda = tf.identity(self.psych_lambda,name="psych_lambda")
    #         self.parent.psych_alpha_ = psych_alpha
    #         self.parent.psych_beta_ = self.psych_layer.beta
    #         self.parent.psych_gamma_, self.parent.psych_lambda_ = self.psych_layer.get_gamma_lambda()
    #         self.parent.sig_a_ = sig_a
    #         self.parent.sig_b_ = self.sig_layer.beta
    #         batch_size = self.psych_lambda_out.shape[0]
    #         output = tf.concat([tf.ones_like(self.psych_linear) * self.psych_gamma_out[0] , tf.ones_like(self.psych_linear) * self.psych_lambda_out[0], self.psych_linear, self.sig_linear], axis=1)

    #         output = [[self.psych_gamma_out], [self.psych_lambda_out], self.psych_linear, self.sig_linear]
    #         dim = int(self.psych_linear.shape[-1] - 2)/2)
    #         indices = [0] * 4
    #         indices[0] = 0
    #         indices[1] = 1
    #         indices[2] = list(np.arange(2, dim + 2))
    #         indices[3] = list(np.arange(2 + dim, 2 * dim + 2))
    #         output = tf.dynamic_stitch(indices, output)

    #         return {'psych_gamma':self.psych_gamma_out, 'psych_lambda':self.psych_lambda_out, 'psych_linear': self.psych_linear, 'sig_linear': self.sig_linear}
    #         self.masking_var.assign(tf.zeros(shape=[self.GP_SSL_L.shape[0]], dtype=tf.float64))
    #         idx = tf.keras.backend.flatten(tf.cast(idx, tf.int32))
    #         masking_vector = tf.keras.backend.flatten(tf.where(idx in tf.range(self.GP_SSL_L.shape[0]), self.masking_var, tf.constant(1., dtype=tf.float64)))
    #         tf.print("masking_var:", self.masking_var, tf.keras.backend.shape(self.masking_var), output_stream=sys.stderr)
    #         def extend_tensor(x=None): 
    #             psych_linear_shape = tf.keras.backend.shape(psych_linear)
    # #             tf.print ("extend_tensor stat:",psych_linear_shape, tf.expand_dims(x, axis=0), output_stream=sys.stderr)
    # #             tf.print ("HAHA:", tf.tile(tf.expand_dims(x, axis=0), multiples=[psych_linear_shape[0], 1]))
    #             return tf.tile(tf.expand_dims(x, axis=0), multiples=[psych_linear_shape[0], 1]) if psych_linear_shape[0] is not None else tf.expand_dims(x, axis=0)
    #         Lap_norm = tf.matmul(tf.matmul(tf.transpose(kernel_active_rows), Lap_mat), kernel_active_rows)
    #         f_k_norm = tf.matmul(tf.transpose(kernel_active_rows), kernel_active_rows)
    #         tf.print("THE SHAPE OF PSYCH_LINEAR AND SIG_LINEAR ARE:", tf.keras.backend.shape(psych_linear), tf.keras.backend.shape(sig_linear), output_stream=sys.stderr)
    #         tf.print ("LENGTHS IN CALL:", 
    #                   "training:", training, 
    #                   "psych_gamma_length:", tf.keras.backend.shape(psych_gamma), 
    #                  "psych_lambda_length:", tf.keras.backend.shape(psych_lambda),
    #                  "psych_linear_length:", tf.keras.backend.shape(psych_linear),
    #                   "sig_linear_length:", tf.keras.backend.shape(sig_linear), 
    #                  "sig_a_length:", tf.keras.backend.shape(sig_a),
    #                   "sig_b_length:", tf.keras.backend.shape(sig_b), 
    #                  "Lap_norm_length:", tf.keras.backend.shape(Lap_norm),
    #                   "f_k_norm_length:", tf.keras.backend.shape(f_k_norm), 
    #                   "lenghts in list:", [tf.keras.backend.shape(item) for item in [tf.expand_dims(psych_gamma, axis=0), tf.expand_dims(psych_lambda, axis=0), psych_linear, sig_linear, sig_a, tf.expand_dims(sig_b, axis=0), Lap_norm, f_k_norm]],
    #                   output_stream=sys.stderr)
            self.psych_layer_input_shape = psych_alpha.shape[0]
            self.sig_layer_input_shape = sig_a.shape[0]
    #         tf.print("psych_layer_input_shape:", self.psych_layer_input_shape)
    #         tf.print("sig_layer_input_shape:", self.sig_layer_input_shape)
    #         tf.print("LAP NORM IS:", Lap_norm)
            try:
                try:
                    try:
                        
                        if self.gp_kernel_lengthscale_trainable:
    #                     tf.print([gp_kernel_lengthscale, manifold_kernel_lengthscale, g_prime, l_prime, psych_linear, psych_alpha, psych_beta, sig_linear, sig_a, sig_b, Lap_norm, f_k_norm, alpha_norm, mix_norm], output_stream=sys.stderr)
                            raise NotImplementedError("gp_kernel_lengthscale_trainable IS ALWAYS FALSE!!! WTF??" )
                            return tf.concat([gp_kernel_lengthscale, manifold_kernel_lengthscale, g_prime, l_prime, psych_linear, psych_alpha, psych_beta, sig_linear, sig_a, sig_b, manifold_norm, f_k_norm], axis=0)
                        else:
                            try:
                                # if tf.rank(g_prime) == 0:
                                #     g_prime = tf.expand_dims(g_prime, axis=0)
                                # if tf.rank(l_prime) == 0:
                                #     l_prime = tf.expand_dims(l_prime, axis=0)
                                # if tf.rank(psych_alpha) == 0:
                                #     psych_alpha = tf.expand_dims(psych_alpha, axis=0)
                                # if tf.rank(psych_beta) == 0:
                                #     psych_beta = tf.expand_dims(psych_beta, axis=0)
                                # if tf.rank(sig_a) == 0:
                                #     sig_a = tf.expand_dims(sig_a, axis=0)
                                # if tf.rank(sig_b) == 0:
                                #     sig_b = tf.expand_dims(sig_b, axis=0)
                                try:
                                    g_prime_l_prime_stack = tf.concat([[[g_prime]], [[l_prime]]], axis=0)
        #                                 g_prime_l_prime_stack = tf.tile(g_prime_l_prime_stack, [num_of_batch, 1])
                                    psych_alpha_beta_stack = tf.concat([psych_alpha, [[psych_beta]]], axis=0)
        #                                 psych_alpha_beta_stack = tf.tile(psych_alpha_beta_stack, [num_of_batch, 1])
                                    sig_a_b_stack = tf.concat([sig_a, [[sig_b]]], axis=0)
                                except Exception as e:
                                    raise type(e)(str(e) + "pairing concatenation is fucked")
        
                            except Exception as e:
                                raise type(e)(str(e) + "the first concat is raising the error" )
#                                 sig_a_b_stack = tf.tile(sig_a_b_stack, [num_of_batch, 1])

                            # if tf.rank(sig_linear[0]) > tf.rank(g_prime_l_prime_stack[0]):
                            #     tf.print ("SHAPES ARE:", tf.shape(g_prime_l_prime_stack), tf.shape(psych_alpha_beta_stack), tf.shape(sig_a_b_stack), tf.shape(psych_linear), tf.shape(sig_linear))
                            #     g_prime_l_prime_stack = tf.expand_dims(g_prime_l_prime_stack, axis=0)
                            #     psych_alpha_beta_stack = tf.expand_dims(psych_alpha_beta_stack, axis=0)
                            #     sig_a_b_stack = tf.expand_dims(sig_a_b_stack, axis=0)
                            try:
                                try:
                                    output = [g_prime_l_prime_stack, psych_linear, psych_alpha_beta_stack, sig_linear, sig_a_b_stack,  manifold_norm, f_k_norm]
            #                                 if tf.rank(input_tensor_dict['sig_input']) == 2:
            #                                     concat_output = tf.concat(output, axis=1)
            #                                
                                    concat_output = tf.concat(output, axis=0)
                                except Exception as e:
                                    raise type(e)(str(e) + "last concat is an asshole ininer layer !!!")
                                
                            except Exception as e:
                                raise type(e)(str(e) + "last concat is an asshole outer layer!!!")
                            return concat_output
                    except Exception as e:
                        raise type(e)(str(e)+"scond layer 2")
                except Exception as e:
                    raise type(e)(str(e)+'ERROR IS HAPPENNING IN CONCAT! gp_kernel_lengthscale_trainable:'+str(self.gp_kernel_lengthscale_trainable)+' training:'+str(training))
#                     try:
#                         try:
                        # output = tf.concat([tf.expand_dims(g_prime, axis=0), tf.expand_dims(l_prime, axis=0), psych_linear, psych_alpha, psych_beta, sig_linear, sig_a, sig_b], axis=0)
#                         except Exception as e:
#                             raise type(e)(str(e)+" second layer")
#                     except Exception as e:
#                         raise type(e)(str(e)+str("This is inconeviently fucked up!"))
        #             tf.print ("THE SHAPE OF CALL OUTPUT:", tf.shape(output))
            except Exception as e:
                raise type(e)(str(e)+"THIS IS PRETTY FUCKED UP" + str(self.gp_kernel_lengthscale_trainable)+', '+str(self.manifold_kernel_type))
    #         tf.print ("concat input:", extend_tensor(psych_gamma), output_stream=sys.stderr)
    #         return tf.concat([extend_tensor(psych_gamma),extend_tensor(psych_lambda), psych_linear, sig_linear, extend_tensor(self.masking_var)], axis=1)
        
        except Exception as e:
            raise type(e)(str(e)+'Squaring W is causing error 46' )
        
        return output

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
            # THE REASON BATCH SIZE ACTUALLY DEFINES THE PARTITIONING IN THE NEXT COMING LINES
            # IS THAT BATCH_SIZE IS WHAT DEFINES THE DIMENSION OF COEFFICEINTS IN OUR KERNEL METHOD
    #         sig_a_shape = tf.keras.backend.shape(sig_a)[0]
    #         if y_pred[2: batch_size + 2, :].shape[0] is None:
    #             self.psych_layer_input_shape = batch_size
    #             self.sig_layer_input_shape = batch_size
    #             print ("FUCKED UP BATCH SIZE:", batch_size)
    #         else:
    #             self.psych_layer_input_shape = 4096
    #             self.sig_layer_input_shape = 4096
            if self.manifold_kernel_type is not None:
    #             tf.print("WRONG PLACE")
                if self.gp_kernel_lengthscale_trainable:
#                     pass
    # #                 shape_list = np.cumsum([1, 1, 1, 1, 1, batch_size, 4096, 1, batch_size, 4096, 1, 1, 1, 1, 1]) - 1
                    gp_kernel_lengthscale, manifold_kernel_lengthscale,\
                    g_prime, l_prime, psych_linear, psych_alpha, psych_beta,\
                    sig_linear, sig_a, sig_b,\
                    manifold_norm, f_k_norm =tf.cast(y_pred[0, 0], tf.float64), tf.cast(y_pred[1, 0], tf.float64), tf.cast(y_pred[2, 0], tf.float64), tf.cast(y_pred[3, 0], tf.float64), tf.cast(y_pred[4: batch_size + 4, :], tf.float64), tf.cast(y_pred[batch_size + 4:self.psych_layer_input_shape + batch_size + 4, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + batch_size + 4, 0], tf.float64),\
                    tf.cast(y_pred[self.psych_layer_input_shape + batch_size + 5:self.psych_layer_input_shape + batch_size * 2 + 5, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + batch_size * 2 + 5: self.psych_layer_input_shape + self.sig_layer_input_shape + batch_size * 2 + 5, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + self.sig_layer_input_shape + batch_size * 2 + 5, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + self.sig_layer_input_shape + batch_size * 2 + 6, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + self.sig_layer_input_shape + batch_size * 2 + 7, :], tf.float64)
        
    # tf.cast(y_pred[self.psych_layer_input_shape + self.sig_layer_input_shape + batch_size * 2 + 9, 0], tf.float64)     
                else:
                    g_prime, l_prime, psych_linear, psych_alpha, psych_beta,\
                    sig_linear, sig_a, sig_b,\
                    manifold_norm, f_k_norm = tf.cast(y_pred[0, 0], tf.float64), tf.cast(y_pred[1, 0], tf.float64), tf.cast(y_pred[2: batch_size + 2, :], tf.float64), tf.cast(y_pred[batch_size + 2:self.psych_layer_input_shape + batch_size + 2, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + batch_size + 2, 0], tf.float64),\
                    tf.cast(y_pred[self.psych_layer_input_shape + batch_size + 3:self.psych_layer_input_shape + batch_size * 2 + 3, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + batch_size * 2 + 3: self.psych_layer_input_shape + self.sig_layer_input_shape + batch_size * 2 + 3, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + self.sig_layer_input_shape + batch_size * 2 + 3, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + self.sig_layer_input_shape + batch_size * 2 + 4, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + self.sig_layer_input_shape + batch_size * 2 + 5, :], tf.float64)
                # if tf.rank(g_prime) == 0:
                #     g_prime = tf.expand_dims(tf.expand_dims(g_prime, axis=0), axis=0)
                #     l_prime = tf.expand_dims(tf.expand_dims(l_prime, axis=0), axis=0)
                manifold_norm = tf.cast(tf.zeros_like(manifold_norm), tf.float64)
                    


            else:

                g_prime, l_prime, psych_linear, psych_alpha, psych_beta,\
                sig_linear, sig_a, sig_b = \
                tf.cast(y_pred[0, 0], tf.float64), tf.cast(y_pred[1, 0], tf.float64), tf.cast(y_pred[2: batch_size + 2, :], tf.float64), tf.cast(y_pred[batch_size + 2:self.psych_layer_input_shape + batch_size + 2, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + batch_size + 2, :], tf.float64),\
                tf.cast(y_pred[self.psych_layer_input_shape + batch_size + 3:self.psych_layer_input_shape + batch_size * 2 + 3, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + batch_size * 2 + 3: self.psych_layer_input_shape + self.sig_layer_input_shape + batch_size * 2 + 3, :], tf.float64), tf.cast(y_pred[self.psych_layer_input_shape + self.sig_layer_input_shape + batch_size * 2 + 3, :], tf.float64)      
                manifold_norm = tf.zeros(shape=[1, 1, 1])
                f_k_norm = tf.zeros(shape=[1, 1, 1])
#             tf.print ("f_k_norm in psychmkeras is:", f_k_norm)

        except Exception as e:
            raise type(e)(str(e) + "calling call() in keras_model casues error" + str(y_pred.shape) + ', ' +str(y_pred))
        try:
    #         tf.print("WE ARE HERE?!", psych_gamma, psych_lambda, psych_linear.shape, psych_alpha.shape, psych_beta,\
    #             sig_linear.shape, sig_a.shape, sig_b,\
    #             Lap_norm, f_k_norm, alpha_norm, mix_norm, y_pred.shape)
    #         tf.print ("LENGTHS IN LOSS:", 
    #                   sig_a_shape+ batch_size*2+4, batch_size*3+4, tf.keras.backend.shape(y_pred),
    #                  "psych_lambda_length:", tf.keras.backend.shape(psych_lambda),
    #                   "psych_gamma_length:", tf.keras.backend.shape(psych_gamma), 
    #                  "psych_linear_length:", tf.keras.backend.shape(psych_linear),
    #                   "sig_linear_length:", tf.keras.backend.shape(sig_linear), 
    #                  "sig_a_length:", tf.keras.backend.shape(sig_a),
    #                   "sig_b_length:", tf.keras.backend.shape(sig_b), 
    #                  "Lap_norm_length:", tf.keras.backend.shape(Lap_norm),
    #                   "f_k_norm_length:", tf.keras.backend.shape(f_k_norm), output_stream=sys.stderr)
            if hasattr(l_y_decimal_true, 'values'):
                l_y_decimal_true = l_y_decimal_true.values
#             l_y_decimal_true = l_y_decimal_true.reshape((1, -1))[0]
            l_true, _ = tf.cast(tf.cast(l_y_decimal_true / 2, tf.int32), tf.float64), tf.math.mod(l_y_decimal_true, 2)
    #         y = tf.cast(l_true, tf.float64)

#             psych_gamma = tf.cast(psych_gamma, tf.float64)
#             psych_lambda = tf.cast(psych_lambda, tf.float64)
#             g_prime_abs, l_prime_abs = gamma_lambda_to_g_l_prime_transformer(psych_gamma, psych_lambda)#, reparametrization_style=self.reparametrization_style)

            if g_prime is None:
                raise NotImplementedError(" FUCK ALL THE NONES!")
            if psych_linear is None:
                raise NotImplementedError(" FUCK ALL THE NONES!")
    #         abs_psych_gamma = tf.abs(psych_gamma)
    #         abs_psych_lambda = tf.abs(psych_lambda)
#             abs_g_prime = tf.abs(g_prime)
#             abs_l_prime = tf.abs(l_prime)
#             sq_sum = abs_g_prime + abs_l_prime

    #         abs_psych_linear = tf.abs(psych_linear)
    #         abs_sig_linear = tf.abs(sig_linear)
    #         neg_relu_psych_linear = -tf.nn.relu(-psych_linear)
    #         neg_relu_sig_linear = -tf.nn.relu(-sig_linear)
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
                        main_loss = expanded_using_softmax_mathematica(l_true, g_prime, l_prime, psych_linear, sig_linear)
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

        total_loss = total_loss + tf.linalg.norm(psych_alpha, ord=2) ** 2 * self.psych_reg_coeff ** 2 / 2. + psych_beta ** 2 / 2.  * 1e-4 + g_prime ** 2 / 2.  + l_prime ** 2 / 2.  + sig_b ** 2 / 2. * 1e-4
        
        return tf.squeeze(total_loss)