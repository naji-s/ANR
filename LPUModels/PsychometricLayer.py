"""Rewritten psychometric layer (older one was psych_TF) for TF2"""
import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')
import sys
sys.path.append('/home/scratch/nshajari/psych_model/LPUModels')
sys.path.append('/home/scratch/nshajari/psych_model/')

from utils.func_lib import gamma_lambda_to_g_l_prime_transformer, g_l_prime_to_gamma_lambda_transformer
sys.path.append('/home/scratch/nshajari/psych_model/utils/')
from FunctionFactory import  post_build_variable_merging_properties, assign_new_model_parameters
class PsychometricLayer(tf.keras.layers.Layer):
    """The idea of handling sparse tensorf is taken from 
    https://medium.com/dailymotion/how-to-design-deep-learning-models-with-sparse-inputs-in-tensorflow-keras-fd5e754abec1
    lot of thanks goes to Sharon!! """
    
    def check_input_float64(self, attrs):
        for attr in attrs:
            attr_val = getattr(self, attr)
            if 'ndarray' in type(attr_val).__name__:
                setattr(self, attr,  attr_val.astype(np.float64))
            elif attr_val is not None:
                if 'int' not in type(attr_val).__name__.lower():
                    setattr(self, attr, np.float64(attr_val))

    def __init__(self, is_sigmoid=False, sparse=False, reg_coeff= None, reg_penalty=None, layer_name=None, g_prime=None, l_prime=None, alpha=None, beta=None, name='psychometric_layer', freeze_psychometric_original=None, freeze_psych_alpha=None,*args, **kwargs):
        super().__init__(name=name)
        self.freeze_psych_alpha = freeze_psych_alpha
        self.freeze_psychometric_original = freeze_psychometric_original
        self.layer_name = layer_name
        self.is_sigmoid = is_sigmoid
        self.reg_penalty = reg_penalty
        self._reg_coeff = reg_coeff
        self._l_prime = l_prime
        self._g_prime = g_prime
        self._alpha = alpha
        self._beta = beta
        self.sparse=sparse
        self.report_g_l_prime()
        self.check_input_float64(attrs=[
            '_reg_coeff',
            '_g_prime',
            '_l_prime',
            '_alpha',
            '_beta']
            )
        
    @property            # first decorate the getter method
    @tf.function
    def g_prime(self): # This getter method name is *the* name
        return self._g_prime.get_weights().numpy()

    @property            # first decorate the getter method
    @tf.function
    def l_prime(self): # This getter method name is *the* name
        return self._l_prime.get_weights().numpy()
        
        
        
    
    def report_g_l_prime(self):
        tf.print ("!!!NOTE: g_prime and l_prime are set to:", self._g_prime, self._l_prime)


    def build(self, input_shape):
        tf.print("INPUT SHAPE FOR BUILD OF PSYCHOMETRIC LAYER IS:", input_shape, type(input_shape))
        try:
            for key in input_shape.keys():
                if 'psych_input' in key:
                    input_shape = input_shape[key]
                    break
        except Exception as e:
            raise type(e)(str(e) + " the error is happening in inferring shape of psychometric layer. the input_shape for build method is:" + str(input_shape))
            
        if self._alpha is None:
            alpha_initializer = lambda x, dtype: tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=.1)(shape=[int(input_shape[-1]), 1], dtype=tf.float64)
        else:
            alpha_initializer = lambda x, dtype: tf.constant(self._alpha, dtype=tf.float64)
        print ("The passed reg_coeff is:", self._reg_coeff, input_shape)
        if self.reg_penalty is None: 
            alpha_regularizer = None
        else: 
            alpha_regularizer = self.reg_penalty(self._reg_coeff)
        self._alpha = self.add_weight("alpha", shape=[int(input_shape[-1]), 1], regularizer=alpha_regularizer ,trainable=(not self.freeze_psychometric_original and not self.freeze_psych_alpha), initializer=alpha_initializer, dtype=tf.float64)
        
        
        if self._beta is None:
            beta_initializer = lambda x, dtype:\
                            tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1e-1)(shape=[], dtype=tf.float64)
        else:
            beta_initializer = lambda x, dtype:\
                            tf.constant(self._beta, dtype=tf.float64)

        self._beta = self.add_weight("beta",shape=[], 
                                    initializer=beta_initializer, 
                                    trainable=not self.freeze_psychometric_original, dtype=tf.float64)

        
        if self.is_sigmoid:
            if self.layer_name is None:
                self.layer_name = 'sigmoid_layer'
            trainable = False
        else:
            if self.layer_name is None:
                self.layer_name = 'psychometric_layer'
            trainable = True

        trainable = trainable and (not self.freeze_psychometric_original)

        
        if self._g_prime is None:
            g_prime_initalizer = tf.keras.initializers.RandomNormal(mean=0., stddev=2)
        else:            
            g_prime_initalizer = lambda x, dtype: tf.constant(self._g_prime, shape=[], dtype=tf.float64)
        if self._l_prime is None:
            l_prime_initalizer = tf.keras.initializers.RandomNormal(mean=0., stddev=2)
        else:
            l_prime_initalizer = lambda x, dtype: tf.constant(self._l_prime, shape=[], dtype=tf.float64)            
            
        print ("g_prime_initial_value is:", self._g_prime)
        print ("l_prime_initial_value is:", self._l_prime)

        
        constraint = None
            
        self._g_prime = self.add_weight(self.layer_name + "_gamma_prime", shape=[], 
                                       constraint=constraint, trainable=trainable, 
                                       initializer=g_prime_initalizer, dtype=tf.float64)
        self._l_prime = self.add_weight(self.layer_name + "_lambda_prime", shape=[], 
                                       constraint=constraint, trainable=trainable, 
                                       initializer=l_prime_initalizer, dtype=tf.float64)

#         self.order = [0] * len(self.variables)

#         for i, variable in enumerate(self.variables):
#             if 'alpha' in variable.name:
#                 self.order[0] = i
#             elif 'beta' in variable.name:
#                 self.order[1] = i
#             elif 'gamma' in variable.name:
#                 self.order[2] = i
#             elif 'lambda' in variable.name:
#                 self.order[3] = i
#             else:
#                 raise NotImplementedError("THE VARIABLE DOES NOT BELONG TO PSYCHOMETRIC LAYER. WTF??"+str(variable.name))
                

    def call(self, input_tensor_dict, training=False):
        try:
            input_idx = input_tensor_dict['idx']
        except KeyError as e:
            try:
                logging = tf.get_logger()
                logging.warning("masker_cast is not set in PsychometricLayer. defaulting to set the mask to be all indices...")            
                input_idx = input_tensor_dict['idx'] = tf.cast(tf.range(len(input_tensor_dict['psych_input'])), tf.int32)
            except Exception as e_2:
                raise type(e_2)("masker_cast setting creates this problem in PsychometricLayer.py with error"+str(e_2))
        
        
        layer_input = input_tensor_dict['psych_input']
        try:
            def expand_layer_input(inp):
                return tf.expand_dims(inp, 1)
            layer_input = tf.cond(tf.math.equal(tf.rank(layer_input), 1), lambda: expand_layer_input(layer_input), lambda: layer_input)
            batch_size = tf.keras.backend.shape(layer_input)[0]
        except Exception as e:
            raise type(e)("Error is in this stpe of PsychometricLayer call:"+str(e))
        try:
            # if not isinstance(layer_input, tf.SparseTensor):
            linear_out = tf.add(tf.matmul(layer_input, self._alpha), self._beta)
        except Exception as e:
            raise type(e)("Error is in this step 2 of PsychometricLayer call:"+str(e))
        
        return linear_out#, self._g_prime, self._l_prime, self._alpha, self._beta


#         try:
#             if tf.rank(input_tensor)==2:
#                 linear_out = tf.expand_dims(linear_out, axis=0)
#         except Exception as e:
#             raise type(e)(str(e) + "linear_out is:", linear_out)
    
        # if tf.rank(input_tensor) == 1:
            # input_tensor = tf.expand_dims(input_tensor, axis=0)
            # raise NotImplementedError("Your input tensor does need to have a rank/ndim of 2. Where rows are data points of dimension d. As such the input is of the shape n x d, where n is the number of datapoints")
    
    
        # THERE ARE 7 POSSSIBLE REPARAMETRIZATIONS TO DEAL WITH CONSTRAINTS ON gamma AND lambda.
        # 1: WE TAKE GAMMMA = ABS(GAMMA_PRIME) / (ABS(GAMMA_PRIME) + ABS(LAMBDA_PRIME) + 1) AND SAME FOR LAMBDA
        # 2: WE DEFINE C = (GAMMA - LAMBDA) / 2 AND D = (GAMMA + LAMBDA) / 2 AND THEN WE CAN USE KERAS CONSTRAINTS ON
        # SINGLE VARIABLES TO STILL MAKE CONSTRAINTS FOR GAMMA AND LAMBDA HOLD
        # 3: LIKE NUMBER 1 EXCEPT WE USE Keras CONSTRAINTS RATHER THAN ABS()
        # 4: LAGRANGIAN METHOD WHICH DOESN'T WORK CURRENTLY
        # 5: SAME AS 1 EXCEPT WITH ABS REPLACED BY SQUARE()
        # 6: WE IMPOSE GAMMA < ELKAN_C / 2 AND LAMBDA < (1 - ELKAN_C) / 2 TO MAKE SURE GAMMA AND LAMBDA NEVER PASS C
        # 7: WE IMPOSE GAMMA < (1 + ELKAN_C) / 2 AND LAMBDA < (2 - ELKAN_C) / 2 TO MAKE SURE GAMMA AND LAMBDA ARE NOT TOO FAR 
        # FROM ELKAN_C
    
    
    # def compute_output_shape(self, input_shape):
    #     input_shape = input_shape.get_shape().as_list()
    #     return input_shape[0], self.num_outputs
    
    
#     def get_output_shape(self, input_shape):
#         input_shape = input_shape.get_shape().as_list()
#         return input_shape[0], self.num_outputs
#     @tf.function
#     def get_output_shape(self, input_shape):
#         input_shape = input_shape.get_shape().as_list()
#         return input_shape[0], self.num_outputs
    
#     def predict_proba(self, X, **kwargs):
#         gamma_, lambda_, linear_out = self.call(X)
#         return gamma_ + (1 - gamma_ - lambda_) * tf.sigmoid(linear_out)
    
#     def predict(self, X, **kwargs):
#         return self.predict_proba(X) > 0.5
        

    #     @tf.function
#     def get_gamma_lambda(self):
#         # THERE ARE 7 POSSSIBLE REPARAMETRIZATIONS TO DEAL WITH CONSTRAINTS ON gamma AND lambda.
#         # 1: WE TAKE GAMMMA = ABS(GAMMA_PRIME) / (ABS(GAMMA_PRIME) + ABS(LAMBDA_PRIME) + 1) AND SAME FOR LAMBDA
#         # 2: WE DEFINE C = (GAMMA - LAMBDA) / 2 AND D = (GAMMA + LAMBDA) / 2 AND THEN WE CAN USE KERAS CONSTRAINTS ON
#         # SINGLE VARIABLES TO STILL MAKE CONSTRAINTS FOR GAMMA AND LAMBDA HOLD
#         # 3: LIKE NUMBER 1 EXCEPT WE USE Keras CONSTRAINTS RATHER THAN ABS()
#         # 4: LAGRANGIAN METHOD WHICH DOESN'T WORK CURRENTLY
#         # 5: SAME AS 1 EXCEPT WITH ABS REPLACED BY SQUARE()
#         # 6: WE IMPOSE GAMMA < ELKAN_C / 2 AND LAMBDA < (1 - ELKAN_C) / 2 TO MAKE SURE GAMMA AND LAMBDA NEVER PASS C
#         # 7: WE IMPOSE GAMMA < (1 + ELKAN_C) / 2 AND LAMBDA < (2 - ELKAN_C) / 2 TO MAKE SURE GAMMA AND LAMBDA ARE NOT TOO FAR FROM ELKAN_C
#         # 9: CORDINATE DESCENT USING ITEM 4-STYLE (SEE ABOVE) PARAMETRIZATION
#         # 10: Method of Multipliers 
# #         _, _, g_prime_init, l_prime_init = [item for item in self.get_weights()]
#         if self.reparametrization_style == 1:
#             gamma_lambda_sum = self.g_prime_init + self.l_prime_init + 1
#             gamma_ = self.g_prime_init / gamma_lambda_sum
#             lambda_ = self.l_prime_init / gamma_lambda_sum
#         elif self.reparametrization_style == 2: 
#             real_g_p_l = self.g_prime_init / 2. + 1 / 2.
#             gamma_ = tf.abs((real_g_p_l + self.l_prime_init) / 2.)
#             lambda_ = tf.abs((real_g_p_l - self.l_prime_init) / 2.)
#         elif self.reparametrization_style == 3:
#             g_prime_abs = tf.abs(self.g_prime_init)
#             l_prime_abs = tf.abs(self.l_prime_init)
#             gamma_lambda_sum = g_prime_abs + l_prime_abs + 1
#             gamma_ = g_prime_abs / gamma_lambda_sum
#             lambda_ = l_prime_abs / gamma_lambda_sum
#         elif self.reparametrization_style in [4, 9, 10]:            
#             gamma_ = self.g_prime_init
#             lambda_ = self.l_prime_init
#         elif self.reparametrization_style == 5:  
#             g_prime_sq = tf.square(self.g_prime_init)
#             l_prime_sq = tf.square(self.l_prime_init)
#             gamma_lambda_sum = g_prime_sq + l_prime_sq + 1
#             gamma_ = g_prime_sq / gamma_lambda_sum
#             lambda_ = l_prime_sq / gamma_lambda_sum
#         elif self.reparametrization_style == 6:
#             print ("Elkan c is:", self.elkan_c)
#             if self.layer_name == 'sigmoidal':
#                 gamma_ = np.float64(1e-16)
#                 lambda_ = np.float64(1e-16)
#             else:
#                 gamma_ = self.elkan_c * tf.sigmoid(self.g_prime_init)
#                 lambda_ = (1 - self.elkan_c) * tf.sigmoid(self.l_prime_init)
# #                 gamma_ = self.elkan_c * (1. / (1. + g_prime_abs))
# #                 lambda_ = (1 - self.elkan_c) * (1. / (1. + l_prime_abs))
#         elif self.reparametrization_style == 7:
#             abs_gamma = tf.abs(self.g_prime_init)
#             abs_lambda = tf.abs(self.l_prime_init)
#             if self.layer_name == 'sigmoidal':
#                 gamma_ = np.float64(1e-16)
#                 lambda_ =np.float64(1e-16)
#             else:                
#                 abs_sum = abs_gamma + abs_lambda + 1
#                 gamma_ = (1 + self.elkan_c) / 2 * (abs_gamma / abs_sum)
#                 lambda_ = (2 - self.elkan_c) / 2 * (abs_lambda / abs_sum)
#         elif self.reparametrization_style == 8:
#             if self.layer_name == 'sigmoidal':
#                 gamma_ = self.g_prime_init * 0.
#                 lambda_ = self.g_prime_init * 0.
#             else:
#                 gamma_ = tf.exp(self.g_prime_init) / (1 + tf.exp(self.g_prime_init) + tf.exp(self.l_prime_init))
#                 lambda_ = tf.exp(self.l_prime_init) / (1 + tf.exp(self.g_prime_init) + tf.exp(self.l_prime_init))     
#         else:
#             raise NotImplementedError     
#         return gamma_, lambda_