import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve
@tf.function
def sparse_input_selector(func):
    def wrapper(X, vec_params, *args, **kwargs):
        if type(X_dict).__name__ == 'dict':
            pass
            
@tf.function            
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def find_optimal_threshold(l, y_score):
    fpr, tpr, thresholds = roc_curve(l, y_score)
    
    gmeans = np.sqrt(tpr)
    # gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    return thresholds[ix].astype(np.float64)



@tf.function
def expanded_using_softmax(l, g_prime, l_prime, psych_linear, sig_linear):
#     if l is not None and psych_linear is not None:
#         l = tf.cond(tf.equal(tf.keras.backend.shape(l), tf.keras.backend.shape(psych_linear)), lambda: l, lambda: tf.reshape(l, tf.keras.backend.shape(psych_linear)))
    l = tf.reshape(l, (1, -1))[0]
    L_s = psych_linear
    L_t = sig_linear
    log_sum_exp = tf.math.reduce_logsumexp
    
    # NOTICE: 
    # C is not appearing because C = E - D + F
    # Also A = log(D_star) + G + H
    # Also C = E - D + F
    try:
        shape = tf.shape(L_s)
#         tf.print ("shape:", shape)
#         tf.print("L_s:", L_s)
        zero = tf.constant(0., tf.float64)
#         tf.print ("secod element shape:", log_sum_exp(tf.concat([-tf.tile([[g_prime]], shape), -g_prime - L_s,  tf.tile([[zero]], shape)], axis=1), axis=1, name='A_leftover').shape)
#         tf.print ("first element shape:", l.shape)
#         tf.print ('L_s:', L_s)
#         tf.print ('L_t:', L_t)
#         tf.print ('l:', 1-l, "with shape:", l.shape)
        A_leftover = tf.multiply(l, log_sum_exp(tf.concat([-tf.tile([[g_prime]], shape), -g_prime - L_s,  tf.tile([[zero]], shape)], axis=1), axis=1, name='A_leftover'))
#         tf.print ("A_leftover:", log_sum_exp(tf.concat([-tf.tile([[g_prime]], shape), -g_prime - L_s,  tf.tile([[zero]], shape)], axis=1), axis=1))
        H_F_B_G =  - log_sum_exp(tf.concat([tf.tile([[zero]], shape), -L_s], axis=1), axis=1, name='H_F_B_G_1') - log_sum_exp(tf.concat([tf.tile([[zero]], shape), -L_t], axis=1), axis=1, name='H_F_B_G_2') \
        - log_sum_exp(tf.concat([tf.tile([[zero]], shape), tf.tile([[-l_prime]], shape), tf.tile([[-g_prime]],shape)], axis=1), axis=1, name='H_F_B_G_3')
        E_D = tf.multiply(1.-l, log_sum_exp(tf.concat([tf.tile([[-l_prime]], shape), -L_s, -L_s - l_prime, -L_t, -L_t - g_prime, -L_t - l_prime, -L_s - L_t, -L_s - L_t - g_prime, -L_s - L_t - l_prime], axis=1), axis=1, name='E_D'))
        log_like = A_leftover + H_F_B_G + E_D
#         tf.print ("E_D:",tf.multiply(1.-l, log_sum_exp(tf.concat([tf.tile([[-l_prime]], shape), -L_s, -L_s - l_prime, -L_t, -L_t - g_prime, -L_t - l_prime, -L_s - L_t, -L_s - L_t - g_prime, -L_s - L_t - l_prime], axis=1), axis=1, name='E_D')))
    except Exception as e:
        raise type(e)(str(e) + "error is happening in using_softmax")
    
    return log_like

@tf.function
def expanded_using_softmax_mathematica(l_i, gamma_prime, lamda_prime, o_l, o_y):
    l_i = tf.reshape(l_i, (1, -1))[0]
    try:
        log_sum_exp = tf.math.reduce_logsumexp
        shape = tf.shape(o_l)
        zero = tf.constant(0., tf.float64)
        zeros_col = tf.tile([[zero]], shape)
    except Exception as e:
        raise type(e)(str(e) + " error is happening in tiling part of softmax mathematica1 ")
        
    try:
        # tf.print ("in expanded_using_softmax inputs are... g_prime is", gamma_prime, "l_prime is:", lamda_prime, "shape is:", shape)
        lamda_prime_tile = tf.tile([[lamda_prime]], multiples=shape)
        gamma_prime_tile = tf.tile([[gamma_prime]], multiples=shape)
    except Exception as e:
        raise type(e)(str(e) + " error is happening in tiling part of softmax mathematica 2" + str(gamma_prime)+','+ str(lamda_prime)+','+str(shape))
    
    try:
        part_1 = log_sum_exp(tf.concat([zeros_col,  gamma_prime_tile, lamda_prime_tile, o_l, [[gamma_prime]] + o_l, [[lamda_prime]] + o_l, o_y, [[lamda_prime]] + o_y, [[lamda_prime]] + o_l + o_y], axis=1), axis=1) - log_sum_exp(tf.concat([zeros_col, gamma_prime_tile, lamda_prime_tile], axis=1),axis=1) -log_sum_exp(tf.concat([zeros_col, o_l], axis=1), axis=1) - log_sum_exp(tf.concat([zeros_col,  o_y], axis=1), axis=1)

        part_2 = -l_i * log_sum_exp(tf.concat([zeros_col, -o_y], axis=1), axis=1)

        part_3 = (log_sum_exp(tf.concat([gamma_prime_tile, o_l, [[gamma_prime]] + o_l], axis=1), axis=1) - log_sum_exp(tf.concat([zeros_col, gamma_prime_tile, lamda_prime_tile], axis=1), axis=1) - log_sum_exp(tf.concat([zeros_col, o_l], axis=1), axis=1)) * l_i

        part_4 = -((log_sum_exp(tf.concat([zeros_col, gamma_prime_tile, lamda_prime_tile, o_l, [[gamma_prime]] + o_l, [[lamda_prime]] + o_l, o_y, [[lamda_prime]] + o_y, [[lamda_prime]] + o_l + o_y], axis=1), axis=1)) - log_sum_exp(tf.concat([zeros_col, gamma_prime_tile, lamda_prime_tile], axis=1), axis=1) - log_sum_exp(tf.concat([zeros_col, o_l], axis=1), axis=1) - log_sum_exp(tf.concat([zeros_col,  o_y], axis=1), axis=1)) * l_i
    except Exception as e:
        raise type(e)(str(e) + "error is happening in calculating parts of softmax loss")
    
    return part_1 + part_2 + part_3 + part_4

@tf.function
def g_l_prime_to_gamma_lambda_transformer(g_prime, l_prime):#, reparametrization_style=3):       
#     if reparametrization_style == 4:
#         gamma_ = g_prime_ 
#         lambda_ = l_prime_
#     elif reparametrization_style == 10:
#         gamma_ = g_prime
#         lambda_ = l_prime
# #     elif reparametrization_style == 6:
# #         psych_gamma_ = conv_to_prob(g_prime_) * elkan_c
# #         psych_lambda_ = conv_to_prob(l_prime_) * (1 - elkan_c)
# #   if self.reparametrization_style == 7:
# #     abs_g = abs(g_prime_)
# #     abs_l = abs(l_prime_)
# #     abs_sum = abs_g + abs_l + 1
# #     self.psych_gamma_ = (1 + elkan_c)/2 * abs_g / abs_sum
# #     self.psych_lambda_ = (2 - elkan_c)/2 * abs_l / abs_sum
#     elif reparametrization_style == 8:
#           gamma_ = tf.exp(g_prime) / (tf.exp(g_prime) + tf.exp(l_prime) + 1)
#           lambda_ = tf.exp(l_prime) / (tf.exp(g_prime) + tf.exp(l_prime) + 1)
#     else:
#       g_l_sum_ = tf.abs(g_prime) + 2 * tf.abs(l_prime) + 1
#       gamma_ = (tf.abs(g_prime) + tf.abs(l_prime)) / g_l_sum_
#       lambda_ = tf.abs(l_prime) / g_l_sum_
#       g_l_sum_ = tf.square(g_prime) + tf.square(l_prime) + 1
#       gamma_ = tf.square(g_prime) / g_l_sum_
#       lambda_ = tf.square(l_prime) / g_l_sum_
#       g_l_sum_ = tf.abs(g_prime) + tf.abs(l_prime) + 1
#       gamma_ =  tf.abs(g_prime) / g_l_sum_
#       lambda_ = tf.abs(l_prime) / g_l_sum_
    try:
        output = tf.keras.layers.Softmax(axis=0)(tf.convert_to_tensor([tf.zeros_like(g_prime), g_prime, l_prime]))
        gamma_ = output[1]
        lambda_ = output[2]
        
        # gamma_ = tf.expand_dims(output[1], axis=0)
        # gamma_ = tf.expand_dims(gamma_, axis=0)
        # lambda_ = tf.expand_dims(output[2], axis=0)
        # lambda_ = tf.expand_dims(lambda_, axis=0)
    except Exception as e:
        raise type(e)(str(e) + "transformer does not work!") 
#       gamma_ = .5 / (tf.abs(g_prime) + 1)
#       lambda_ = .5 / (tf.abs(l_prime) + 1)

#       gamma_ = (0.5 - 1e-8) * tf.math.sigmoid(g_prime)
#       lambda_ = (0.5 - 1e-8) * tf.math.sigmoid(l_prime)
#       gamma_ = g_prime
#       lambda_ = l_prime
#       g_prime_sq = tf.square(g_prime)
#       l_prime_sq = tf.square(l_prime)
    
#       g_l_sum_ = tf.add(tf.add(g_prime_sq, l_prime_sq),  tf.cast(1, tf.float64))
#       gamma_ = tf.math.divide(g_prime_sq, g_l_sum_)
#       lambda_ = tf.math.divide(l_prime_sq, g_l_sum_)

    return gamma_, lambda_   
@tf.function
def gamma_lambda_to_g_l_prime_transformer(gamma_, lambda_):#, reparametrization_style=3):
#         g_l_diff = tf.math.subtract(tf.cast(1, tf.float64), tf.math.add(gamma_, lambda_))
#         g_prime = tf.math.divide(gamma_-lambda_, g_l_diff)
#         l_prime = tf.math.divide(lambda_, g_l_diff)
#         g_l_diff = tf.math.subtract(tf.cast(1, tf.float64), tf.math.add(tf.abs(gamma_), tf.abs(lambda_)))
#         l_prime = tf.math.divide(lambda_, g_l_diff)
#         g_prime = tf.math.divide(gamma_, g_l_diff)
    if gamma_ is None or lambda_ is None:
        logging = tf.get_logger()
        logging.warning("either gamma or lambda as input are null, returning the inputs as is!!!")  
        
        return gamma_, lambda_
    constant = tf.math.log(1. - gamma_ - lambda_)
    g_prime = (tf.math.log(gamma_) - constant)
    l_prime = (tf.math.log(lambda_) - constant)

#         g_prime = gamma_
#         l_prime = lambda_
#         g_prime = tf.math.log(2 * gamma_ / (1 - 2 * gamma_))
#         l_prime = tf.math.log(2 * lambda_ / (1 - 2 * lambda_))
#         g_prime = 0.5 / gamma_ - 1
#         l_prime = 0.5 / lambda_ - 1

#         l_prime = tf.sqrt(tf.math.divide(lambda_, g_l_diff))
#         c = (gamma_ - 0.5) / lambda_
#         g_prime = tf.math.divide( 2* (gamma_ - 0.5), 1. - 2 * (c + 1) * lambda_) 
#         l_prime = tf.math.divide(2 * lambda_,  1. - 2 * (c + 1) * lambda_) 
#     if reparametrization_style == 3:
#         g_prime = gamma_ / (1. - gamma_ - lambda_)
#         l_prime = lambda_ / (1. - gamma_ - lambda_)

#     if reparametrization_style == 7:
#         warm_gamma_prime = 1. / (1. + warm_start_model.c)
#         warm_lambda_prime = (1. - warm_start_model.c) / (2. - warm_start_model.c)
#         self.keras_model.psych_layer.elkan_c.assign(warm_start_model.c)                                        
#         g_l_sum_plus_1 = 1. / (1. / (1. + warm_start_model.c) - (1. - warm_start_model.c) / (2. - warm_start_model.c))
#     g_prime = (gamma_ - lambda_) / (1 - gamma_ -  lambda_)
#     if reparametrization_style in [1, 3, 6, 7]:
#         g_prime = (gamma_ - lambda_) / (1 - gamma_ -  lambda_)
#         l_prime = lambda_ / (1 - gamma_ -  lambda_)
#     elif reparametrization_style == 8:
#         g_prime = np.log(gamma_) - np.log(1 - gamma_ - lambda_)
#         l_prime = np.log(lambda_) - np.log(1 - gamma_ - lambda_)
#     elif reparametrization_style == 10:
#         g_prime = gamma_
#         l_prime = lambda_
#     else: 
#         raise NotImplementedError



    return g_prime, l_prime