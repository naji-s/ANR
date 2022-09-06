import tensorflow as tf
# tf.keras.backend.set_floatx('float64')
# from sklearn._base import _check_pos_label_consistency
import sys
sys.path.append('/home/scratch/nshajari/psych_model/')
sys.path.append('/home/scratch/nshajari/psych_model/utils/')

import sys
# def LPU_scorer(estimator, X, y):
#     # imidiately decoding the categorical variable l_y_cat to l and y, and  
#     # dropping real class (y) by assigning it to _
#     l, y = self.encoder.inverse_transform(y)
#     from numpy import isnan
#     from sklearn import metrics
#     try:
#         estimator_outputs = estimator.predict_proba(X).values
#     except Exception as e:
# #         print ("FUCKING Exception!!")
#         estimator_outputs = estimator.predict_proba(X)
# #         print ("value:", estimator_outputs)
#     if isnan(estimator_outputs).any():
#         print(type(estimator).__name__)
# #     return metrics.average_precision_score(l, estimator_outputs)
# #     return metrics.roc_auc_score(l, estimator_outputs)
#     if estimator_outputs.shape[-1] == 2:
#         estimator_outputs = estimator_outputs[:, -1]
#     return metrics.f1_score(l, estimator_outputs>0.5, average='weighted')
from functools import partial
import numpy as np
import inspect
from sklearn import metrics
import dill as pickle
# from LPUModels.PsychMKeras import g_l_prime_to_gamma_lambda_transformer 
def f1_0_score(y_true, y_pred, *args, **kwargs):
    return metrics.f1_score((1-y_true).reshape((-1, 1)), (1-y_pred).reshape((-1, 1)), *args, **kwargs)

def elkan_c_as_score(y_true, y_pred, *args, **kwargs):
    return y_pred[(y_true==1).reshape((-1, 1))].mean()

# def full_beta_score(y_true, y_pred,  *args, **kwargs):
def my_brier_score(y_true, y_pred, *args, **kwargs):
#     try:
#         pos_label = _check_pos_label_consistency(pos_label, y_true)
#     except ValueError:
#         classes = np.unique(y_true)
#         if classes.dtype.kind not in ('O', 'U', 'S'):
#             # for backward compatibility, if classes are not string then
#             # `pos_label` will correspond to the greater label
#             pos_label = classes[-1]
#         else:
#             raise    
#     y_true = np.array(y_true == pos_label, int)
    y_pred[y_pred < 1e-15] = 1e-15
    y_pred[y_pred > 1- 1e-15] = 1. - 1e-15
    return np.average((y_true.ravel() - y_pred.ravel()) ** 2)

def model_choser(cv_results):
    """
    Balance model complexity with cross-validated score.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`.

    Return
    ------
    int
        Index of a model that has the fewest PCA components
        while has its test score within 1 standard deviation of the best
        `mean_test_score`.
    """
    from numpy import argmax
#     threshold = lower_bound(cv_results)
#     candidate_idx = np.flatnonzero(cv_results['mean_test_score'] >= threshold)
#     best_idx = candidate_idx[cv_results['param_reduce_dim__n_components']
#                              [candidate_idx].argmin()]
    
     
    return argmax(cv_results['mean_test_lpu_brier_scorer'])
    
def scorer_general(estimator, X, y, scorer, learning_type=None, average_type=None, balanced=False):
    try:
        sample_weight = None
    #     print ("Now scorer", scorer.__name__, "is happening for estimator", type(estimator).__name__)
        l_y_decimal = estimator['clf'].encoder.inverse_transform(y)
        l, true_y = (l_y_decimal/2).astype(int), np.mod(l_y_decimal, 2).astype(int)
        from numpy import isnan, zeros
        from sklearn import metrics
        # going through pipeline and applying the necessary transformatons on X
        counter = 0
        for key, _ in estimator.steps:
            if key == 'clf':
                break
            X = estimator[key].transform(X)
            counter += 1
        if counter < 2:
            raise NotImplementedError("Scorer_library is fucking something up!!")
        if learning_type == 'lpu':
            if balanced:
                pos_freq = l.sum()/l.shape[0]
                sample_weight = np.zeros_like(l).astype(float)
                sample_weight[l==1] = 1-pos_freq
                sample_weight[l==0] = pos_freq
                

            estimator_outputs = estimator['clf'].predict_proba(X)
            if hasattr(estimator_outputs, 'values'):
                estimator_outputs = estimator['clf'].predict_proba(X).values
            binary_output = l
            estimator_outputs = np.nan_to_num(estimator_outputs, nan=1e-10)
            if isnan(estimator_outputs).any():
                print ("DOOM's day coming:", inspect.stack()[0][3])
                print ("Proceeding with STANDARD LOSS...")
    #             print ("psych_layer weights:", self.model.psych_layer.get_weights())
    #             print ("sig_layer weights:", self.model.sig_layer.get_weights())
                raise ValueError("DOOM's day coming:")
        elif learning_type == 'real':
            estimator_outputs = estimator['clf'].predict_prob_y_given_x(X)
            estimator_outputs = np.nan_to_num(estimator_outputs, nan=1e-10)
            if hasattr(estimator_outputs, 'values'):
                estimator_outputs = estimator['clf'].predict_prob_y_given_x(X).values
            binary_output = true_y
    #         print ("value:", estimator_outputs)
            if isnan(estimator_outputs).any():
                print("DOOM's day coming 2:", type(estimator).__name__)
                estimator_outputs = zeros(estimator_outputs.shape)
    #             exit(0)
                raise ValueError("DOOM DAY HAPPENED!")
    #     return metrics.average_precision_score(l, estimator_outputs)
    #     return metrics.roc_auc_score(l, estimator_outputs)
        if estimator_outputs.shape[-1] == 2:
            estimator_outputs = estimator_outputs[:, -1]
        binary_output_reshaped = binary_output.reshape((-1, 1))
        estimator_outputs_reshaped = estimator_outputs.reshape((-1, 1))
        # if  scorer.__name__  == 'brier_score_loss':
        #     scorer = self.safe_brier_score_loss
        if scorer.__name__ in ['brier_score_loss', 'log_loss', 'my_brier_score']:
            final_score = -scorer(binary_output_reshaped.ravel(), estimator_outputs_reshaped.ravel(), sample_weight=sample_weight)
        elif scorer.__name__ in ['roc_auc_score', 'average_precision_score']:
            if average_type is None:
                average_type = 'macro'
            if average_type == 'positive_only':
                final_score = scorer(binary_output_reshaped, estimator_outputs_reshaped, sample_weight=sample_weight)
            else:
                final_score = scorer(np.hstack([1-binary_output_reshaped, binary_output_reshaped]) , np.hstack([1 - estimator_outputs_reshaped, estimator_outputs_reshaped]), average=average_type, sample_weight=sample_weight)
                
        elif scorer.__name__ in ['f1_score', 'accuracy_score', 'f1_0_score']:
            final_score =  scorer(binary_output_reshaped.ravel(), estimator_outputs_reshaped.ravel()>0.5, sample_weight=sample_weight)
        elif scorer.__name__ == 'elkan_c_as_score':
            estimator_outputs = estimator['clf'].predict_prob_l_given_y_x(X)
            if hasattr(estimator_outputs, 'values'):
                estimator_outputs = estimator['clf'].predict_prob_l_given_y_x(X).values
            if type(estimator['clf']).__name__ == 'PropensityEM':
    #             final_score_1 = l.mean() / estimator['clf'].predict_prob_y_given_x(X).mean()
                final_score = estimator['clf'].predict_prob_l_given_y_x(X)[true_y==1].mean()
    #             final_score_2 = elkan_c_as_score(y, estimator_outputs)
    #             string_result =  final_score_1)[2:]
    #             string_result_2 = "{:.6f}".format(final_score_2)[2:]
    #             final_score = np.float64(string_result +'.'+ string_result_2)
                return final_score
            else:
                final_score = elkan_c_as_score(true_y.reshape((-1, 1)), estimator_outputs_reshaped)
        else:
            raise NotImplementedError("The scoring function of your choice does not exist. PLZ implement it!")
    except Exception as e:
        raise type(e)(str(e) + "WELL AT LEAAST WE KNOW WHO FUCKED IT UP!!")
    return final_score

def flexible_scorer(scorer, average_type=None, learning_type=None, balanced=False):
    ret_func = partial(scorer_general, scorer=scorer, average_type=average_type, learning_type=learning_type, balanced=balanced)
    return ret_func

def real_clf_scorer(estimator, X, y):
    from numpy import isnan
    from sklearn import metrics
    return -metrics.brier_score_loss(binary_output_reshaped, estimator.predict_proba(X).reshape((-1, 1)))



class LPUF1ScoreForY(tf.keras.metrics.Metric):
    """Computes F-Beta score.
    It is the weighted harmonic mean of precision
    and recall. Output range is [0, 1]. Works for
    both multi-class and multi-label classification.
    F-Beta = (1 + beta^2) * (prec * recall) / ((beta^2 * prec) + recall)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision and recall
            in harmonic mean. Determines the weight given to the
            precision and recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-Beta Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
        ValueError: If the `beta` value is less than or equal
        to 0.
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    def __init__(self,
                 num_classes=2,
                 average=None,
                 beta=1.0,
                 threshold=None,
                 name='lpu_f1_score_for_y',
                 dtype=tf.float64,
                 model=None,
                 from_logits=True
                ):
        super(LPUF1ScoreForY, self).__init__(name=name)
        if average not in (None, 'micro', 'macro', 'weighted'):
            raise ValueError("Unknown average type. Acceptable values "
                             "are: [None, micro, macro, weighted]")

        if not isinstance(beta, float):
            raise TypeError("The value of beta should be a python float")

        if beta <= 0.0:
            raise ValueError("beta value should be greater than zero")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError(
                    "The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")
        self.from_logits = from_logits
        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.init_shape = []
        self.model = model
        if self.average != 'micro':
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                name,
                shape=self.init_shape,
                initializer='zeros',
                dtype=self.dtype)
        self.true_positives = _zero_wt_init('true_positives')
        self.false_positives = _zero_wt_init('false_positives')
        self.false_negatives = _zero_wt_init('false_negatives')
        self.weights_intermediate = _zero_wt_init('weights_intermediate')

#     TODO: Add sample_weight support, currently it is
#     ignored during calculations.
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         tf.print("y_pred is:", y_pred, y_pred.shape, output_stream=sys.stderr)
#         tf.print("Model is:", self.model,  output_stream=sys.stderr)
#         if self.model is None:
#             pass
#         else:
#             self.custom_update_state(lambda x:self.model.keras_model(x))



    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_shape = tf.keras.backend.shape(y_true)[0]
        psych_layer_shape = self.model.keras_model.psych_layer_input_shape
        sig_layer_shape = self.model.keras_model.sig_layer_input_shape
        y_true = tf.math.mod(y_true, 2)
        y_true = tf.cast(y_true, tf.int32)
#         tf.print("y_pred in update_state is:", y_pred, tf.shape(y_pred), output_stream=sys.stderr)
        y_pred = y_pred[batch_shape + 3 + psych_layer_shape: 2 * batch_shape + 3 + psych_layer_shape]
#         y_pred = y_pred[batch_shape+3+psych_layer_shape: 2*batch_shape+3+psych_layer_shape]
# y_pred[batch_size * 2 + 3 + self.psych_layer_input_shape: batch_size * 2 + 3 + self.sig_layer_input_shape  + self.psych_layer_input_shape, :]
#         tf.print("y_pred in update_state is 2:", y_pred, tf.shape(y_pred), output_stream=sys.stderr)
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
#         tf.print("y_pred is:", y_pred, tf.shape(y_pred), output_stream=sys.stderr)

#         y_true = tf.reshape(y_true, [batch_shape, 1])
#         y_pred = tf.reshape(y_pred, [batch_shape, 1])
        y_true = tf.stack((1-y_true, y_true), axis=1)
        y_pred = tf.stack((1-y_pred, y_pred), axis=1)

        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.math.logical_and(y_pred >= threshold,
                                    tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold
#         y_pred = tf.cast(tf.cast((tf.sigmoid(self.model.keras_model.sig_linear) > 0.5), tf.int32), tf.float32)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        def _count_non_zero(val):
            non_zeros = tf.math.count_nonzero(val, axis=self.axis)
            return tf.cast(non_zeros, self.dtype)
        result = _count_non_zero(y_pred * y_true)
        self.true_positives.assign_add(tf.reshape(_count_non_zero(y_pred * y_true), [2]))
        self.false_positives.assign_add(tf.reshape(_count_non_zero(y_pred * (y_true - 1)), [2]))
        self.false_negatives.assign_add(tf.reshape(_count_non_zero((y_pred - 1) * y_true), [2]))
        self.weights_intermediate.assign_add(tf.reshape(_count_non_zero(y_true), [2]))

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives)
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives)
        precision = tf.cast(precision, tf.float32)
        recall = tf.cast(recall, tf.float32)
        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = (tf.math.divide_no_nan(mul_value, add_value))
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == 'weighted':
            weights = tf.math.divide_no_nan(
                self.weights_intermediate,
                tf.reduce_sum(self.weights_intermediate))
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)
            
        return f1_score
    
    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
        }

        if self.threshold is not None:
            config["threshold"] = self.threshold

        base_config = super(LPUF1ScoreForY, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_negatives.assign(tf.zeros(self.init_shape, self.dtype))
        self.weights_intermediate.assign(tf.zeros(self.init_shape, self.dtype))
        


    

class LPUBrierScoreForL(tf.keras.metrics.Metric):
    """Base class to create scores for p(l|x) for LPU setting, i.e.
    when we want to classify X to be either l=1 or l=0 (annotated vs. not annotated)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision and recall
            in harmonic mean. Determines the weight given to the
            precision and recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-Beta Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
        ValueError: If the `beta` value is less than or equal
        to 0.
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    def __init__(self,
                 name='lpu_brier_score_for_l',
                 dtype=tf.float64,
                 model=None,
                 from_logits=True, 
                 score_func=None,
                 num_classes=1                
                ):
        super(LPUBrierScoreForL, self).__init__(name=name)
#         if average not in (None, 'micro', 'macro', 'weighted'):
#             raise ValueError("Unknown average type. Acceptable values "
#                              "are: [None, micro, macro, weighted]")

#         if not isinstance(beta, float):
#             raise TypeError("The value of beta should be a python float")

#         if beta <= 0.0:
#             raise ValueError("beta value should be greater than zero")

        self.from_logits = from_logits
        self.init_shape = [num_classes]
        self.model = model
#         if self.average != 'micro':
#             self.axis = 0
#             self.init_shape = [self.num_classes]
        def _zero_wt_init(name):
            return self.add_weight(
                name,
                shape=self.init_shape,
                initializer='zeros',
                dtype=self.dtype)

        self.squared_difference = _zero_wt_init('squared_difference')

#     TODO: Add sample_weight support, currently it is
#     ignored during calculations.
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         tf.print("y_pred is:", y_pred, y_pred.shape, output_stream=sys.stderr)
#         tf.print("Model is:", self.model,  output_stream=sys.stderr)
#         if self.model is None:
#             pass
#         else:
#             self.custom_update_state(lambda x:self.model.keras_model(x))
        
#     def custom_update_state(output):
    
    def reset_state(self):
        self.squared_difference.assign(tf.zeros(self.init_shape, self.dtype))
        
        





#     def custom_update_state(output):
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
        batch_shape = tf.keras.backend.shape(y_true)[0]
        psych_layer_shape = self.model.keras_model.psych_layer_input_shape
        sig_layer_shape = self.model.keras_model.sig_layer_input_shape
        l_true = tf.math.divide(y_true, 2)
        l_true = tf.cast(l_true, tf.int32)        
#         tf.print("y_pred in update_state is:", y_pred, tf.shape(y_pred), output_stream=sys.stderr)
        g_prime, l_prime, psych_linear = tf.cast(y_pred[0, 0], tf.float64), tf.cast(y_pred[1, 0], tf.float64), tf.cast(y_pred[2: batch_shape + 2, :], tf.float64)
        sig_linear = y_pred[batch_shape + 3 + psych_layer_shape: 2 * batch_shape + 3 + psych_layer_shape]
        if self.from_logits:
            psych_gamma, psych_lambda = g_l_prime_to_gamma_lambda_transformer(g_prime, l_prime)
            sig_out = tf.sigmoid(sig_linear)
            psych_out = tf.sigmoid(psych_linear) * (1 - psych_gamma - psych_lambda) + psych_gamma                                      
        kwargs['sig_out'] = sig_out
        kwargs['psych_out'] = psych_out
        kwargs['l_true'] = l_true                                                                                          
        self.squared_difference.assign(tf.reduce_mean(tf.square(tf.cast(l_true, tf.float64) - tf.multiply(sig_out, psych_out)), axis=0))

    def result(self):
#         precision = tf.math.divide_no_nan(
#             self.true_positives, self.true_positives + self.false_positives)
#         recall = tf.math.divide_no_nan(
#             self.true_positives, self.true_positives + self.false_negatives)
#         precision = tf.cast(precision, tf.float32)
#         recall = tf.cast(recall, tf.float32)
#         mul_value = precision * recall
#         add_value = (tf.math.square(self.beta) * precision) + recall
#         mean = (tf.math.divide_no_nan(mul_value, add_value))
#         f1_score = mean * (1 + tf.math.square(self.beta))

#         if self.average == 'weighted':
#             weights = tf.math.divide_no_nan(
#                 self.weights_intermediate,
#                 tf.reduce_sum(self.weights_intermediate))
#             f1_score = tf.reduce_sum(f1_score * weights)

#         elif self.average is not None:  # [micro, macro]
#             f1_score = tf.reduce_mean(f1_score)
        brier_score = tf.reduce_mean(self.squared_difference)
        return brier_score
    
    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
#             "num_classes": self.num_classes,
#             "average": self.average,
#             "beta": self.beta,
        }

#         if self.threshold is not None:
#             config["threshold"] = self.threshold

        base_config = super(LPUBrierScoreForL, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    

class LPUBrierScoreForY(tf.keras.metrics.Metric):
    """Base class to create scores for p(l|x) for LPU setting, i.e.
    when we want to classify X to be either l=1 or l=0 (annotated vs. not annotated)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision and recall
            in harmonic mean. Determines the weight given to the
            precision and recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-Beta Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
        ValueError: If the `beta` value is less than or equal
        to 0.
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    def __init__(self,
                 name='lpu_brier_score_for_y',
                 dtype=tf.float64,
                 model=None,
                 from_logits=True, 
                 score_func=None,
                 num_classes=1                
                ):
        super(LPUBrierScoreForY, self).__init__(name=name)
#         if average not in (None, 'micro', 'macro', 'weighted'):
#             raise ValueError("Unknown average type. Acceptable values "
#                              "are: [None, micro, macro, weighted]")

#         if not isinstance(beta, float):
#             raise TypeError("The value of beta should be a python float")

#         if beta <= 0.0:
#             raise ValueError("beta value should be greater than zero")

        self.from_logits = from_logits
        self.init_shape = [num_classes]
        self.model = model
#         if self.average != 'micro':
#             self.axis = 0
#             self.init_shape = [self.num_classes]
        def _zero_wt_init(name):
            return self.add_weight(
                name,
                shape=self.init_shape,
                initializer='zeros',
                dtype=self.dtype)

        self.squared_difference = _zero_wt_init('squared_difference')

#     TODO: Add sample_weight support, currently it is
#     ignored during calculations.
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         tf.print("y_pred is:", y_pred, y_pred.shape, output_stream=sys.stderr)
#         tf.print("Model is:", self.model,  output_stream=sys.stderr)
#         if self.model is None:
#             pass
#         else:
#             self.custom_update_state(lambda x:self.model.keras_model(x))
        
#     def custom_update_state(output):
    
    def reset_state(self):
        self.squared_difference.assign(tf.zeros(self.init_shape, self.dtype))
        
        





#     def custom_update_state(output):
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
        batch_shape = tf.keras.backend.shape(y_true)[0]
        psych_layer_shape = self.model.keras_model.psych_layer_input_shape
        sig_layer_shape = self.model.keras_model.sig_layer_input_shape
        y_true = tf.math.mod(y_true, 2)
        y_true = tf.cast(y_true, tf.int32)        
#         tf.print("y_pred in update_state is:", y_pred, tf.shape(y_pred), output_stream=sys.stderr)
        g_prime, l_prime, psych_linear = tf.cast(y_pred[0, 0], tf.float64), tf.cast(y_pred[1, 0], tf.float64), tf.cast(y_pred[2: batch_shape + 2, :], tf.float64)
        sig_linear = y_pred[batch_shape + 3 + psych_layer_shape: 2 * batch_shape + 3 + psych_layer_shape]

        if self.from_logits:
            psych_gamma, psych_lambda = g_l_prime_to_gamma_lambda_transformer(g_prime, l_prime)
            sig_out = tf.sigmoid(sig_linear)
            psych_out = tf.sigmoid(psych_linear) * (1 - psych_gamma - psych_lambda) + psych_gamma                                      
        kwargs['sig_out'] = sig_out
        kwargs['psych_out'] = psych_out
        kwargs['y_true'] = y_true                                                                                          
        self.squared_difference.assign(tf.reduce_mean(tf.square(tf.cast(y_true, tf.float64) - sig_out), axis=0))

    def result(self):
#         precision = tf.math.divide_no_nan(
#             self.true_positives, self.true_positives + self.false_positives)
#         recall = tf.math.divide_no_nan(
#             self.true_positives, self.true_positives + self.false_negatives)
#         precision = tf.cast(precision, tf.float32)
#         recall = tf.cast(recall, tf.float32)
#         mul_value = precision * recall
#         add_value = (tf.math.square(self.beta) * precision) + recall
#         mean = (tf.math.divide_no_nan(mul_value, add_value))
#         f1_score = mean * (1 + tf.math.square(self.beta))

#         if self.average == 'weighted':
#             weights = tf.math.divide_no_nan(
#                 self.weights_intermediate,
#                 tf.reduce_sum(self.weights_intermediate))
#             f1_score = tf.reduce_sum(f1_score * weights)

#         elif self.average is not None:  # [micro, macro]
#             f1_score = tf.reduce_mean(f1_score)
        brier_score = tf.reduce_mean(self.squared_difference)
        return brier_score
    
    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
#             "num_classes": self.num_classes,
#             "average": self.average,
#             "beta": self.beta,
        }

#         if self.threshold is not None:
#             config["threshold"] = self.threshold

        base_config = super(LPUBrierScoreForY, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    
    
    
    
    
    

    
class LPUF1ScoreForL(tf.keras.metrics.Metric):
    """Base class to create scores for p(l|x) for LPU setting, i.e.
    when we want to classify X to be either l=1 or l=0 (annotated vs. not annotated)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision and recall
            in harmonic mean. Determines the weight given to the
            precision and recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-Beta Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
        ValueError: If the `beta` value is less than or equal
        to 0.
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    def __init__(self,
                 num_classes=2,
                 average=None,
                 beta=1.0,
                 threshold=None,
                 dtype=tf.float64,
                 model=None,
                 from_logits=True,
                 name='lpu_f1_score_for_l'
                ):
        super(LPUF1ScoreForL, self).__init__(name=name)
        if average not in (None, 'micro', 'macro', 'weighted'):
            raise ValueError("Unknown average type. Acceptable values "
                             "are: [None, micro, macro, weighted]")

        if not isinstance(beta, float):
            raise TypeError("The value of beta should be a python float")

        if beta <= 0.0:
            raise ValueError("beta value should be greater than zero")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError(
                    "The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")
        self.from_logits = from_logits
        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.init_shape = [num_classes]
        self.model = model
        if self.average != 'micro':
            self.axis = 0
            self.init_shape = [self.num_classes]
        def _zero_wt_init(name):
            return self.add_weight(
                name,
                shape=self.init_shape,
                initializer='zeros',
                dtype=self.dtype)

#         self.squared_difference = _zero_wt_init('squared_difference')
        self.true_positives = _zero_wt_init('true_positives')
        self.false_positives = _zero_wt_init('false_positives')
        self.false_negatives = _zero_wt_init('false_negatives')
        self.weights_intermediate = _zero_wt_init('weights_intermediate')

#     TODO: Add sample_weight support, currently it is
#     ignored during calculations.
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         tf.print("y_pred is:", y_pred, y_pred.shape, output_stream=sys.stderr)
#         tf.print("Model is:", self.model,  output_stream=sys.stderr)
#         if self.model is None:
#             pass
#         else:
#             self.custom_update_state(lambda x:self.model.keras_model(x))
        
#     def custom_update_state(output):
    
    @tf.function
    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_negatives.assign(tf.zeros(self.init_shape, self.dtype))
        self.weights_intermediate.assign(tf.zeros(self.init_shape, self.dtype))
        
        





#     def custom_update_state(output):
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
        batch_shape = tf.keras.backend.shape(y_true)[0]
        psych_layer_shape = self.model.keras_model.psych_layer_input_shape
        sig_layer_shape = self.model.keras_model.sig_layer_input_shape
        l_true = tf.math.divide(y_true, 2)
        l_true = tf.cast(l_true, tf.int32)        
#         tf.print("y_pred in update_state is:", y_pred, tf.shape(y_pred), output_stream=sys.stderr)
        g_prime, l_prime, psych_linear = tf.cast(y_pred[0, 0], tf.float64), tf.cast(y_pred[1, 0], tf.float64), tf.cast(y_pred[2: batch_shape + 2, :], tf.float64)
        sig_linear = y_pred[batch_shape + 3 + psych_layer_shape: 2 * batch_shape + 3 + psych_layer_shape]

        if self.from_logits:
            psych_gamma, psych_lambda = g_l_prime_to_gamma_lambda_transformer(g_prime, l_prime)
            sig_out = tf.sigmoid(sig_linear)
            psych_out = tf.sigmoid(psych_linear) * (1 - psych_gamma - psych_lambda) + psych_gamma                                      
        kwargs['sig_out'] = sig_out
        kwargs['psych_out'] = psych_out
        kwargs['l_true'] = l_true     
        l_pred = tf.multiply(sig_out, psych_out)
        l_true = tf.stack((1-l_true, l_true), axis=1)
        l_pred = tf.stack((1-l_pred, l_pred), axis=1)
        if self.threshold is None:
            threshold = tf.reduce_max(l_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            l_pred = tf.math.logical_and(l_pred >= threshold,
                                    tf.abs(l_pred) > 1e-12)
        else:
            l_pred = l_pred > self.threshold
        l_true = tf.cast(l_true, tf.int32)
        l_pred = tf.cast(l_pred, tf.int32)

#         y_pred = tf.cast(tf.cast((tf.sigmoid(self.model.keras_model.sig_linear) > 0.5), tf.int32), tf.float32)

#         self.squared_difference.assign_add(tf.reduce_mean(tf.square(tf.cast(l_true, tf.float64) - tf.multiply(sig_out, psych_out)), axis=0))
        def _count_non_zero(val):
            non_zeros = tf.math.count_nonzero(val, axis=self.axis)
            return tf.cast(non_zeros, self.dtype)
        result = _count_non_zero(l_pred * l_true)
        self.true_positives.assign_add(tf.reshape(_count_non_zero(l_pred * l_true), [2]))
        self.false_positives.assign_add(tf.reshape(_count_non_zero(l_pred * (l_true - 1)), [2]))
        self.false_negatives.assign_add(tf.reshape(_count_non_zero((l_pred - 1) * l_true), [2]))
        self.weights_intermediate.assign_add(tf.reshape(_count_non_zero(l_true), [2]))

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives)
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives)
        precision = tf.cast(precision, tf.float32)
        recall = tf.cast(recall, tf.float32)
        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = (tf.math.divide_no_nan(mul_value, add_value))
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == 'weighted':
            weights = tf.math.divide_no_nan(
                self.weights_intermediate,
                tf.reduce_sum(self.weights_intermediate))
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)
#         brier_score = tf.reduce_mean(self.squared_difference)
#         return brier_score
        return f1_score
    
    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
#             "num_classes": self.num_classes,
#             "average": self.average,
#             "beta": self.beta,
        }

#         if self.threshold is not None:
#             config["threshold"] = self.threshold

        base_config = super(LPUF1ScoreForL, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    
# class LPUROCAUCScoreForL(tf.keras.metrics.Metric):
#     """Base class to create scores for p(l|x) for LPU setting, i.e.
#     when we want to classify X to be either l=1 or l=0 (annotated vs. not annotated)
#     Args:
#         num_classes: Number of unique classes in the dataset.
#         average: Type of averaging to be performed on data.
#             Acceptable values are `None`, `micro`, `macro` and
#             `weighted`. Default value is None.
#         beta: Determines the weight of precision and recall
#             in harmonic mean. Determines the weight given to the
#             precision and recall. Default value is 1.
#         threshold: Elements of `y_pred` greater than threshold are
#             converted to be 1, and the rest 0. If threshold is
#             None, the argmax is converted to 1, and the rest 0.
#     Returns:
#         F-Beta Score: float
#     Raises:
#         ValueError: If the `average` has values other than
#         [None, micro, macro, weighted].
#         ValueError: If the `beta` value is less than or equal
#         to 0.
#     `average` parameter behavior:
#         None: Scores for each class are returned
#         micro: True positivies, false positives and
#             false negatives are computed globally.
#         macro: True positivies, false positives and
#             false negatives are computed for each class
#             and their unweighted mean is returned.
#         weighted: Metrics are computed for each class
#             and returns the mean weighted by the
#             number of true instances in each class.
#     """

#     def __init__(self,
#                  num_classes,
#                  average=None,
#                  beta=1.0,
#                  threshold=None,
#                  dtype=tf.float64,
#                  model=None,
#                  from_logits=True,
#                  name='lpu_roc_auc_score_for_l'
#                 ):
#         super(LPUF1ScoreForL, self).__init__(name=name)
#         if average not in (None, 'micro', 'macro', 'weighted'):
#             raise ValueError("Unknown average type. Acceptable values "
#                              "are: [None, micro, macro, weighted]")

#         if not isinstance(beta, float):
#             raise TypeError("The value of beta should be a python float")

#         if beta <= 0.0:
#             raise ValueError("beta value should be greater than zero")

#         if threshold is not None:
#             if not isinstance(threshold, float):
#                 raise TypeError(
#                     "The value of threshold should be a python float")
#             if threshold > 1.0 or threshold <= 0.0:
#                 raise ValueError("threshold should be between 0 and 1")
#         self.from_logits = from_logits
#         self.num_classes = num_classes
#         self.average = average
#         self.beta = beta
#         self.threshold = threshold
#         self.axis = None
#         self.init_shape = [num_classes]
#         self.model = model
#         if self.average != 'micro':
#             self.axis = 0
#             self.init_shape = [self.num_classes]
#         def _zero_wt_init(name):
#             return self.add_weight(
#                 name,
#                 shape=self.init_shape,
#                 initializer='zeros',
#                 dtype=self.dtype)

# #         self.squared_difference = _zero_wt_init('squared_difference')
#         self.true_positives = _zero_wt_init('true_positives')
#         self.false_positives = _zero_wt_init('false_positives')
#         self.false_negatives = _zero_wt_init('false_negatives')
#         self.weights_intermediate = _zero_wt_init('weights_intermediate')

# #     TODO: Add sample_weight support, currently it is
# #     ignored during calculations.
# #     def update_state(self, y_true, y_pred, sample_weight=None):
# #         tf.print("y_pred is:", y_pred, y_pred.shape, output_stream=sys.stderr)
# #         tf.print("Model is:", self.model,  output_stream=sys.stderr)
# #         if self.model is None:
# #             pass
# #         else:
# #             self.custom_update_state(lambda x:self.model.keras_model(x))
        
# #     def custom_update_state(output):
    
#     def reset_state(self):
#         self.true_positives.assign(tf.zeros(self.init_shape, self.dtype))
#         self.false_positives.assign(tf.zeros(self.init_shape, self.dtype))
#         self.false_negatives.assign(tf.zeros(self.init_shape, self.dtype))
#         self.weights_intermediate.assign(tf.zeros(self.init_shape, self.dtype))
        
        





# #     def custom_update_state(output):
#     @tf.function
#     def update_state(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
#         batch_shape = tf.keras.backend.shape(y_true)[0]
#         psych_layer_shape = self.model.keras_model.psych_layer_input_shape
#         sig_layer_shape = self.model.keras_model.sig_layer_input_shape
#         l_true = tf.math.divide(y_true, 2)
#         l_true = tf.cast(l_true, tf.int32)        
# #         tf.print("y_pred in update_state is:", y_pred, tf.shape(y_pred), output_stream=sys.stderr)
#         psych_gamma, psych_lambda, psych_linear = tf.cast(y_pred[0, 0], tf.float64), tf.cast(y_pred[1, 0], tf.float64), tf.cast(y_pred[2: batch_shape + 2, :], tf.float64)
#         sig_linear = y_pred[batch_shape + 3 + psych_layer_shape: 2 * batch_shape + 3 + psych_layer_shape]

#         if self.from_logits:
#             sig_out = tf.sigmoid(sig_linear)
#             psych_out = tf.sigmoid(psych_linear) * (1 - psych_gamma - psych_lambda) + psych_gamma                                      
#         kwargs['sig_out'] = sig_out
#         kwargs['psych_out'] = psych_out
#         kwargs['l_true'] = l_true     
#         l_pred = tf.multiply(sig_out, psych_out)
#         l_true = tf.stack((1-l_true, l_true), axis=1)
#         l_pred = tf.stack((1-l_pred, l_pred), axis=1)
#         if self.threshold is None:
#             threshold = tf.reduce_max(l_pred, axis=-1, keepdims=True)
#             # make sure [0, 0, 0] doesn't become [1, 1, 1]
#             # Use abs(x) > eps, instead of x != 0 to check for zero
#             l_pred = tf.math.logical_and(l_pred >= threshold,
#                                     tf.abs(l_pred) > 1e-12)
#         else:
#             l_pred = l_pred > self.threshold
#         l_true = tf.cast(l_true, tf.int32)
#         l_pred = tf.cast(l_pred, tf.int32)

# #         y_pred = tf.cast(tf.cast((tf.sigmoid(self.model.keras_model.sig_linear) > 0.5), tf.int32), tf.float32)

# #         self.squared_difference.assign_add(tf.reduce_mean(tf.square(tf.cast(l_true, tf.float64) - tf.multiply(sig_out, psych_out)), axis=0))
#         def _count_non_zero(val):
#             non_zeros = tf.math.count_nonzero(val, axis=self.axis)
#             return tf.cast(non_zeros, self.dtype)
#         result = _count_non_zero(l_pred * l_true)
#         self.true_positives.assign_add(tf.reshape(_count_non_zero(l_pred * l_true), [2]))
#         self.false_positives.assign_add(tf.reshape(_count_non_zero(l_pred * (l_true - 1)), [2]))
#         self.false_negatives.assign_add(tf.reshape(_count_non_zero((l_pred - 1) * l_true), [2]))
#         self.weights_intermediate.assign_add(tf.reshape(_count_non_zero(l_true), [2]))

#     def result(self):
#         precision = tf.math.divide_no_nan(
#             self.true_positives, self.true_positives + self.false_positives)
#         recall = tf.math.divide_no_nan(
#             self.true_positives, self.true_positives + self.false_negatives)
#         precision = tf.cast(precision, tf.float32)
#         recall = tf.cast(recall, tf.float32)
#         mul_value = precision * recall
#         add_value = (tf.math.square(self.beta) * precision) + recall
#         mean = (tf.math.divide_no_nan(mul_value, add_value))
#         f1_score = mean * (1 + tf.math.square(self.beta))

#         if self.average == 'weighted':
#             weights = tf.math.divide_no_nan(
#                 self.weights_intermediate,
#                 tf.reduce_sum(self.weights_intermediate))
#             f1_score = tf.reduce_sum(f1_score * weights)

#         elif self.average is not None:  # [micro, macro]
#             f1_score = tf.reduce_mean(f1_score)
# #         brier_score = tf.reduce_mean(self.squared_difference)
# #         return brier_score
#         return f1_score
    
#     def get_config(self):
#         """Returns the serializable config of the metric."""

#         config = {
# #             "num_classes": self.num_classes,
# #             "average": self.average,
# #             "beta": self.beta,
#         }

# #         if self.threshold is not None:
# #             config["threshold"] = self.threshold

#         base_config = super(LPUF1ScoreForL, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))    

def safe_brier_score_loss(y_true, y_prob):
    output = y_true - y_prob
    too_small = abs(output) < 1e-8
    output[too_small] = 0.
    return (output ** 2).mean()