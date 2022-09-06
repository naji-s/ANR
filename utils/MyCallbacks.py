import tensorflow as tf
import numpy as np
import logging
import warnings
import dill as pickle
import sys
import sys
sys.path.append('/home/scratch/nshajari/psych_model/puLearning')
sys.path.append('/home/scratch/nshajari/psych_model/utils')
sys.path.append('/home/scratch/nshajari/psych_model/LPUModels')

from utils.func_lib import g_l_prime_to_gamma_lambda_transformer

class TrainingSetPerturber(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.X_train['sig_input'] += tf.random.uniform(all_train_sig_input_features.shape, minval=0., maxval=tf.math.reduce_std(self.model.X_train['sig_input']) * 0.2, dtype=tf.dtypes.float64,
    )
class ParameterReporter(tf.keras.callbacks.Callback):
    def __init__(self, *args, **kwargs):
        super(ParameterReporter, self).__init__(*args, **kwargs)        

    def on_train_begin(self, logs=None):
        self.weights = []
        self.gamma_hist = []
        self.lambda_hist = []
        self.a_norm_1_hist = []
        self.a_norm_2_hist = []
        self.a_norm_laplacian_hist = []
        self.a_norm_kernel_hist = []
        self.a_norm_inf_hist = []
        self.a_norm_neg_inf_hist = []
        self.alpha_norm_1_hist = []
        self.alpha_norm_2_hist = []
        self.alpha_norm_inf_hist = []
        self.alpha_norm_neg_inf_hist = []
        self.gp_lengthscale_hist = []
        self.lap_lengthscale_hist = []
        
#         self.model.parent.full_weight_history = []

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.psych_layer
        if self.model.psych_layer.g_prime is not None:
            try:
                g_prime = self.model.psych_layer.g_prime.value()[0]
                l_prime = self.model.psych_layer.l_prime.value()[0]
            except Exception as e:
                raise type(e)(str(e) + "it is epoch_end call in MyCallbacks that fucks things u0p") 
                
            gamma_, lambda_ = g_l_prime_to_gamma_lambda_transformer(g_prime, l_prime)#, 
        else:
            gamma_ = -1
            lambda_ = -1
#         reparametrization_style=self.model.reparametrization_style
                
#         if self.model.reparametrization_style in [1, 3, 4, 5]:
#             g_prime = abs(g_prime)
#             l_prime = abs(l_prime)
#             gamma_ = g_prime / (g_prime + l_prime + 1)
#             lambda_ = l_prime / (g_prime + l_prime + 1)            
#         elif self.model.reparametrization_style == 6:
#             elkan_c = self.model.psych_layer.elkan_c.value()
#             gamma_ = conv_to_prob(g_prime) * elkan_c
#             lambda_ = conv_to_prob(l_prime) * (1-elkan_c)
#         elif self.model.reparametrization_style == 7:
#             elkan_c = self.model.psych_layer.elkan_c.value()
#             abs_g = abs(g_prime)
#             abs_l = abs(l_prime)
#             abs_sum = abs_g + abs_l + 1.
#             gamma_ = (1 + elkan_c) / 2 * abs_g / abs_sum
#             lambda_ = (2-elkan_c) / 2 * abs_l / abs_sum
#         elif self.model.reparametrization_style == 8:
#             gamma_ = np.exp(g_prime) / (1 + np.exp(g_prime) + np.exp(l_prime))
#             lambda_ = np.exp(l_prime) / (1 + np.exp(g_prime) + np.exp(l_prime))
#         elif self.model.reparametrization_style == 10:
#             gamma_ = g_prime
#             lambda_ = l_prime
        alpha = self.model.psych_layer.alpha.value()
        a = self.model.sig_layer.alpha.value()
        self.a_norm_1_hist.append(np.linalg.norm(a, ord=1))
        self.a_norm_2_hist.append(np.linalg.norm(a, ord=2))
        self.a_norm_inf_hist.append(np.linalg.norm(a, ord=np.inf))
        self.a_norm_neg_inf_hist.append(np.linalg.norm(a, ord=-np.inf))
                                  
        self.alpha_norm_1_hist.append(np.linalg.norm(alpha, ord=1))
        self.alpha_norm_2_hist.append(np.linalg.norm(alpha, ord=2))
        self.alpha_norm_inf_hist.append(np.linalg.norm(alpha, ord=np.inf))
        self.alpha_norm_neg_inf_hist.append(np.linalg.norm(alpha, ord=-np.inf))
        self.gamma_hist.append(gamma_)
        self.lambda_hist.append(lambda_)
        if self.model.gp_kernel_lengthscale_trainable:
            self.gp_lengthscale_hist.append(self.model.sig_layer.gp_kernel_lengthscale.value)
            self.lap_lengthscale_hist.append(self.model.sig_layer.manifold_kernel_lengthscale.value)
        
    def on_train_end(self, logs=None):
        self.model.parent.keras_fitting_final_weights = self.model.get_weights()
        self.model.parent.child = None
        self.model.stop_training = True
        self.model = None


    # class EndTraining(tf.keras.callbacks.EarlyStopping):
#     def __init__(self, *args, **kwargs):
#         super(EndTraining, self).__init__(*args, **kwargs)
# #         self.getting_better = False
# #     def on_epoch_begin(self, epoch, logs=None):
# #         self.weights_before_learning = self.model.get_weights()
# #     def on_train_end(self, epoch,logs=None):
# #         self.model = None

        
# class EndTraining(tf.keras.callbacks.Callback):
#     """Stop training when a monitored metric has stopped improving.

#     Assuming the goal of a training is to minimize the loss. With this, the
#     metric to be monitored would be 'loss', and mode would be 'min'. A
#     `model.fit()` training loop will check at end of every epoch whether
#     the loss is no longer decreasing, considering the `min_delta` and
#     `patience` if applicable. Once it's found no longer decreasing,
#     `model.stop_training` is marked True and the training terminates.

#     The quantity to be monitored needs to be available in `logs` dict.
#     To make it so, pass the loss or metrics at `model.compile()`.

#     Example:

#     >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
#     >>> # This callback will stop the training when there is no improvement in
#     >>> # the validation loss for three consecutive epochs.
#     >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
#     >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
#     >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
#     ...                                         epochs=10, batch_size=1, callbacks=[callback],
#     ...                                         verbose=0)
#     >>> len(history.history['loss'])    # Only 4 epochs are run.
#     4
#     """


#     def __init__(self,
#                              monitor='val_loss',
#                              min_delta=0,
#                              patience=0,
#                              verbose=0,
#                              mode='min',
#                              baseline=None,
#                              restore_best_weights=False):
#         """Initialize an EarlyStopping callback.

#         Arguments:
#                 monitor: Quantity to be monit ored.
#                 min_delta: Minimum change in the monitored quantity
#                         to qualify as an improvement, i.e. an absolute
#                         change of less than min_delta, will count as no
#                         improvement.
#                 patience: Number of epochs with no improvement
#                         after which training will be stopped.
#                 verbose: verbosity mode.
#                 mode: One of `{"auto", "min", "max"}`. In `min` mode,
#                         training will stop when the quantity
#                         monitored has stopped decreasing; in `max`
#                         mode it will stop when the quantity
#                         monitored has stopped increasing; in `auto`
#                         mode, the direction is automatically inferred
#                         from the name of the monitored quantity.
#                 baseline: Baseline value for the monitored quantity.
#                         Training will stop if the model doesn't show improvement over the
#                         baseline.
#                 restore_best_weights: Whether to restore model weights from
#                         the epoch with the best value of the monitored quantity.
#                         If False, the model weights obtained at the last step of
#                         training are used.
#         """
#         super(EndTraining, self).__init__()


#         self.monitor = monitor
#         self.patience = patience
#         self.verbose = verbose
#         self.baseline = baseline
#         self.min_delta = abs(min_delta)
#         self.wait = 0
#         self.stopped_epoch = 0
#         self.restore_best_weights = restore_best_weights
#         self.best_weights = None
#         self.ending_activated = False


#         if mode not in ['auto', 'min', 'max']:
#             logging = tf.get_logger()
#             logging.warning('EarlyStopping mode %s is unknown, '
#                                             'fallback to auto mode.', mode)
#             mode = 'auto'


#         if mode == 'min':
#             self.monitor_op = np.less
#         elif mode == 'max':
#             self.monitor_op = np.greater
#         else:
#             if 'acc' in self.monitor:
#                 self.monitor_op = np.greater
#             else:
#                 self.monitor_op = np.less


#         if self.monitor_op == np.greater:
#             self.min_delta *= 1
#         else:
#             self.min_delta *= -1


#     def on_train_begin(self, logs=None):
#         # Allow instances to be re-used
#         self.wait = 0
#         self.stopped_epoch = 0
#         if self.baseline is not None:
#             self.best = self.baseline
#         else:
#             self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            

                

                
    

#     def on_epoch_end(self, epoch, logs=None):
#         current = self.get_monitor_value(logs)
#         if current is None:
#             return
#         if self.monitor_op(np.abs(self.min_delta), np.Inf if self.best in [np.Inf, -np.Inf] else np.true_divide(np.abs(current - self.best), np.abs(self.best))):
#             self.best = current
#             self.wait = 0
#             if self.restore_best_weights:
#                 self.model.best_weights = self.model.get_weights()
#         else:
#             self.wait += 1
#             if self.wait >= self.patience:
#                 self.stopped_epoch = epoch
#                 self.model.stop_training = True
#                 self.ending_activated = True
#                 if self.restore_best_weights:
#                     if self.verbose > 0:
#                         print('Restoring model weights from the end of the best epoch.')
#                     self.model.set_weights(self.model.best_weights)


#     def on_train_end(self, logs=None):
#         if self.stopped_epoch > 0 and self.verbose > 0:
#             print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


#     def get_monitor_value(self, logs):
#         logs = logs or {}
#         monitor_value = logs.get(self.monitor)
#         if monitor_value is None:
#             logging = tf.get_logger()
#             logging.warning('Early stopping conditioned on metric `%s` '
#                                             'which is not available. Available metrics are: %s',
#                                             self.monitor, ','.join(list(logs.keys())))
#         return monitor_value

    
# class ReduceLROnRelativePlateau(tf.keras.callbacks.Callback):
#     """Reduce learning rate when a metric has stopped improving.
#     Models often benefit from reducing the learning rate by a factor
#     of 2-10 once learning stagnates. This callback monitors a
#     quantity and if no improvement is seen for a 'patience' number
#     of epochs, the learning rate is reduced.
#     Example:
#     ```python
#     reduce_lr = ReduceLROnRelativePlateau(monitor='val_loss', factor=0.2,
#                                                                 patience=5, min_lr=0.001)
#     model.fit(X_train, Y_train, callbacks=[reduce_lr])
#     ```
#     Arguments:
#             monitor: quantity to be monitored.
#             factor: factor by which the learning rate will be reduced. new_lr = lr *
#                 factor
#             patience: number of epochs with no improvement after which learning rate
#                 will be reduced.
#             verbose: int. 0: quiet, 1: update messages.
#             mode: one of {auto, min, max}. In `min` mode, lr will be reduced when the
#                 quantity monitored has stopped decreasing; in `max` mode it will be
#                 reduced when the quantity monitored has stopped increasing; in `auto`
#                 mode, the direction is automatically inferred from the name of the
#                 monitored quantity.
#             min_delta: threshold for measuring the new optimum, to only focus on
#                 significant changes.
#             cooldown: number of epochs to wait before resuming normal operation after
#                 lr has been reduced.
#             min_lr: lower bound on the learning rate.
#     """

#     def __init__(self,monitor='val_loss',
#                              factor=0.1,
#                              patience=10,
#                              verbose=0,
#                              mode='min',
#                              min_delta=1e-4,
#                              cooldown=0,
#                              min_lr=0,
#                              **kwargs):
        
#         super(ReduceLROnRelativePlateau, self).__init__(monitor=monitor,
#                              factor=factor,
#                              patience=patience,
#                              verbose=verbose,
#                              mode=mode,
#                              min_delta=min_delta,
#                              cooldown=cooldown,
#                              min_lr=min_lr,
#                              **kwargs)

#         self.monitor = monitor
#         if factor >= 1.0:
#             raise ValueError('ReduceLROnRelativePlateau ' 'does not support a factor >= 1.0.')
#         if 'epsilon' in kwargs:
#             min_delta = kwargs.pop('epsilon')
#             logging = tf.get_logger()
#             logging.warning('`epsilon` argument is deprecated and '
#                                             'will be removed, use `min_delta` instead.')
#         self.factor = factor
#         self.min_lr = min_lr
#         self.min_delta = min_delta
#         self.patience = patience
#         self.verbose = verbose
#         self.cooldown = cooldown
#         self.cooldown_counter = 0    # Cooldown counter.
#         self.wait = 0
#         self.best = 0
#         self.mode = mode
#         self.monitor_op = None
#         self._reset()

#     def _reset(self):
#         """Resets wait counter and cooldown counter.
#         """
#         if self.mode not in ['auto', 'min', 'max']:
#             logging = tf.get_logger()
#             logging.warning('Learning Rate Plateau Reducing mode %s is unknown, '
#                                             'fallback to auto mode.', self.mode)
#             self.mode = 'auto'
#         if (self.mode == 'min' or
#                 (self.mode == 'auto' and 'acc' not in self.monitor)):
#             self.monitor_op = lambda a, b: np.less(self.min_delta, np.Inf if self.best == np.Inf else np.true_divide(np.abs(a - b), np.abs(b)))
#             self.best = np.Inf
#         else:
#             self.monitor_op = lambda a, b:  lambda a, b: np.less(a, b - self.min_delta)
#             self.best = -np.Inf
#         self.cooldown_counter = 0
#         self.wait = 0

#     def on_train_begin(self, logs=None):
#         self._reset()

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         logs['learning_rate'] =  tf.keras.backend.get_value(self.model.optimizer.lr)
#         current = logs.get(self.monitor)
#         if current is None:
#             logging = tf.get_logger()
#             logging.warning('Reduce LR on plateau conditioned on metric `%s` '
#                                             'which is not available. Available metrics are: %s',
#                                             self.monitor, ','.join(list(logs.keys())))

#         else:
#             if self.in_cooldown():
#                 self.cooldown_counter -= 1
#                 self.wait = 0

#             if self.monitor_op(current, self.best):
#                 self.best = current
#                 self.wait = 0
#             elif not self.in_cooldown():
#                 self.wait += 1
#                 if self.wait >= self.patience:
#                     old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
#                     if old_lr > self.min_lr:
#                         new_lr = old_lr * self.factor
#                         new_lr = max(new_lr, self.min_lr)
#                         tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
#                         if self.verbose > 0:
#                             print('\nEpoch %05d: ReduceLROnRelativePlateau reducing learning '
#                                         'rate to %s.' % (epoch + 1, new_lr))
#                         self.cooldown_counter = self.cooldown
#                         self.wait = 0

#     def in_cooldown(self):
#         return self.cooldown_counter > 0    
    
    
# class EndTrainingWithLookback(tf.keras.callbacks.Callback):
#     """Stop training when a monitored metric has stopped improving.

#     Assuming the goal of a training is to minimize the loss. With this, the
#     metric to be monitored would be 'loss', and mode would be 'min'. A
#     `model.fit()` training loop will check at end of every epoch whether
#     the loss is no longer decreasing, considering the `min_delta` and
#     `patience` if applicable. Once it's found no longer decreasing,
#     `model.stop_training` is marked True and the training terminates.

#     The quantity to be monitored needs to be available in `logs` dict.
#     To make it so, pass the loss or metrics at `model.compile()`.

#     Example:

#     >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
#     >>> # This callback will stop the training when there is no improvement in
#     >>> # the validation loss for three consecutive epochs.
#     >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
#     >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
#     >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
#     ...                                         epochs=10, batch_size=1, callbacks=[callback],
#     ...                                         verbose=0)
#     >>> len(history.history['loss'])    # Only 4 epochs are run.
#     4
#     """


#     def __init__(self,
#                              monitor='val_loss',
#                              min_delta=0,
#                              patience=0,
#                              verbose=0,
#                              mode='min',
#                              baseline=None,
#                              k=10, 
#                              restore_best_weights=False):
#         """Initialize an EarlyStopping callback.

#         Arguments:
#                 monitor: Quantity to be monit ored.
#                 min_delta: Minimum change in the monitored quantity
#                         to qualify as an improvement, i.e. an absolute
#                         change of less than min_delta, will count as no
#                         improvement.
#                 patience: Number of epochs with no improvement
#                         after which training will be stopped.
#                 verbose: verbosity mode.
#                 mode: One of `{"auto", "min", "max"}`. In `min` mode,
#                         training will stop when the quantity
#                         monitored has stopped decreasing; in `max`
#                         mode it will stop when the quantity
#                         monitored has stopped increasing; in `auto`
#                         mode, the direction is automatically inferred
#                         from the name of the monitored quantity.
#                 baseline: Baseline value for the monitored quantity.
#                         Training will stop if the model doesn't show improvement over the
#                         baseline.
#                 restore_best_weights: Whether to restore model weights from
#                         the epoch with the best value of the monitored quantity.
#                         If False, the model weights obtained at the last step of
#                         training are used.
#         """
#         super(EndTrainingWithLookback, self).__init__()


#         self.monitor = monitor
#         self.patience = patience
#         self.verbose = verbose
#         self.baseline = baseline
#         self.min_delta = abs(min_delta)
#         self.wait = 0
#         self.stopped_epoch = 0
#         self.restore_best_weights = restore_best_weights
#         self.best_weights = None
#         self.ending_activated = False
#         self.k = k


#         if mode not in ['auto', 'min', 'max']:
#             logging = tf.get_logger()
#             logging.warning('EarlyStopping mode %s is unknown, '
#                                             'fallback to auto mode.', mode)
#             mode = 'auto'


#         if mode == 'min':
#             self.monitor_op = np.less
#             self.set_min_in_k = lambda arr: np.min(arr[-self.k:])
#             self.set_min_in_2k_k = lambda arr: np.min(arr[-2 * self.k:-self.k])
#         elif mode == 'max':
#             self.monitor_op = np.greater
#             self.set_min_in_k = lambda arr: np.max(arr[-self.k:])
#             self.set_min_in_2k_k = lambda arr: np.max(arr[-2 * self.k:-self.k])
#         else:
#             if 'acc' in self.monitor:
#                 self.monitor_op = np.greater
#             else:
#                 self.monitor_op = np.less


#         if self.monitor_op == np.greater:
#             self.min_delta *= 1
#         else:
#             self.min_delta *= -1


#     def on_train_begin(self, logs=None):
#         # Allow instances to be re-used
#         self.wait = 0
#         self.stopped_epoch = 0
#         self.monitor_list = []
        
#         if self.baseline is not None:
#             self.best_in_last_k = self.baseline
#             self.best_in_last_2k_k = self.baseline
#         else:
#             self.best_in_last_k = np.Inf if self.monitor_op == np.less else -np.Inf
#             self.best_in_last_2k_k = np.Inf if self.monitor_op == np.less else -np.Inf


#     def on_epoch_end(self, epoch, logs=None):
#         current = self.get_monitor_value(logs)
#         self.monitor_list.append(current)
#         if current is None:
#             return
#         if epoch < self.k * 2:
#             print ("epoch is", epoch)
#             return
#         self.best_in_last_k = self.set_min_in_k(self.monitor_list)
#         self.best_in_last_2k_k = self.set_min_in_2k_k(self.monitor_list)
#         if epoch == self.k * 2 or self.monitor_op(np.abs(self.min_delta), np.true_divide(np.abs(self.best_in_last_k - self.best_in_last_2k_k), np.abs(self.best_in_last_k))):
#             self.wait = 0
#             if self.restore_best_weights:
#                 self.best_weights = self.model.get_weights()
#         else:
#             self.wait += 1
#             if self.wait >= self.patience:
#                 self.stopped_epoch = epoch
#                 self.model.stop_training = True
#                 self.ending_activated = True
#                 if self.restore_best_weights:
#                     if self.verbose > 0:
#                         print('Restoring model weights from the end of the best epoch.')
#                     self.model.set_weights(self.best_weights)


#     def on_train_end(self, logs=None):
#         if self.stopped_epoch > 0 and self.verbose > 0:
#             print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


#     def get_monitor_value(self, logs):
#         logs = logs or {}
#         monitor_value = logs.get(self.monitor)
#         if monitor_value is None:
#             logging = tf.get_logger()
#             logging.warning('Early stopping conditioned on metric `%s` '
#                                             'which is not available. Available metrics are: %s',
#                                             self.monitor, ','.join(list(logs.keys())))
#         return monitor_value
    
    
class ReduceLROnRelativePlateauWithLookback(tf.keras.callbacks.Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    Example:
    ```python
    reduce_lr = ReduceLROnRelativePlateau(monitor='val_loss', factor=0.2,
                                                                patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    Arguments:
            monitor: quantity to be monitored.
            factor: factor by which the learning rate will be reduced. new_lr = lr *
                factor
            patience: number of epochs with no improvement after which learning rate
                will be reduced.
            verbose: int. 0: quiet, 1: update messages.
            mode: one of {auto, min, max}. In `min` mode, lr will be reduced when the
                quantity monitored has stopped decreasing; in `max` mode it will be
                reduced when the quantity monitored has stopped increasing; in `auto`
                mode, the direction is automatically inferred from the name of the
                monitored quantity.
            min_delta: threshold for measuring the new optimum, to only focus on
                significant changes.
            cooldown: number of epochs to wait before resuming normal operation after
                lr has been reduced.
            min_lr: lower bound on the learning rate.
    """

    def __init__(self, monitor='loss',
                             factor=.5,
                             patience=10,
                             verbose=0,
                             mode='min',
                             min_delta=1e-4,
                             cooldown=0,
                             min_lr=0,
                             k=20,
                             restore_best_weights=True,
                             only_positive=True,
                             *args, 
                             **kwargs):
        super(ReduceLROnRelativePlateauWithLookback, self).__init__(*args, **kwargs)        
        
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        if factor > 1.0:
            raise ValueError('ReduceLROnRelativePlateau ' 'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging = tf.get_logger()
            logging.warning('`epsilon` argument is deprecated and '
                                            'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.k = k
        self.min_lr = min_lr
        self.halving_count = 0
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        # cooldown PARAMETER FOR NOW NOT IN USE!!! AS OPPOSED TO IT'S ORIGINAL USE IN REDUCE_LR FROM KERAS IMPLEMENTATION!
        # I'D RATHER SET IT MYSELF WITH self.k
#         self.cooldown = cooldown
#         self.cooldown_counter = 2 * self.k    # Cooldown counter.
#         self.best = np.inf
        self.gradient_norm = np.inf
        self.mode = mode
        self._reset()
        self.resetting_counter = 0
        self.only_positive = only_positive
    def monitor_acceptable_change(self, a, b):
        if b == np.Inf:
            return True
        
        division_result = np.true_divide(a - b, np.abs(b))
#         division_result = a - b
        
        if np.isnan(division_result) or np.isinf(division_result):
            return None
        else: 
            return np.greater(self.min_delta, division_result)
        
    def monitor_op(self, a, b): 
        if b == np.Inf:
            return True
        
        division_result = np.true_divide(b - a, np.abs(b))        
#         division_result = b - a
        
        if np.isnan(division_result) or np.isinf(division_result):
            return None
        else: 
            return np.less(self.min_delta, division_result) 

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        self.learning_rate_changing = False
        self.best = np.inf
        if self.mode not in ['auto', 'min', 'max']:
            logging = tf.get_logger()
            logging.warning('Learning Rate Plateau Reducing mode %s is unknown, '
                                            'fallback to auto mode.', self.mode)
            self.mode = 'min'

        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.best_monitor_op = lambda a, b: np.less(a, b)
            self.set_min_in_k = lambda arr: np.min(arr[-self.k:])
            self.set_min_in_2k_k = lambda arr: np.min(arr[-2 * self.k:-self.k])
            self.monitor_sign = lambda x: x > 0
        
#         elif self.mode == 'max':
#             self.monitor_op = lambda a, b:  lambda a, b: np.less(a, b - self.min_delta)
#             self.set_min_in_k = lambda arr: np.max(arr[-self.k:])
#             self.set_min_in_2k_k = lambda arr: np.max(arr[-2 * self.k:-self.k])
            
            
        self.cooldown_counter = 2 * self.k
        self.terminate_wait = 0
        self.wait = 0
        self.last_2k_weights_list = []
        self.last_2k_optimizer_weights_2k_list = []
        self.monitor_list = []
        self.first_run = True
        self.descend_epochs = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        output_1_loss = logs.get('output_1_loss')
        output_2_loss = logs.get('output_2_loss')
        output_3_loss = logs.get('output_3_loss')
        output_4_loss = logs.get('output_4_loss')
        loss_gradient_norm = logs.get('loss_gradient_norm')
#         tf.print("LOSS:", loss)
#         tf.print("LOSS GRADIENT NORM:", loss_gradient_norm)
        
        if loss is not None:        
            if np.isnan(loss) or np.isinf(loss):
                print ("TRAINING IS FUCKED!!")
#                 print ("Weights are:", self.model.get_weights())
#                 if self.model.psych_layer.g_prime is not None and self.model.psych_layer.l_prime is not None:
#                     print ("g_prime and l_prime are:", tf.keras.backend.get_value(self.model.psych_layer.g_prime), tf.keras.backend.get_value(self.model.psych_layer.l_prime))
                logging = tf.get_logger()
                logging.warning('We are in the BATCH'+ str(batch) +','+ str(loss) + ',' + str(val_loss) + ',' + str(output_1_loss) + ',' + str(output_2_loss) + ', '+ str(output_3_loss) + ', ' + str(output_4_loss))
                logging.warning('TRAINING IS FUCKED DUE TO NANS')
#                 self.model.stop_training = True
#                 self.model.reset_training = True
        
    def on_train_begin(self, logs=None):        
#         self._reset()
        self.best_weights = self.model.get_weights()
        self.best_optimizer_weights = self.model.optimizer.get_weights()

    def on_epoch_begin(self, epoch, logs=None):
        self.last_metric_value = self.get_monitor_value(logs) 
            
            
        self.last_epoch_weights = self.model.get_weights()
        self.last_epoch_optimizer_weights = self.model.optimizer.get_weights()
        self.first_epoch_weights = self.model.get_weights()
        self.first_epoch_optimizer_weights = self.model.optimizer.get_weights()
        self.first_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
#         if logs is not None:
#         current_brier = logs.get('lpu_brier_score_for_l')
#         tf.print("Brier and loss WTF???:", current_brier,  current, output_stream=sys.stderr)
#         current_brier = current
#         beta_score = logs.get('fbeta_score')
#         if beta_score is not None:
#             if beta_score[0] > .8 and beta_score[1] > 0.8:
#                 address_base = '/home/scratch/nshajari/psych_model/test/'+str(id(self))
#                 self.model.save(address_base+'keras_model')
#                 with open (address_base+'.pkl', 'wb') as f:
#                     pickle.dump([self.model.parent.X_train_, self.model.parent.l_y_decimal_train_, logs, self.model.sig_layer.kernel], f)
#                 tf.print ('************************************------------------------------********************')
        
        
        if current is None:
            logging = tf.get_logger()
#             logging.warning('Reduce LR on plateau conditioned on metric `%s` '
#                                             'which is not available. Available metrics are: %s',
#                                             self.monitor, ','.join(list(logs.keys())))

        else:
            
#             if hasattr(self, 'last_metric_value'):
#                 if self.last_metric_value is not None and self.last_metric_value < 0. and current <= self.last_metric_value:
#                     self.reset_training = True
#                     self.stop_training = True
#             if not self.monitor_sign(current):
#                     self.reset_training = True
#                     self.stop_training = True
                

            loss_gradient_norm = logs.get('loss_gradient_norm')
#             if np.isnan(current) or np.isinf(current):
#                 self.model.set_weights(self.last_epoch_weights)
# #                 self.model.optimizer.set_weights(self.last_epoch_optimizer_weights)
#                 old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
#                 new_lr = old_lr * self.factor
#                 if new_lr < 1e-8:
#                     self.model.set_weights(self.first_epoch_weights)
#                     self.model.optimizer.set_weights(self.first_epoch_optimizer_weights)
                    
#                     if self.resetting_counter > 3:
#                         self.model.stop_training = True
#                     if self.model.psych_layer.alpha._trainable == False:
#                         a_init = tf.random.truncated_normal(self.model.sig_layer.alpha.shape, mean=0.0, stddev=0.01, dtype=tf.dtypes.float64, seed=None, name=None)
#                         self.model.sig_layer.alpha.assign(self.model.parent.warm_a)
#                         self.model.sig_layer.beta.assign([self.model.parent.warm_b])
#                     elif self.model.sig_layer.alpha._trainable == False:
#                         alpha_init = tf.random.truncated_normal(self.model.psych_layerpickle.shape, mean=0.0, stddev=0.01, dtype=tf.dtypes.float64, seed=None, name=None)
#                         self.model.psych_layer.alpha.assign(alpha_init)
#                         self.model.psych_layer.beta.assign([np.random.randn(1)])
#                     else:                              
#                         a_init = tf.random.truncated_normal(self.model.sig_layer.alpha.shape, mean=0.0, stddev=0.01, dtype=tf.dtypes.float64, seed=None, name=None)
#                         self.model.sig_layer.alpha.assign(self.model.parent_warm_a)
#                         self.model.sig_layer.beta.assign([self.model.parent_warm_b])
#                         alpha_init = tf.random.truncated_normal(self.model.psych_layer.alpha.shape, mean=0.0, stddev=0.01, dtype=tf.dtypes.float64, seed=None, name=None)
#                         self.model.psych_layer.alpha.assign(alpha_init)
#                         self.model.psych_layer.beta.assign([np.random.randn(1)])

#                     self.first_lr = self.first_lr / 2.
#                     new_lr = self.first_lr
#                     self.resetting_counter += 1
#                     self._reset()
#                     if self.verbose > 0:
#                         print('\epoch %05d: resetting to a previous  epoch is not helping. Resetting completely...')        
#                 else:
#                     if self.verbose > 0:
#                         print('\epoch %05d: NaN noticed... reducing learning'
#                                         'rate to %s. and reverting to the best state in the last 2k steps' % (epoch + 1, new_lr))        
#                 tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            
#             else:            
#             if np.isnan(current):
#                 return
            gradient_norm = 0.
#             for item in self.model.gradient:
#                 gradient_norm += np.linalg.norm(tf.keras.backend.get_value(item))
            
            if len(self.last_2k_weights_list) > 2 * self.k - 1:
                self.last_2k_weights_list.pop(0)
                self.last_2k_optimizer_weights_2k_list.pop(0)
            self.last_2k_weights_list.append(self.model.get_weights())
            self.last_2k_optimizer_weights_2k_list.append(self.model.optimizer.get_weights())
            self.monitor_list.append(current)
            self.num_of_accepted_steps = len(self.monitor_list)
#             self.model.parent.full_weight_history.append(.dump.get_weights())

            if np.isnan(current) or np.isinf(current):
                if epoch < self.model.parent.epochs * 0.5:
                    self.model.parent.reset_training = True
                    self.model.reset_training = True
                self.model.stop_training = True
                return

            logs = logs or {}
            logs['learning_rate'] =  tf.keras.backend.get_value(self.model.optimizer.lr)

            if self.best_monitor_op(current, self.best):
                self.best_weights = self.model.get_weights()
                self.best_optimizer_weights = self.model.optimizer.get_weights()
                self.best = current
            if self.in_cooldown():
#                 tf.print ("cool donw running")
                self.cooldown_counter -= 1
                return


            self.best_in_last_k = self.set_min_in_k(self.monitor_list)
            self.best_in_last_2k_k = self.set_min_in_2k_k(self.monitor_list)
#             with warnings.filterwarnings("error"):
            is_acceptable_change = self.monitor_acceptable_change(self.best_in_last_k, self.best_in_last_2k_k)
#             is_acceptable_change = True
            is_less = self.monitor_op(self.best_in_last_k, self.best_in_last_2k_k)
        
            if is_less and is_acceptable_change:
                self.descend_epochs.append(epoch)
            else:
                if not is_acceptable_change:
#                     tf.print("BAD THINGS HAPPENING...")#, self.best_in_last_2k_k, self.best_in_last_k, self.monitor_list[-20:])
                    # FIDNING THE BEST INDEXT IN THE PAST 2K EXPLORATIONS
                    best_place = np.argmin(self.monitor_list[-2 * self.k:])
                    stub_len = len(self.monitor_list[:-2*self.k])
                    try:
                        assert (self.monitor_list[stub_len + best_place] == min(self.best_in_last_2k_k, self.best_in_last_k))
                    except AssertionError as e:
                        e.args += ('CHECKING FAILED 1', self.monitor_list[stub_len + best_place], self.best_in_last_2k_k, self.best_in_last_k, self.monitor_list, 42)
                        raise e               

                    if self.best > np.min(self.monitor_list[:-2*self.k]):
                        self.best_weights = self.last_2k_weights_list[best_place]
                        self.best_optimizer_weights = self.last_2k_optimizer_weights_2k_list[best_place]
                        self.best = self.monitor_list[-2 * self.k:][best_place]
                    self.model.set_weights(self.last_2k_weights_list[best_place])
                    self.model.optimizer.set_weights(self.last_2k_optimizer_weights_2k_list[best_place])
                    # DELETING EVERYTHING FROM FUTURE THAT WAS USELESS EXPLORATION
                    self.monitor_list = self.monitor_list[:-2 * self.k + best_place + 1]
                    self.last_2k_weights_list = self.last_2k_weights_list[:best_place + 1]
                    self.last_2k_optimizer_weights_2k_list = self.last_2k_optimizer_weights_2k_list[: best_place + 1]

#                     self.descend_epochs = self.descend_epochs[:best_place + 1]

#                     for key, value in logs.items():
#                         if type(value).__name__ == 'list':
#                             if len(value) > 0:
#                                 logs[key] = value[: best_place + 1]

                    self.cooldown_counter = np.max([2 * self.k - len(self.monitor_list), self.k, 2 * self.k - best_place - 1])
                    return

                else:
                    self.cooldown_counter = max(2 * self.k - len(self.monitor_list), 0)



                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:# and self.halving_count <= 5:
                    self.halving_count += 1
                    self.learning_rate_changing = True
                    new_lr = old_lr * self.factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    try:
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnRelativePlateau reducing learning '
                                        'rate to %s.' % (epoch + 1, new_lr))
                    except Exception as e:
                        raise type(e)(str(e)+'problem of string is here in MyCallbacks' )

                else: 
                    self.model.stop_training = True
                    self.ending_activated = True
                    self.learning_rate_changing = False
# #                     if hasattr(e, 'message'):
#                         if "invalid value encountered in double_scalars" in e.message:
#                             print ("CAUGHT THE MOFO!")
#                             self.model.stop_training = True
#                             self.model.reset_training = True
#                             return
#                         else:
#                             print ("FUCK", e.message)


            self.last_epoch_optimizer_weights = self.model.optimizer.get_weights()
            self.last_epoch_weights = self.model.get_weights()
        

    def in_cooldown(self):
        return self.cooldown_counter > 0    

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging = tf.get_logger()
#             logging.warning('Early stopping conditioned on metric `%s` '
#                                             'which is not available. Available metrics are: %s',
#                                             self.monitor, ','.join(list(logs.keys())))
        return monitor_value    