from sklearn import metrics
import tensorflow as tf
from sys import path
# path.append('/home/scratch/nshajari/psych_model/LPUModels')
from dask.distributed import get_client, Client, wait
from sklearn.base import BaseEstimator
from dask import bag as db
import numpy as np
from joblib import parallel_backend
from LPUModels.PsychM import PsychM
from sklearn.base import clone
class MultiPsychM(PsychM):
    def __init__(self, num_of_psych_models=5, name='multi_psychm', *args, **kwargs):
        
        super().__init__(*args, **kwargs) 
        self.name = name
        self.num_of_psych_models = num_of_psych_models
        self.create_models()
#         self.set_params(self.psychm_model_list[0].get_params())
#     def __init__(self, *args, **kwargs):
        
    def create_models(self):
        self.psychm_model_list = []
        for i in range(self.num_of_psych_models):
            model = PsychM()
            model.set_params(**self.get_params())
            self.psychm_model_list.append(model)
    
#     def set_params(self, params):
#         super().set_params(**params)
#         for model in self.psychm_model_list:
#             model.set_params(**params)
        
        
#     def set_params(self, **parameters):
#         try:
#             for parameter, value in parameters.items():                    
#                 setattr(self, parameter, value)
#             if parameters['lap_kernel_params_pack'] is not None:
#                 for i, param in enumerate(['lbo_temperature', 'lap_kernel_lengthscale', 'lap_kernel_noise', 'lap_kernel_power', 'lap_kernel_var']):
#                     setattr(self, param, list(parameters['lap_kernel_params_pack'])[i])   

#             if parameters['gp_kernel_params_pack'] is not None:
#                 for i, param in enumerate(['gp_kernel_lengthscale', 'lap_kernel_var']):
#                     setattr(self, param, list(parameters['gp_kernel_params_pack'])[i])    
#         except:
#             TypeError("This is the NONE!!!")
#         return self  

#     def get_params(self, deep=True):
#         return_dict = {'encoder':self.encoder,
#         'num_of_psych_models': self.num_of_psych_models, 
#         'is_SPM': self.is_SPM,
#         'little_input_dict': self.little_input_dict, 
#         'sig_reg_coeff': self.sig_reg_coeff,
#         'sig_reg_penalty': self.sig_reg_penalty,
#         'ssl_type' : self.ssl_type,
#         'epochs' : self.epochs,
#         'is_fitted': self.is_fitted,
#         'psych_reg_coeff' : self.psych_reg_coeff,
#         'psych_reg_penalty' : self.psych_reg_penalty,
#         'metrics' : self.metrics,
#         'batch_size' : self.batch_size,
#         'workers' : self.workers,
#         'verbose' : self.verbose,
#         'child' : self.child,
#         'fit_counts': self.fit_counts,
#         'loss_type' : self.loss_type,
#         'use_multiprocessing' : self.use_multiprocessing,
#         'keras_model_initialized' : self.keras_model_initialized,
#         'calibrate' : self.calibrate,
#         'warm_f_g_prime_init' : self.warm_f_g_prime_init,
#         'warm_f_l_prime_init' : self.warm_f_l_prime_init,
#         'f_g_prime_init' : self.f_g_prime_init,
#         'f_l_prime_init' : self.f_l_prime_init,
#         'alternate_training' : self.alternate_training,
#         'constrained_optimization' : self.constrained_optimization,
#         'barrier_initial_C' : self.barrier_initial_C,
#         'optimizer_dict' : self.optimizer_dict,
#         'warm_cv_params' : self.warm_cv_params,
#         'gp_kernel_lengthscale' : self.gp_kernel_lengthscale,
#         'gp_kernel_lengthscale_trainable' : self.gp_kernel_lengthscale_trainable,
#         'gp_kernel_var' : self.gp_kernel_var,
#         'lap_kernel_lengthscale': self.lap_kernel_lengthscale,
#         'lap_kernel_lengthscale_trainable': self.lap_kernel_lengthscale_trainable,
#         'lap_kernel_var': self.lap_kernel_var,
#         'lap_kernel_var_trainable': self.lap_kernel_var_trainable,
#         'noise_sigma' : self.noise_sigma,
#         'lr_reduce_min_delta' : self.lr_reduce_min_delta,
#         'end_training_min_delta' : self.end_training_min_delta,
#         'dropout_rate' : self.dropout_rate,
#         'lap_connectivity' : self.lap_connectivity,
#         'lap_kernel_type' : self.lap_kernel_type,
#         'gp_kernel_type' : self.gp_kernel_type,
#         'lbo_temperature' : self.lbo_temperature,
#         'lap_kernel_noise' : self.lap_kernel_noise,
#         'lap_kernel_power' : self.lap_kernel_power,
#         'lap_kernel_normed' : self.lap_kernel_normed,
#         'g_prime' : self.g_prime,
#         'l_prime' : self.l_prime,
#         'psych_alpha' : self.psych_alpha,
#         'psych_beta' : self.psych_beta,
#         'sig_a' : self.sig_a,
#         'sig_b' : self.sig_b,
#         'psych_alpha' : self.psych_alpha,
#         'psych_beta' : self.psych_beta,
#         'psych_lambda' : self.psych_lambda,
#         'psych_gamma' : self.psych_gamma,
#         'alternate_history_list': self.alternate_history_list,
#         'alternate_param_reporter_list': self.alternate_param_reporter_list,
#         'alternate_descend_epoch_list': self.alternate_descend_epoch_list,
#         'parameter_reporter': self.parameter_reporter,
#         'gp_kernel_params_pack': self.gp_kernel_params_pack,
#         'lap_kernel_params_pack': self.lap_kernel_params_pack,
#         'number_of_successful_attempts': self.number_of_successful_attempts,
#         'sampled_sig_input_features': self.sampled_sig_input_features,
#         'kernel': self.kernel
#         }
#         try:
#             if self.lap_kernel_params_pack is not None:
#                 for i, param in enumerate(['lbo_temperature', 'lap_kernel_lengthscale', 'lap_kernel_noise', 'lap_kernel_power', 'lap_kernel_var']):
#                     return_dict[param] = list(self.lap_kernel_params_pack.values)[i]

#             if self.gp_kernel_params_pack is not None:
#                 for i, param in enumerate(['gp_kernel_lengthscale','gp_lap_kernel_var']):
#                     return_dict[param] = list(self.gp_kernel_params_pack.values)[i]
#         except Exception as e:
#             if hasattr(e, 'message'):
#                 raise type(e)(e.message+'get_params has a problem' )
#             else:
#                 raise type(e)(str(e)+'get_params has a problem' )

#         return return_dict
#     def initialize_model(self, sampled_sig_input_features=None, reinitialize=True, set_fit_params=False):
#         for model in self.psychm_model_list:
#             if sampled_sig_input_features is None:
#                 model.initialize_model(self.sampled_sig_input_features, reinitialize, set_fit_params)
#             else:
#                 model.initialize_model(sampled_sig_input_features, reinitialize, set_fit_params)
        
    def fit(self, X, y, validation_data=None, c_estimation_method=None, weights=None, reinitialize=True, loss_type = 'loss'):
        for model in self.psychm_model_list:
            model.set_params(**self.get_params())
#             print ("manifold_kernel_k:", model.manifold_kernel_k, self.manifold_kernel_k, self.get_params())
#         client = Client('192.168.6.99:8786') 
        client = get_client()
#         self.client = LocalCluster()
#         client.restart()
#         client.restart()
        
#         client.upload_file('utils/func_lib.py')
#         client.upload_file('utils/LogUniformProduct.py')
#         client.upload_file('utils/IdentityTransformer.py') 
#         client.upload_file('utils/TransformToCOO.py') 
#         client.upload_file('utils/MultipleVectorizer.py') 
#         client.upload_file('utils/RepresentationPacker.py') 
#         client.upload_file('utils/math_utils.py') 
#         client.upload_file('utils/tensor_utils.py') 
#         client.upload_file('LPUModels/MyModel.py') 
#         client.upload_file('LPUModels/naive_clf.py') 
#         client.upload_file('LPUModels/KMEModel.py') 
#         client.upload_file('LPUModels/PropensityEM.py') 
#         client.upload_file('LPUModels/PsychometricLayer.py') 
#         client.upload_file('LPUModels/GPLayer.py')                 
#         client.upload_file('LPUModels/PsychMKeras.py') 
#         client.upload_file('utils/scorer_library.py') 
#         client.upload_file('LPUModels/PsychM.py') 
#         client.upload_file('LPUModels/puAdapterTF.py') 
#         client.upload_file('utils/text_utils.py') 
#         client.upload_file('DataFrameVectorizer.py') 
#         client.upload_file('bootstrap_library.py') 
#         args_list = list(map(list, zip([X] * self.num_of_psych_models, [y] * self.num_of_psych_models, [validation_data] * self.num_of_psych_models, self.psychm_model_list)))
        args_list = []
        for model in self.psychm_model_list:
            args_list.append([X, y, validation_data, model])
        b = db.from_sequence(args_list)
        def parallel_fitter(args_list):
            X, y, validation_data, clf = args_list
            clf.fit(X, y, validation_data, c_estimation_method, weights, reinitialize)
            return clf
#         print ("ARGS LIST:", args_list)
#         for m in self.psychm_model_list:
#             m.fit(X, y, validation_data, c_estimation_method, weights, reinitialize)

        with parallel_backend('threading') as backend:
            client.scatter(X)
            client.scatter(y) 
            client.scatter(validation_data)
            client.scatter(self.psychm_model_list)
            futures = b.map(parallel_fitter).compute()
        wait(futures)
        client.wait_for_workers()
        self.psychm_model_list = futures
        loss_list = []
        l = (self.encoder.inverse_transform(y)/2).astype(int)
        if loss_type == 'brier':
            for model in self.psychm_model_list:
                loss_list.append(metrics.brier_score_loss(l, model.predict_proba(X)))
        else:
            for model in self.psychm_model_list:
                loss_list.append(model.alternating_best_loss)
                
        self.best_estimator_ = self.psychm_model_list[np.argmin(loss_list)]
        self.psychm_model_list = []
        self.set_params(**self.best_estimator_.get_params())            
        for param_name, param_value in self.best_estimator_.get_params().items():
#             eval('self.'+param_name+'_=param_value')
            setattr(self, param_name+'_', param_value)
            setattr(self, param_name, param_value)
    
            
        
    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)
    
    def predict_prob_y_given_x(self, X):
        return self.best_estimator_.predict_prob_y_given_x(X)

    def predict_prob_l_given_y_x(self, X):
        return self.best_estimator_.predict_prob_l_given_y_x(X)
    
#         for future in futures:
#             print ("PsychM:", future.kernel)
