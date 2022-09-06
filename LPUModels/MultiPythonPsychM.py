from sklearn import metrics
import tensorflow as tf
from sys import path
path.append('/home/scratch/nshajari/psych_model/LPUModels')

from dask.distributed import get_client, Client, wait
from sklearn.base import BaseEstimator
from dask import bag as db
import numpy as np
from joblib import parallel_backend
from PythonPsychM import PythonPsychM



class MultiPythonPsychM(BaseEstimator):
    def __init__(self, encoder=None,  sig_a=None, sig_b=None, psych_alpha=None, psych_beta=None, psych_gamma=None, psych_lambda=None, g_prime=None, l_prime=None, kernel=None, loss_type=None,
        is_SPM=False,
        sig_reg_coeff=None,
        sig_reg_penalty=tf.keras.regularizers.L2,
        ssl_type=None, 
        psych_reg_coeff=None,
        psych_reg_penalty=tf.keras.regularizers.L2,
        gp_kernel_lengthscale=None,                 
        gp_kernel_var=None,
        lap_kernel_lengthscale=None,
        lap_kernel_var=None,
        lap_kernel_k = None,
        lap_connectivity=None,
        lap_kernel_type=None,
        gp_kernel_type=None,
        lbo_temperature=None,
        lap_kernel_noise=None,
        lap_kernel_power=None,
        lap_kernel_normed=None,
        num_of_ppsych_models=5):
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
        self.num_of_ppsych_models = num_of_ppsych_models
        self.loss_type = loss_type
        self.create_models()
#     def __init__(self, *args, **kwargs):
        
    def create_models(self):
#         from PythonPsychM import PythonPsychM
        self.psychm_model_list = []
        for i in range(self.num_of_ppsych_models):
            self.psychm_model_list.append(PythonPsychM(
        encoder=self.encoder, 
        g_prime=self.g_prime,
        l_prime=self.l_prime,
        sig_a=self.sig_a,
        sig_b=self.sig_b,
        psych_alpha=self.psych_alpha,
        psych_beta = self.psych_beta,
        psych_lambda = self.psych_lambda,
        psych_gamma = self.psych_gamma,
        is_SPM = self.is_SPM,
        gp_kernel_var= self.gp_kernel_var,
        lap_kernel_var = self.lap_kernel_var,
        lap_kernel_k=self.lap_kernel_k,
        lap_connectivity =self.lap_connectivity,
        lap_kernel_type =self.lap_kernel_type,
        gp_kernel_type=self.gp_kernel_type,
        gp_kernel_lengthscale =self.gp_kernel_lengthscale,
        lbo_temperature =self.lbo_temperature,
        lap_kernel_lengthscale =self.lap_kernel_lengthscale,
        lap_kernel_noise =self.lap_kernel_noise,
        lap_kernel_power =self.lap_kernel_power,
        lap_kernel_normed =self.lap_kernel_normed,
        sig_reg_coeff =self.sig_reg_coeff,
        sig_reg_penalty =self.sig_reg_penalty,
        psych_reg_coeff =self.psych_reg_coeff,
        psych_reg_penalty =self.psych_reg_penalty,
        ssl_type=self.ssl_type,
            ))

    def fit(self, X, y, validation_data=None, c_estimation_method='sar-em', weights=None, reinitialize=True, num_of_ppsych_models=5, loss_type = 'loss', warm_start=False):
        for model in self.psychm_model_list:
            model.set_params(**self.get_params())
#         client = Client('192.168.6.80:8786') 
        client = get_client()
#         client.upload_file('/home/scratch/nshajari/psych_model/LPUModels/PythonPsychM.py') 

        args_list = []
        for i in range(self.num_of_ppsych_models):
            args_list.append([X, y, validation_data, self.psychm_model_list[i]])
        b = db.from_sequence(args_list)
        def parallel_fitter(args_list):
            X, y, validation_data, clf = args_list
            clf.fit(X, y, warm_start=False)
            return clf

        with parallel_backend('threading') as backend:
            client.scatter(X)
            client.scatter(y) 
            client.scatter(validation_data)
            client.scatter(self.psychm_model_list)
            futures = b.map(parallel_fitter).compute()
        client.wait_for_workers()
        wait(futures)
        self.psychm_model_list = futures
        loss_list = []
        l = (self.encoder.inverse_transform(y)/2).astype(int)
        if loss_type == 'brier':
            for model in self.psychm_model_list:
                loss_list.append(metrics.brier_score_loss(l, model.predict_proba(X)))
        else:
            for model in self.psychm_model_list:
                print (model.final_loss, model.final_success)
                loss_list.append(model.final_loss)
                
        self.best_estimator_ = self.psychm_model_list[np.argmin(loss_list)]
        self.psychm_model_list = []
        self.set_params(**self.best_estimator_.get_params())            
        for param_name, param_value in self.best_estimator_.get_params().items():
            setattr(self, param_name+'_', param_value)
            
        
    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)
    
    def predict_prob_y_given_x(self, X):
        return self.best_estimator_.predict_prob_y_given_x(X)

    def predict_prob_l_given_y_x(self, X):
        return self.best_estimator_.predict_prob_l_given_y_x(X)
