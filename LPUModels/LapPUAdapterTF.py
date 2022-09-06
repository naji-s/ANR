import sys
sys.path.append('/home/nshajari/master_thesis/')
sys.path.append('/Users/naji/Box Sync/Box Sync/CMU/Masters Thesis/master_thesis/')
# from utils.text_utils import tokenize
from miscellaneous.Kernel_MPE_grad_threshold import wrapper

#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
Created on Dec 21, 2012

@author: Alexandre
"""
import os
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

# from utils.text_utils import tokenize
# from LPUModels.GPLayer import invert_mat_with_cholesky
# from dask_ml.linear_model import LogisticRegressionCV
# from utils.func_lib import find_optimal_threshold
DELTA = 1e-4
import tensorflow as tf
EPSILON = 1e-16
from GPLayer import build_manifold_mat, build_W, invert_mat_with_cholesky

from scipy.linalg import lapack

def upper_triangular_to_symmetric(ut):
    ut += np.triu(ut, k=1).T

def fast_positive_definite_inverse(m):
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        raise ValueError('dpotrf failed on input {}'.format(m))
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError('dpotri failed on input {}'.format(cholesky))
    upper_triangular_to_symmetric(inv)
    return inv

class LapPUAdapterTF(BaseEstimator):
    """
    Adapts any probabilistic binary classifier to positive-unlabled learning using the PosOnly method proposed by
    Elkan and Noto:

    Elkan, Charles, and Keith Noto. \"Learning classifiers from only positive and unlabeled data.\"
    Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.
    """


    def __init__(self, hold_out_ratio=None, 
                 penalty=None, 
                 encoder=None,
                 preprocessing_type=None, 
                 calibrate=False, 
                 maxiter=100, 
                 tol=1e-6, 
                 gp_kernel_type=None, 
                 manifold_regularize=False, 
                 sampled_input_features=None, 
                manifold_kernel_amplitude=None,
                manifold_kernel_lengthscale=None,
                gp_kernel_amplitude=None,
                gp_kernel_lengthscale=None,
                manifold_neighbor_mode=None,
                manifold_kernel_k=None, 
                 manifold_kernel_power=None, 
                 manifold_kernel_noise=None, 
                 optimal_threshold=True, 
                 use_eigsh=None, 
                 svc_cache=None,
                estimator_type=None,
                 scar_method=None,
                 kme_kernel_mode=None
                ):
        """
        estimator -- An estimator of p(s=1|x) that must implement:
                     * predict_proba(X): Takes X, which can be a list of feature vectors or a precomputed
                                         kernel matrix and outputs p(s=1|x) for each example in X
                     * fit(X,y): Takes X, which can be a list of feature vectors or a precomputed
                                 kernel matrix and takes y, which are the labels associated to the
                                 examples in X
        hold_out_ratio -- The ratio of training examples that must be held out of the training set of examples
                          to estimate p(s=1|y=1) after training the estimator
        precomputed_kernel -- Specifies if the X matrix for predict_proba and fit is a precomputed kernel matrix
        """

        self.optimal_threshold = optimal_threshold
        self.svc_cache = svc_cache
        self.manifold_kernel_amplitude = manifold_kernel_amplitude
        self.manifold_kernel_lengthscale = manifold_kernel_lengthscale
        self.gp_kernel_amplitude = gp_kernel_amplitude
        self.gp_kernel_lengthscale = gp_kernel_lengthscale
        self.manifold_neighbor_mode = manifold_neighbor_mode
        self.manifold_kernel_noise = manifold_kernel_noise
        self.manifold_kernel_k = manifold_kernel_k
        self.manifold_kernel_power = manifold_kernel_power
        self.gp_kernel_type = gp_kernel_type
        self.sampled_input_features = sampled_input_features
        self.manifold_regularize = manifold_regularize
        self.tol = tol
        self.maxiter = maxiter
        self.calibrate = calibrate
        self.use_eigsh = use_eigsh
        self.estimator_type = estimator_type
        self.scar_method = scar_method
        self.kme_kernel_mode = kme_kernel_mode
#         if None not in [sig_vec, sig_vec_params]:
#             self.multiple_representations = True
#         else:
#             self.multiple_representations = False
#         self.sig_vec = sig_vec
#         self.sig_vec_params = sig_vec_params
        
        self.encoder = encoder
        if penalty is None:
            self.penalty = 'l2'
        else:
            self.penalty = penalty
            
            
        if hold_out_ratio is None:
            self.hold_out_ratio = 0.1
        else:
            self.hold_out_ratio = hold_out_ratio
        self.preprocessing_type = preprocessing_type

        # if precomputed_kernel:
        #     self.fit = self.__fit_precomputed_kernel
        # else:
        #     self.fit = self.__fit_no_precomputed_kernel
#         self.esimtator = None
        self.estimator_fitted = False
    
    def svd_decompose(self, Lap_mat, padding=False):
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
    

    
    
    
    
    
    def check_laplacian_kernel(self):
        if self.gp_kernel_type == 'linear':
            self.original_kernel_func = lambda X, Y: (linear_kernel(X, Y) / \
                                                      (2*self.gp_kernel_lengthscale**2)) * \
                                                            self.gp_kernel_amplitude ** 2 
        elif self.gp_kernel_type == 'rbf':
            self.original_kernel_func = lambda X, Y: rbf_kernel(X, Y, 1 / (2*self.gp_kernel_lengthscale**2))*\
            self.gp_kernel_amplitude ** 2          
        else:
            raise NotImplementedError("The kernel for Kernel matrix can only be rbf or linear")
        K = self.original_kernel_func(self.sampled_input_features, self.sampled_input_features)
        if self.manifold_regularize:
            # if 'lbo' in self._ssl_type.lower():
            #     method = self._I_plus_KM_inv_M_calc_method #  'invert_M_first' or invert_I_plus_MK
            #     I_plus_MK_inverter = self._I_plus_KM_inv_M_using_factorization
            #     with_expm_acting= self._I_plus_KM_inv_M_using_expm_acting
            #     use_eigsh = self._I_plus_KM_inv_M_using_eigsh
            # elif 'laplacian' in self._ssl_type.lower():
            #     method = self._I_plus_KM_inv_M_calc_method # or 'invert_M_first'
            I_plus_MK_inverter = 'invert_M_first'
            with_expm_acting = False
            # use_eigsh = self._I_plus_KM_inv_M_using_eigsh
            # else:
            #     raise NotImplementedError("No other SSL method available")

        #         original_kernel_mat, self._manifold_mat, self._manifold_mat_inv,\
        #         self._Lap_mat, decomposition_dict, self._original_kernel_func =\
        #                                     calculate_main_kernels(self.sampled_input_features, 
        #                                                                 use_eigsh=use_eigsh,
        #                                                                 return_manifold_mat_inv=return_manifold_mat_inv)

        #         self.new_kernel =calculate_mixed_kernel(manifold_mat=self._manifold_mat, 
        #                                                              manifold_mat_inv=self._manifold_mat_inv,
        #                                                              kernel_mat=original_kernel_mat, 
        #                                                              Lap_mat=self._Lap_mat, 
        #                                                              decomposition_dict=decomposition_dict, 
        #                                                              I_plus_MK_inverter=I_plus_MK_inverter, 
        #                                                              with_expm_acting=with_expm_acting, 
        #                                                              method=method)            
            return_manifold_mat_inv = True
            manifold_kernel_type = 'laplacian'
            manifold_kernel_normed = False
            lbo_temperature = 0.0001
            self.W = build_W(self.sampled_input_features, k_neighbours=self.manifold_kernel_k,
                        lengthscale=self.manifold_kernel_lengthscale, connectivity=self.manifold_neighbor_mode)

            self.manifold_mat, Lap_mat, decomposition_dict, self.manifold_mat_inv =\
                                build_manifold_mat(self.W, manifold_kernel_noise=self.manifold_kernel_noise, 
                                                        manifold_kernel_type=manifold_kernel_type, 
                                                        manifold_kernel_power=self.manifold_kernel_power, 
                                                        manifold_kernel_amplitude=self.manifold_kernel_amplitude, 
                                                        lbo_temperature=lbo_temperature, 
                                                        manifold_kernel_normed=manifold_kernel_normed, 
                                                        use_eigsh=self.use_eigsh, return_inverse=return_manifold_mat_inv)        

            self.M_inv_plus_K_inv = invert_mat_with_cholesky(self.manifold_mat_inv + K)
    
    def new_kernel(self, X, Y, gamma=None):
        k_x = self.original_kernel_func(self.sampled_input_features, X)
        k_y = self.original_kernel_func(self.sampled_input_features, Y)
        return self.original_kernel_func(X, Y) - k_x.T @ self.M_inv_plus_K_inv @ k_y
        
        

        

    def fit(self, X, y, cv=False, gammas=None, Cs=1., random_cv=None, sample_size=None, method='elkan'):
        """
        Fits an estimator of p(s=1|x) and estimates the value of p(s=1|y=1,x)

        X -- List of feature vectors
        y -- Labels associated to each feature vector in X (Positive label: 1.0, Negative label: -1.0)
        """
        # if self.calibrate:
        #     from sklearn.model_selection import train_test_split
        #     X, X_cal, y, y_cal = train_test_split(X, y, stratify=y)
        #     l_y_decimal_cal = self.encoder.inverse_transform(y_cal)
        #     y_cal, _ = (l_y_decimal_cal/2).astype(int), np.mod(l_y_decimal_cal, 2).reshape((-1, 1))
            
        # imidiately decoding the categorical variable l_y_cat to l and y, replacing l with y and implicitly 
        # dropping real class (y) by assigning it to _
        X_ = X['sig_input']
        
        l_y_decimal = self.encoder.inverse_transform(y)
        if self.scar_method == 'real_clf':
            _, l = (l_y_decimal/2).astype(int), np.mod(l_y_decimal, 2).reshape((-1, 1))
        else:
            l, _ = (l_y_decimal/2).astype(int), np.mod(l_y_decimal, 2).reshape((-1, 1))

        X_ = X_.astype(np.float64)
        l = l.astype(np.float64)


        l = np.copy(l).astype(np.float64)

        from sklearn.model_selection import train_test_split
        self.X_remaining, self.X_hold_out, self.l_remaining, self.l_hold_out = train_test_split(X_, l, test_size=int(len(l) * self.hold_out_ratio))
        if self.scar_method in ['real_clf', 'naive_lpu']:
            self.X_remaining = X_
            self.l_remaining = l

        self.sampled_input_features = self.X_remaining
        self.check_laplacian_kernel()
        if self.gp_kernel_type is None:
            if self.penalty == 'l2':
                self.estimator = LogisticRegression(C=self.C, solver='lbfgs', penalty=self.penalty, tol=self.tol, 
                                                max_iter=self.maxiter, random_state=2022)
            else:
                self.estimator = LogisticRegression(C=self.C, solver='liblinear', penalty=self.penalty, 
                                                    tol=self.tol, max_iter=self.maxiter, random_state=2022)
            # if random_cv:
            #     Cs = np.random.choice(Cs, size=sample_size)
            if cv:
                self.estimator = LogisticRegression(C=1. / self.gp_kernel_amplitude ** 2,tol=self.tol, solver='lbfgs', penalty='l2', max_iter=self.maxiter, random_state=2022)
                
        else:
                
            from sklearn.svm import SVC
            if self.manifold_regularize:
                self.kernel = self.new_kernel
            else:
                self.kernel = self.original_kernel_func

            if self.estimator_type == 'logistic':
                self.estimator = LogisticRegression(C=1.,tol=self.tol, solver='lbfgs',max_iter=self.maxiter, random_state=2022)
            elif self.estimator_type == 'svc':
                # raise NotImplementedError("notice that the constant factor for kernel amplitude and the C in SVC can screw up the final solution.  This needs further investigation")
                self.estimator = SVC(C=1., gamma=1., kernel=self.kernel, tol=self.tol, 
                                     probability=True, cache_size=self.svc_cache, max_iter=self.maxiter,random_state=2022)
            else:
                raise NotImplementedError("estimator_type needs to be 'logistic' or 'svc'")
                
        


        if self.estimator_type is None:
            self.estimator.fit(self.kernel(self.X_remaining, self.X_remaining), self.l_remaining)
        else:
            self.estimator.fit(self.X_remaining, self.l_remaining)

            

        # print self.estimator.classes_
        if self.estimator_type is None:
            X_hold_out_kernel_embedding = self.kernel(self.X_hold_out, self.sampled_input_features)
        else:
            X_hold_out_kernel_embedding = self.X_hold_out
            
        hold_out_predictions = self.estimator.predict_proba(X_hold_out_kernel_embedding)
        
        # try:
        hold_out_predictions = hold_out_predictions[:,1]

        if self.scar_method == 'elkan':
            self.c = np.mean(hold_out_predictions)
        elif self.scar_method == 'kme':
            if type(X_).__name__ == 'csr_matrix':
                kappa_2, kappa_1 = wrapper(self.X_hold_out.toarray(), self.X_hold_out[self.l_hold_out==1].toarray())
            else:
                kappa_2, kappa_1 = wrapper(self.X_hold_out.astype(np.float64), self.X_hold_out[self.l_hold_out==1].astype(np.float64))
            if self.kme_kernel_mode == 1:
                self.c = self.l_hold_out.mean() / kappa_1
            else:
                self.c = self.l_hold_out.mean() / kappa_2
        elif self.scar_method in ['real_clf', 'naive_lpu']:
                self.c = 1.
        if self.c > 1:
            logging = tf.get_logger()
            logging.warning("The estimated value for c in SCAR method:" + self.scar_method +\
                            " is larger than 1. Thresholding to 1.")
            self.c = 1
        self.estimator_fitted = True
        return self
    
    def predict_prob_l_given_y_x(self, X):
        X_ = X['sig_input']
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        return self.c * np.ones(X_.shape[0])
        
    
    def predict_prob_y_given_x(self, X):
        """
        Predicts p(y=1|x) using the estimator and the value of p(s=1|y=1) estimated in fit(...)

        X -- List of feature vectors or a precomputed kernel matrix
        """

        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict_proba(...).')

        X_ = X['sig_input']

        if self.estimator_type is None:
            X_ = self.kernel(X_, self.sampled_input_features)
        
        output = self.estimator.predict_proba(X_)[:, 1]
        output = output / self.c
        output [output>1-EPSILON] = 1-EPSILON
        return output
    
    def predict_y_given_x(self, X):
        return self.predict_prob_y_given_x(X)>=0.5 * self.c

    def predict_proba(self, X):
        X_ = X['sig_input']
        if self.estimator_type is None:
            X_ = self.kernel(X_, self.sampled_input_features)

        return self.estimator.predict_proba(X_)[:, 1]


    def predict(self, X, treshold=0.5):
        """
        Assign labels to feature vectors based on the estimator's predictions

        X -- List of feature vectors or a precomputed kernel matrix
        treshold -- The decision treshold between the positive and the negative class
        """
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict(...).')

        return np.array([1. if p > treshold else 0. for p in self.predict_proba(X)])
        
        

