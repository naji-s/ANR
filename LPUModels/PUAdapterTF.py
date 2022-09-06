#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
Created on Dec 21, 2012

@author: Alexandre
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import make_scorer, brier_score_loss
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
# from utils.text_utils import tokenize
# from dask_ml.linear_model import LogisticRegressionCV
from utils.scorer_library import flexible_scorer
class PUAdapterTF(BaseEstimator):
    """
    Adapts any probabilistic binary classifier to positive-unlabled learning using the PosOnly method proposed by
    Elkan and Noto:

    Elkan, Charles, and Keith Noto. \"Learning classifiers from only positive and unlabeled data.\"
    Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.
    """


    def __init__(self, hold_out_ratio=None, penalty=None, C = None, encoder=None,
#         sig_vec=None, sig_vec_params=None, 
                 preprocessing_type=None, calibrate=False, maxiter=1000, tol=1e-6, kernel_type=None, gamma=None, svc_cache=None):
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
        self.kernel_type = kernel_type
        self.tol = tol
        self.svc_cache = svc_cache
        self.maxiter = maxiter
        self.calibrate = calibrate
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
            
        if C is None:
            self.C = 1.
        else:
            self.C = C
            
        self.gamma = gamma
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
        
#     def __str__(self):
#         return 'Estimator:' + str(self.estimator) + '\n' + 'p(s=1|y=1,x) ~= ' + str(self.c) + '\n' + \
#             'Fitted: ' + str(self.estimator_fitted)
    
    
#     def __fit_precomputed_kernel(self, X, y):
#         """
#         Fits an estimator of p(s=1|x) and estimates the value of p(s=1|y=1) using a subset of the training examples

#         X -- Precomputed kernel matrix
#         y -- Labels associated to each example in X (Positive label: 1.0, Negative label: -1.0)
#         """
#         positives = np.where(y == 1.)[0]
#         hold_out_size = np.ceil(len(positives) * self.hold_out_ratio)
#         print ("Holdout size for Elkan's method: " + hold_out_size)

#         if len(positives) <= hold_out_size:
#             raise('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')
        
#         np.random.shuffle(positives)
#         hold_out = positives[:hold_out_size]
        
#         #Hold out test kernel matrix
#         X_test_hold_out = X[hold_out]
#         keep = list(set(np.arange(len(y))) - set(hold_out))
#         X_test_hold_out = X_test_hold_out[:, keep]
        
#         #New training kernel matrix
#         X_ = X[:, keep]
#         X_ = X[keep]

#         y = np.delete(y, hold_out)
        
#         self.estimator.fit(X, y)
        
#         hold_out_predictions = self.estimator.predict_proba(X_test_hold_out)
        
#         try:
#             hold_out_predictions = hold_out_predictions[:,1]
#         except:
#             pass
        
#         c = np.mean(hold_out_predictions)
#         self.c = c
        
#         self.estimator_fitted = True
        
        
    # def __fit_no_precomputed_kernel(self, X, y):
    def fit(self, X, y, cv=False, gammas=None, Cs=1., random_cv=None, sample_size=None):
        """
        Fits an estimator of p(s=1|x) and estimates the value of p(s=1|y=1,x)

        X -- List of feature vectors
        y -- Labels associated to each feature vector in X (Positive label: 1.0, Negative label: -1.0)
        """
        if self.calibrate:
            from sklearn.model_selection import train_test_split
            X, X_cal, y, y_cal = train_test_split(X, y, stratify=y)
            l_y_decimal_cal = self.encoder.inverse_transform(y_cal)
            y_cal, _ = (l_y_decimal_cal/2).astype(int), np.mod(l_y_decimal_cal, 2).reshape((-1, 1))
        # imidiately decoding the categorical variable l_y_cat to l and y, replacing l with y and implicitly 
        # dropping real class (y) by assigning it to _
        l_y_decimal = self.encoder.inverse_transform(y)
        l, _ = (l_y_decimal/2).astype(int), np.mod(l_y_decimal, 2).reshape((-1, 1))
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        X_ = X['sig_input']
        # the line to take care of extracting s from the passed argument in form of s_y
#         if np.ndim(y) > 1:
#             y = y[:, 0]
        if self.kernel_type is None:
            if self.penalty == 'l2':
                self.estimator = LogisticRegression(C=self.C, solver='lbfgs', penalty=self.penalty, tol=self.tol, 
                                                max_iter=self.maxiter, random_state=2022)
            else:
                self.estimator = LogisticRegression(C=self.C, solver='liblinear', penalty=self.penalty, 
                                                    tol=self.tol, max_iter=self.maxiter, random_state=2022)
            # if random_cv:
            #     Cs = np.random.choice(Cs, size=sample_size)
            if cv:
                # def fit_estimator(args):#model):
                #     model, X, l_y = args
                #     model.fit(X, l_y)
                #     # model.fit(transformed_X_train, l_y_cat_transformed_train)
                #     return model
                # if 'spm' in 
                # parallel_rerunlist = [delayed(fit_estimator)([model, transformed_X_train, l_y_cat_transformed_train]) for model in copied_best_model_list]
                # from dask.distributed import get_client
            # client = get_client()
#                 strat_kfold = StratifiedKFold(3, shuffle=True, random_state=1100)
#                 param_grid = {'Cs': Cs,'clf__penalty':['l2'],                       
#                                         'gammas': gammas, 
# #                                         'clf__sig_vec': [default_transformation],
#                                         'clf__tol': [1e-6],
#                                         'clf__maxiter': [1000],
#                                        }
#                 model_cv = RandomizedSearchCV(
#                                 LogisticRegression(),
#                                 n_iter=sample_size,
#                                 scoring=make_scorer(lambda y, y_pred: metrics.roc_auc_score(y, y_pred, average='micro')),
#                                 cv=strat_kfold,
#                                 param_distributions=param_grid,
#                                 n_jobs=-1,
#                                 random_state=2022
#                             )

                # self.estimator = model_cv
                # # with parallel_backend('threading') as backend:
                # copied_best_model_results = dask.compute(*parallel_rerunlist)
                # self.estimator = LogisticRegressionCV(Cs=Cs, cv=cv,  solver='lbfgs', penalty='l2', 
                #                                     tol=self.tol, max_iter=self.maxiter, random_state=2022,
                #                                     scoring=make_scorer(lambda y, y_pred: metrics.roc_auc_score(y, y_pred, average='micro')))
                self.estimator = LogisticRegression(tol=self.tol, solver='lbfgs', penalty='l2', max_iter=self.maxiter, random_state=2022)
                
        else:
            if cv:
                self.gamma = 50
                self.C = .01
                
            from sklearn.svm import SVC
            self.estimator = SVC(C=self.C, gamma=self.gamma, kernel=self.kernel_type, tol=self.tol, probability=True, cache_size=self.svc_cache, max_iter=self.maxiter, random_state=2022)
                
                
        
#         X_ = X[:]



        from sklearn.utils import shuffle
        l = np.copy(l).astype(np.float64)
        positives = np.where(l == 1.)[0]
        shuffled_positive = shuffle(positives, random_state=2022)
        hold_out_size = int(np.ceil(len(shuffled_positive) * self.hold_out_ratio))
        # print ("Holdout size for Elkan's method: " + str(hold_out_size))
        if len(positives) <= hold_out_size:
            raise('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')
        hold_out = shuffled_positive[:hold_out_size]
        temp = np.zeros(l.shape[0]) 
        temp[hold_out] = 1
        hold_out = temp.astype(bool)
        #print ("THE SHAPES ARE: ", hold_out_size, X.shape, y.shape, y)
        self.X_hold_out = X_[hold_out]
        self.l_hold_out = l[hold_out]
        # print "this the original data for LePU: " + str((y==1).mean())
        #X = np.delete(X, hold_out, axis=0)
        #y = np.delete(y, hold_out)
        #y = y.reshape(-1, 1)
        # print "this is the y labels used for fitting in LePU: " + str((y==1).mean())
        self.estimator.fit(X_[~hold_out], l[~hold_out])
        # print self.estimator.classes_
        hold_out_predictions = self.estimator.predict_proba(self.X_hold_out)
        
        try:
            hold_out_predictions = hold_out_predictions[:,1]
        except:
            pass
        
        c = np.mean(hold_out_predictions)
        self.c = c

        self.estimator_fitted = True
        if self.calibrate:
            from sklearn.calibration import CalibratedClassifierCV
            clf_calibrator = CalibratedClassifierCV(self, cv='refit')
            X_to_scalar = self.predict_proba(X_cal)
            l_to_scalar = (l_cal / 2).astype(int)
            clf_calibrator.fit(self.predict_proba(X_to_scalar), l_to_scalar)

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
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict

        
        # try:
        #     probabilistic_predictions = probabilistic_predictions[:,1]
        # except:
        #     pass
        
#         output = probabilistic_predictions / self.c
        
        output = self.estimator.predict_proba(X_)[:, 1]
        output = output / self.c
        EPSILON = 1e-16
        output [output>1-EPSILON] = 1-EPSILON
        return output
    
    def predict_y_given_x(self, X):
        return self.predict_prob_y_given_x(X)>=0.5 * self.c

    def predict_proba(self, X):
#         if self.preprocessing_type == 'text':
#             X = np.copy(X)
#             X = self.input_transformer.transform(X)
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        X_ = X['sig_input']

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
        
        

