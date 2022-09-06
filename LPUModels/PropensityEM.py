#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Dec 21, 2012

@author: Alexandre
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append('/home/nshajari/master_thesis/')
sys.path.append('/Users/naji/Box Sync/Box Sync/CMU/Masters Thesis/master_thesis/')
# from utils.text_utils import tokenize
from scipy import sparse
sys.path.insert(0,'/home/scratch/nshajari/psych_model/SAR_PU/sarpu')
sys.path.insert(0,'/home/scratch/nshajari/psych_model/SAR_PU/sarpu/sarpu')
from pu_learning import pu_learn_sar_em
from scipy.sparse import csr_matrix, coo_matrix

class PropensityEM(BaseEstimator):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """

    def __init__(self, C_c=1., C_p=1.,  gamma=None, encoder=None, c_penalty='l2', p_penalty='l2', 
#         sig_vec=None, sig_vec_params=None, 
                 tol=1e-6, cv=False, clf=None, maxiter=1000,
                prop=None, expected_posterior_y1=None, info=None, calibrate=False, propensity_attributes=None, kernel_type=None):         
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
        super().__init__()
        self.cv = cv
        self.maxiter = maxiter
        self.calibrate = calibrate
#         self.sig_vec = sig_vec
#         self.sig_vec_params = sig_vec_params
        self.c_penalty = c_penalty
        self.p_penalty = p_penalty
        self.encoder = encoder
        self.C_c = C_c
        self.C_p = C_p
        self.tol = tol
        self.clf = clf
        self.gamma = gamma
        self.kernel_type = kernel_type
        self.prop = prop
        self.propensity_attributes = propensity_attributes
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {
                "c_penalty": self.c_penalty,
                "p_penalty": self.p_penalty,
                "encoder": self.encoder,
                'kernel_type':self.kernel_type, 
                "C_p": self.C_p,
                "gamma": self.gamma,
                "C_c": self.C_c,
                "tol": self.tol,
                "cv": self.cv,
                "clf": self.clf,
                "prop": self.prop,
                "propensity_attributes": self.propensity_attributes,
                "maxiter": self.maxiter
               }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def fit(self, X, y, cv=False, encoder=None):
        """
        X -- List of feature vectors
        y -- Labels associated to each feature vector in X (Positive label: 1.0, Negative label: -1.0)
        """
            
        self.input_transformer = None
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#         else:
#             self.multiple_representations = False

        # imidiately decoding the categorical variable l_y_cat to l and y, and implicitly 
        # dropping annotation variable (l) by assigning it to _
        if not self.cv:
            l_y_cat_transformed = y
            l_y_decimal = self.encoder.inverse_transform(y)
            y, _ = (l_y_decimal/2).astype(int), np.mod(l_y_decimal, 2)
#             if None not in [self.sig_vec, self.sig_vec_params]:
#                 self.multiple_representations = True
#                 temp_dict = dict()
#                 temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#                 del(X)
            X_ = X['sig_input']
            if type(X).__name__ == 'coo_matrix':
                X_ = X.tocsr()
                
#             if self.propen_prop != []:
#                 self.propensity_attributes = np.arange(X.shape[-1])
#                 self.C_p = self.propen_prop[0]
#                 self.penalty = self.propen_prop[1]
                
#             else:
#                 self.C_p = 1.
#                 self.propensity_attributes = []
#                 self.p_penalty = 'l2'
                
                
        if cv:
            print ("SUPPOSED TO RUN ONCE!!!")
            clf = PropensityEM()
            from sklearn.model_selection import GridSearchCV
            from sklearn.model_selection import StratifiedKFold
            from sklearn.pipeline import Pipeline
            
            strat_kfold = StratifiedKFold(10, shuffle=True, random_state=2020)
            param_grid = dict()
            from sklearn.preprocessing import OneHotEncoder
#             from sklearn.preprocessing import LabelEncoder
#             encoder = OneHotEncoder()
#             encoder.fit(l_y_decimal.reshape((-1, 1)))
#             param_grid['clf__encoder'] = [encoder]
            param_grid['C_c'] = np.logspace(-4, 4, 5)
            param_grid['cv'] = [True]
            param_grid['encoder'] = [self.encoder]
            from sklearn.metrics import log_loss, make_scorer
            from numpy import hstack
#             def my_log_loss(y_true, y_score):
#                 y_temp = y_score.reshape(-1, 1)
#                 y_score = hstack(1 - y_temp, y_temp)
#                 return log_loss(y_true, y_score)
            LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)            
            model_cv = GridSearchCV(
                    clf,
                    scoring=LogLoss,
                    cv=strat_kfold,
                    param_grid=param_grid,
                    n_jobs=1,
                )
            model_cv.fit(X_, y)
            self = model_cv.best_estimator_
            self.clf, self.prop, self.info, self.expected_posterior_y1 = model_cv.best_estimator_.clf, model_cv.best_estimator_.prop, model_cv.best_estimator_.info, model_cv.best_estimator_.expected_posterior_y1
            self.set_params(parameters={'clf':self.clf, 'prop':self.prop})
#             self.clf, self.prop, self.info, self.expected_posterior_y1 = best_clf.clf, best_clf.prop, best_clf.expected_posterior_y1
            return model_cv
        else:
            if self.kernel_type is None:
                self.clf, self.prop, self.info, self.expected_posterior_y1 = pu_learn_sar_em(X_, y, C_p=self.C_p, C_c=self.C_c,  c_penalty=self.c_penalty,p_penalty=self.p_penalty, maxiter=self.maxiter, tol=self.tol, propensity_attributes=self.propensity_attributes, max_its=500)
                self.set_params(parameters={'clf':self.clf, 'prop':self.prop})
            else:
                from sklearn.svm import SVC
                svc_model = SVC(kernel=self.kernel_type, C=self.C_c, gamma=self.gamma, probability=True)
                self.clf, self.prop, self.info, self.expected_posterior_y1 = pu_learn_sar_em(X_, y, C_p=self.C_p, C_c=self.C_c,  c_penalty=self.c_penalty,p_penalty=self.p_penalty, maxiter=self.maxiter, tol=self.tol, propensity_attributes=self.propensity_attributes, classification_model=svc_model, max_its=500)
#         print ("fitting done!!")
#         self.em_estimator.fit()
        
#         if self.preprocessing_type == 'text':
#             X = np.copy(X)
#             tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
#             self.input_transformer = tfidf.fit(X)
#             X = self.input_transformer.transform(X).todense()
#         print ("Naive fitting is running... with x shape", X.shape)

        # the line to take care of extracting s from the passed argument in form of s_y
        # if np.ndim(y) > 1:
        #     y = y[:, 0]

#         if self.penalty == 'l2':
#             self.estimator = LogisticRegression(C=self.C, solver='lbfgs', penalty=self.penalty, tol=self.tol, 
#                                             max_iter=1000, random_state=2019)
#         else:
#             self.estimator = LogisticRegression(C=self.C, solver='liblinear', penalty=self.penalty, 
#                                                 tol=self.tol, max_iter=1000, random_state=2019)
#         self.estimator.fit(X, y)
#         print ("Naive fitting is ended... with x shape", X.shape)
        if self.calibrate:
            from sklearn.calibration import CalibratedClassifierCV
            clf_calibrator = CalibratedClassifierCV(self, cv='refit')
            X_to_scalar = self.predict_proba(X_cal)
            y_to_scalar = (y_cal / 2).astype(int)
            clf_calibrator.fit(self.predict_proba(X_to_scalar), y_to_scalar)

        return self


    def predict_prob_y_given_x(self, X):
        X_ = X['sig_input']
        if type(X_).__name__ == 'coo_matrix':
            X_ = X_.tocsr()
#         if self.multiple_representations:
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del (X)
#             X = temp_dict
        output = self.clf.predict_proba(X_)
        # EPSILON = 1e-16
        # output[output > 1-EPSILON] = 1 - EPSILON
        return output

    def predict_y_given_x(self, X):
        return self.clf.predict(X)

    def predict_prob_l_given_y_x(self, X):
        X_ = X['sig_input']        
        if type(X_).__name__ == 'coo_matrix':
            X_ = X_.tocsr()
#         if self.multiple_representations:
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del (X)
#             X = temp_dict
        output = self.prop.predict_proba(X_)
        # EPSILON = 1e-16
        # output[output > 1-EPSILON] = 1 - EPSILON
        
        return output

    def predict_l_given_x(self, X):
        return self.prop.predict(X)>=0.5

    
    def predict_proba(self, X):
        from numpy import hstack
        if not self.cv:
            return np.multiply(self.predict_prob_y_given_x(X), self.predict_prob_l_given_y_x(X))
        else:
            scores = np.multiply(self.predict_prob_y_given_x(X), self.predict_prob_l_given_y_x(X)).reshape((-1, 1))
            return hstack((1-scores, scores))

    def predict(self, X, treshold=0.5):
        """
        Assign labels to feature vectors based on the estimator's predictions

        X -- List of feature vectors or a precomputed kernel matrix
        treshold -- The decision treshold between the positive and the negative class
        """

        return np.array([1. if p > treshold else 0. for p in self.predict_proba(X)])



