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
# from utils.text_utils import tokenize

class NaiveLPU(BaseEstimator):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """

    def __init__(self, C=None, penalty='None', tol=None, preprocessing_type=None, encoder=None,
#         sig_vec=None, sig_vec_params=None, 
                 calibrate=False, maxiter=1000, kernel_type=None, gamma=None):         
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
#         if None not in [sig_vec, sig_vec_params]:
#             self.multiple_representations = True
#         else:
#             self.multiple_representations = False
#         self.sig_vec = sig_vec
#         self.sig_vec_params = sig_vec_params
#         if self.multiple_representations:
#             temp_output = X_train[self.sig_vec][self.sig_vec_params]
#             del(X_train)
#             X_train = temp_output
        
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
        self.tol = tol
        self.preprocessing_type = preprocessing_type
        self.input_transformer = None
        self.calibrate = calibrate
        self.maxiter = maxiter
        self.kernel_type = kernel_type


    def fit(self, X, y, sar_em_model=None):
        """
        X -- List of feature vectors
        y -- Labels associated to each feature vector in X (Positive label: 1.0, Negative label: -1.0)
        """
        self.sar_em_model = sar_em_model
        # imidiately decoding the categorical variable l_y_cat to l and y, replacing l with y and implicitly 
        # dropping real class (y) by assigning it to _
            
            
        l_y_decimal = self.encoder.inverse_transform(y)
        y, _ = (l_y_decimal/2).astype(int), np.mod(l_y_decimal, 2).reshape((-1, 1))
        X_ = X['sig_input']
        if self.calibrate:
            from sklearn.model_selection import train_test_split
            X_, X_cal, y, y_cal = train_test_split(X_, y, stratify=y)
            l_y_decimal_cal = self.encoder.inverse_transform(y_cal)
            y_cal, _ = (l_y_decimal_cal/2).astype(int), np.mod(l_y_decimal_cal, 2).reshape((-1, 1))

#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict

        if self.kernel_type is None:
            if self.penalty == 'l2':
                self.estimator = LogisticRegression(C=self.C, solver='lbfgs', penalty=self.penalty, tol=self.tol, 
                                                max_iter=self.maxiter, random_state=2019)
            else:
                self.estimator = LogisticRegression(C=self.C, solver='liblinear', penalty=self.penalty, tol=self.tol, max_iter=self.maxiter, random_state=2019)
        else:
            from sklearn.svm import SVC
            self.estimator = SVC(C=self.C, kernel=self.kernel_type, tol=self.tol, probability=True, gamma=self.gamma)

        if self.sar_em_model is None:
            self.estimator.fit(X_, y)
        else:
            sample_weight=self.sar_em_model.prop.predict_proba(X)
            print ("SAMPLE WEIGHTS ARE ACTUALY:", sample_weight)
            self.estimator.fit(X_, y, sample_weight=sample_weight)
            
            
        if self.calibrate:
            from sklearn.calibration import CalibratedClassifierCV
            clf_calibrator = CalibratedClassifierCV(self, cv='refit')
            X_to_scalar = self.predict_proba(X_cal)
            y_to_scalar = (y_cal / 2).astype(int)
            clf_calibrator.fit(self.predict_proba(X_to_scalar), y_to_scalar)

        return self
    def predict_prob_y_given_x(self, X):
        X_ = X['sig_input']
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        return self.estimator.predict_proba(X_)[:, 1]

    def predict_y_given_x(self, X):
        X_ = X['sig_input']
        return self.estimator.predict(X_)

    def predict_proba(self, X):
        X_ = X['sig_input']
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        return self.estimator.predict_proba(X_)[:, 1]
    def predict_prob_l_given_y_x(self, X):
        X_ = X['sig_input']
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        return np.ones(shape=X_.shape[0])
        
    def predict(self, X, treshold=0.5):
        """
        Assign labels to feature vectors based on the estimator's predictions

        X -- List of feature vectors or a precomputed kernel matrix
        treshold -- The decision treshold between the positive and the negative class
        """
        # if not self.estimator_fitted:
        #     raise Exception('The estimator must be fitted before calling predict(...).')

        return np.array([1. if p > treshold else 0. for p in self.predict_proba(X)])



