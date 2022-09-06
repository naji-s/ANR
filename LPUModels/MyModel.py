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
from sklearn.metrics.pairwise import rbf_kernel
sys.path.append('/home/scratch/nshajari/psych_model/')
sys.path.append('/home/scratch/nshajari/psych_model/utils')
# from utils.text_utils import tokenize
from scipy import sparse

class MyModel(BaseEstimator):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """
    def __repr__(self):
        return '{} {}'.format(self.__class__.__name__, self.name)
    def __str__(self):
        return '{self.__class__.__name__} {self.name}'.format(self=self)  
    def __call__(self, X):
        return self.predict_prob_y_given_x(X)
    
    def __init__(self, C=1.,  tol=1e-6, encoder=None, penalty=None,
#         sig_vec=None, sig_vec_params=None, 
                 calibrate=False, maxiter=1000, kernel_type=None, name='', gamma=None, model_type='svm', sampled_input_features=None):         
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
        self.gamma = gamma
        self.maxiter = maxiter
#         self.sig_vec = sig_vec
#         self.sig_vec_params = sig_vec_params        
        self.encoder = encoder
        self.penalty = penalty
        self.C = C
        self.kernel_type = kernel_type
        self.calibrate = calibrate
        self.tol = tol
        self.name = name
        self.model_type = model_type
        self.sampled_input_features = sampled_input_features

    def fit(self, X, y):
        """
        X -- List of feature vectors
        y -- Labels associated to each feature vector in X (Positive label: 1.0, Negative label: -1.0)
        """
        print ("C is:", self.C)
        # imidiately decoding the categorical variable l_y_cat to l and y, and implicitly 
        # dropping annotation variable (l) by assigning it to _
        X_ = X['sig_input']
        from sklearn.metrics.pairwise import paired_distances                
        # paired_dist = paired_distances(X_, X_, gamma=self.gamma) / self.C
        # X_[paired_dist>1e50] = 1e50
        # X_[paired_dist<1e-20] = 1e-20
        
        if self.calibrate:
            from sklearn.model_selection import train_test_split
            X_, X_cal, y, y_cal = train_test_split(X_, y, stratify=y)
            # X_cal = {'sig_input':X_cal, 'psych_input':X_cal}
            l_y_decimal_cal = self.encoder.inverse_transform(y_cal)
            _, y_cal = (l_y_decimal_cal/2).astype(int), np.mod(l_y_decimal_cal, 2).reshape((-1, 1))

        l_y_decimal = self.encoder.inverse_transform(y)
        _, y = (l_y_decimal/2).astype(int), np.mod(l_y_decimal, 2).astype(bool)
        self.sampled_input_features = X_
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        
#         if self.preprocessing_type == 'text':
#             X = np.copy(X)
#             tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
#             self.input_transformer = tfidf.fit(X)
#             X = self.input_transformer.transform(X).todense()
#         print ("Naive fitting is running... with x shape", X.shape)

        # the line to take care of extracting s from the passed argument in form of s_y
        # if np.ndim(y) > 1:
        
        #     y = y[:, 0]
        
        if self.kernel_type is None:
            if self.penalty == 'l2':
                self.estimator = LogisticRegression(C=self.C, solver='lbfgs', penalty=self.penalty, tol=self.tol, 
                                                max_iter=self.maxiter, random_state=2019)
            else:
                self.estimator = LogisticRegression(C=self.C, solver='liblinear', penalty=self.penalty, 
                                                    tol=self.tol, max_iter=self.maxiter, random_state=2019)
    #         print ("Naive fitting is ended... with x shape", X.shape)
        else:
            if self.model_type == 'svc':
                from sklearn.svm import SVC
                self.estimator = SVC(C=self.C, kernel=self.kernel_type,tol=self.tol, probability=True, gamma=self.gamma)
            elif self.model_type == 'logistic':
                from sklearn.metrics.pairwise import paired_distances                
                X_ = rbf_kernel(X_, X_, gamma=self.gamma) / self.C
                self.estimator = LogisticRegression(C=1e16, solver='lbfgs', penalty=self.penalty, tol=self.tol, 
                                                max_iter=self.maxiter, random_state=2019)
            else:
                raise NotImplementedError
        self.estimator.fit(X_, y.reshape((1, -1))[0])
        if self.calibrate:
            from sklearn.calibration import CalibratedClassifierCV
            self.clf_calibrator = CalibratedClassifierCV(self.estimator, cv='prefit')
            if self.model_type == 'logistic':
                X_cal = rbf_kernel(X_cal, self.sampled_input_features, gamma=self.gamma) / self.C

            
            # X_to_scalar = self.predict_proba(X_cal)
            # y_to_scalar =  np.mod(y_cal, 2)
            self.clf_calibrator.fit(X_cal, y_cal)
            
            

        return self
    def predict_prob_y_given_x(self, X):
        
        X_ = X['sig_input']
        from sklearn.metrics.pairwise import paired_distances                
        if self.model_type == 'logistic' and self.kernel_type is not None:
            X_ = rbf_kernel(X_, self.sampled_input_features, gamma=self.gamma) / self.C
        
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        if self.calibrate:
            return self.clf_calibrator.predict_proba(X_)[:, 1]
        else:
            return self.estimator.predict_proba(X_)[:, 1]

    def predict_y_given_x(self, X):
        
        X_ = X['sig_input']
        from sklearn.metrics.pairwise import paired_distances                
        # paired_dist = paired_distances(X_, X_, gamma=self.gamma) / self.C
        # X[paired_dist>1e50] = 1e50
        # X[paired_dist<1e-20] = 1e-20
        if self.model_type == 'logistic' and self.kernel_type is not None:
            X_ = rbf_kernel(X_, self.sampled_input_features, gamma=self.gamma) / self.C
        return self.estimator.predict(X_)

    def predict_prob_l_given_y_x(self, X):
        X_ = X['sig_input']
        from sklearn.metrics.pairwise import paired_distances                
        # paired_dist = paired_distances(X, X, gamma=self.gamma) / self.C
        # X[paired_dist>1e50] = 1e50
        # X[paired_dist<1e-20] = 1e-20
        if self.model_type == 'logistic' and self.kernel_type is not None:
            X_ = rbf_kernel(X_, self.sampled_input_features, gamma=self.gamma) / self.C
        
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        return self.estimator.predict_proba(X_)[:, 1]

    def predict_l_given_y_x(self, X):
        return self.predict_prob_y_given_x(X)>=0.5

    
    def predict_proba(self, X):
        
        X_ = X['sig_input']
        from sklearn.metrics.pairwise import paired_distances                
        if self.model_type == 'logistic' and self.kernel_type is not None:
            X_ = rbf_kernel(X_, self.sampled_input_features, gamma=self.gamma) / self.C
        
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        if self.calibrate:
            return self.clf_calibrator.predict_proba(X_)[:, 1]
        else:
            return self.estimator.predict_proba(X_)[:, 1]


    def predict(self, X, treshold=0.5):
        """
        Assign labels to feature vectors based on the estimator's predictions

        X -- List of feature vectors or a precomputed kernel matrix
        treshold -- The decision treshold between the positive and the negative class
        """
        if not self.estimator:
            raise Exception('The estimator must be fitted before calling predict(...).')

        return np.array([1. if p > treshold else 0. for p in self.predict_proba(X)])



