#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Dec 21, 2012

@author: Alexandre
"""
import numpy as np


class PUAdapter(object):
    """
    Adapts any probabilistic binary classifier to positive-unlabled learning using the PosOnly method proposed by
    Elkan and Noto:

    Elkan, Charles, and Keith Noto. \"Learning classifiers from only positive and unlabeled data.\"
    Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.
    """

    def __init__(self, estimator, hold_out_ratio=0.1, precomputed_kernel=False):
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
        self.estimator = estimator
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio

        self.fit = self.__fit

        self.estimator_fitted = False
        self.encoder = None
    def __str__(self):
        return 'Estimator:' + str(self.estimator) + '\n' + 'p(s=1|y=1,x) ~= ' + str(self.c) + '\n' + \
               'Fitted: ' + str(self.estimator_fitted)


    def __fit(self, X, y):
        """
        Fits an estimator of p(s=1|x) and estimates the value of p(s=1|y=1,x)

        X -- List of feature vectors
        y -- Labels associated to each feature vector in X (Positive label: 1.0, Negative label: -1.0)
        """
        positives = np.where(y == 1.)[0]
        hold_out_size = int(np.ceil(len(positives) * self.hold_out_ratio))
        print hold_out_size
        if len(positives) <= hold_out_size:
            raise (
            'Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')

        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]
        X_hold_out = X[hold_out]
        X = np.delete(X, hold_out, 0)
        y = np.delete(y, hold_out)

        self.estimator.fit(X, y)

        hold_out_predictions = self.estimator.predict_proba(X_hold_out)

        try:
            hold_out_predictions = hold_out_predictions[:, 1]
        except:
            pass

        c = np.mean(hold_out_predictions)
        self.c = c

        self.estimator_fitted = True

    def predict_proba(self, X):
        """
        Predicts p(y=1|x) using the estimator and the value of p(s=1|y=1) estimated in fit(...)

        X -- List of feature vectors or a precomputed kernel matrix
        """
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict_proba(...).')

        probabilistic_predictions = self.estimator.predict_proba(X)

        try:
            probabilistic_predictions = probabilistic_predictions[:, 1]
        except:
            pass

        return probabilistic_predictions / self.c

    def predict(self, X, treshold=0.5):
        """
        Assign labels to feature vectors based on the estimator's predictions

        X -- List of feature vectors or a precomputed kernel matrix
        treshold -- The decision treshold between the positive and the negative class
        """
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict(...).')

        return np.array([1. if p > treshold else -1. for p in self.predict_proba(X)])



