from numpy import mod, ones, array
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append('/home/nshajari/master_thesis/')
sys.path.append('/Users/naji/Box Sync/Box Sync/CMU/Masters Thesis/master_thesis/')
# from utils.text_utils import tokenize
from miscellaneous.Kernel_MPE_grad_threshold import wrapper

EPSILON = 1e-16

class KMEModel(BaseEstimator):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """

    def __init__(self, C=1.,  tol=1e-6, encoder=None, penalty='l2',
#                 sig_vec=None, sig_vec_params=None, 
                 kernel_mode=1, calibrate=False, kernel_type=None, gamma=None):
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
#         self.sig_vec = sig_vec
#         self.sig_vec_params = sig_vec_params
        super().__init__()
        self.encoder = encoder
        self.penalty = penalty
        self.gamma = gamma
        self.C = C
        self.tol = tol
        self.kernel_mode = kernel_mode
        self.calibrate = calibrate
        self.kernel_type = kernel_type
        

    def fit(self, X, y):
        """
        X -- List of feature vectors
        y -- Labels associated to each feature vector in X (Positive label: 1.0, Negative label: -1.0)
        """
        # imidiately decoding the categorical variable l_y_cat to l and y, and implicitly 
        # dropping annotation variable (l) by assigning it to _
        X_ = X['sig_input']
        if self.calibrate:
            from sklearn.model_selection import train_test_split
            X_, X_cal, y, y_cal = train_test_split(X_, y, stratify=y)
            l_y_decimal_cal = self.encoder.inverse_transform(y_cal)
            y_cal, _ = (l_y_decimal_cal/2).astype(int),  mod(l_y_decimal_cal, 2).reshape((-1, 1))
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self.encoder)
        l_y_decimal = self.encoder.inverse_transform(y)
        y, _ = (l_y_decimal/2).astype(int),  mod(l_y_decimal, 2)
        from sklearn.model_selection import train_test_split
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
            
        _, X_holdout, _, y_holdout = train_test_split(X_, y, test_size=int(min(max(1000, len(y) * 0.2), len(y)-1)))
#         X_holdout = X
#         y_holdout = y
        if self.kernel_type is None:
            if self.penalty == 'l2':
                self.estimator = LogisticRegression(C=self.C, solver='lbfgs', penalty=self.penalty, tol=self.tol, 
                                                max_iter=50, random_state=2019)
            else:
                self.estimator = LogisticRegression(C=self.C, solver='liblinear', penalty=self.penalty, 
                                                    tol=self.tol, max_iter=5000, random_state=2019)
        else:
            from sklearn.svm import SVC
            self.estimator = SVC(kernel=self.kernel_type, C=self.C, probability=True, gamma=self.gamma)

            
        if type(X_).__name__ == 'coo_matrix':
            X_ = X_.tocsr()
#         y_holdout = y_holdout.reshape((-1, 1))
        if type(X_).__name__ == 'csr_matrix':
            kappa_2, kappa_1 = wrapper(X_holdout.toarray(),X_holdout[y_holdout==1].toarray())
        else:
            kappa_2, kappa_1 = wrapper(X_holdout, X_holdout[y_holdout==1])
        if self.kernel_mode == 1:
            self.c = y_holdout.mean() / kappa_1
        else:
            self.c = y_holdout.mean() / kappa_2
        print ("c value estimate for KME is:", self.c, "with other estimate being:", y.mean() / kappa_1)
        self.estimator.fit(X_, y)
        if self.calibrate:
            from sklearn.calibration import CalibratedClassifierCV
            clf_calibrator = CalibratedClassifierCV(self, cv='refit')
            X_to_scalar = self.predict_proba(X_cal)
            y_to_scalar = (y_cal / 2).astype(int)
            clf_calibrator.fit(self.predict_proba(X_to_scalar), y_to_scalar)

        return self
    
        

    def predict_prob_y_given_x(self, X):
        X_ = X['sig_input']
#         if self.multiple_representations:
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        output = self.estimator.predict_proba(X_)[:, 1]
        # output[output < 1e-30] = 1e-30
        # output[output > 1. - EPSILON] = 1 -  EPSILON
        output = output / self.c
        output[output>1-EPSILON] = 1-EPSILON
        return output
    def predict_prob_l_given_y_x(self, X):
        X_ = X['sig_input']
#         if None not in [self.sig_vec, self.sig_vec_params]:
#             self.multiple_representations = True
#             temp_dict = dict()
#             temp_dict = X[self.sig_vec][tuple(sorted((k, v) for k, v in self.sig_vec_params.items()))]
#             del(X)
#             X = temp_dict
        output = self.c *  ones(X_.shape[0])
        output[output>1-EPSILON] = 1-EPSILON

        return output

    def predict_y_given_x(self, X):
        return self.predict_prob_y_given_x(X)>=0.5 * self.c


    def predict_l_given_x(self, X):
        output = self.predict_prob_l_given_y_x(X)>=0.5
        return output

    
    def predict_proba(self, X):
        output = self.c * self.predict_prob_y_given_x(X)
        output[output>1-EPSILON] = 1-EPSILON
        return output

    def predict(self, X, treshold=0.5):
        """
        Assign labels to feature vectors based on the estimator's predictions

        X -- List of feature vectors or a precomputed kernel matrix
        treshold -- The decision treshold between the positive and the negative class
        """

        return  array([1. if p > treshold else 0. for p in self.predict_proba(X)])



