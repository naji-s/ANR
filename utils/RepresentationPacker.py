# Custom Transformer to break the concatenated input for sig and psych into parts 
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import normalize
class RepresentationPacker( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__(self, sig_input_dim=None, psych_input_dim=None, normalize_sig_input=False, normalize_psych_input=False):
        #Return self nothing else to do here    
        self.sig_input_dim = sig_input_dim
        self.psych_input_dim = psych_input_dim
        self.normalize_psych_input = normalize_psych_input
        self.normalize_sig_input = normalize_sig_input
        self.sig_input_normalizer = StandardScaler()
        self.psych_input_normalizer = StandardScaler()
        
    def fit( self, X, y=None):
        print ("RepresentationPacker fitting is running...")
        try:
            assert self.sig_input_dim + self.psych_input_dim == X.shape[-1]
        except:
            print ("The breaking of X is not working. X shape is", X.shape[-1], 
                   "while sig_input_dim is", self.sig_input_dim, "while psych_input_dim is", self.psych_input_dim)
            raise AssertionError
        if self.normalize_psych_input:
            self.psych_input_normalizer.fit(X[:, :self.psych_input_dim])

        if self.normalize_sig_input:
            self.sig_input_normalizer.fit(X[:, self.psych_input_dim:])
        print ("RepresentationPacker fitting ENDED...")
        return self
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None, matrix_output=False):
        if self.normalize_psych_input:
            X_1 = self.psych_input_normalizer.transform(X[:, :self.psych_input_dim])
        else:
            X_1 = X[:, :self.psych_input_dim]
        if self.normalize_sig_input:
            X_2 = self.sig_input_normalizer.transform(X[:, self.psych_input_dim:])
        else:
            X_2 = X[:, self.psych_input_dim:]
        
        output_dict = {'sig_input': X_2, 'psych_input':X_1}
        return output_dict
