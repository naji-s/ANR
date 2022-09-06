#Custom Transformer that extracts columns passed as argument to its constructor 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import tensorflow as tf
from func_lib import convert_sparse_matrix_to_sparse_tensor
from sys import path
path.append('/home/scratch/nshajari/psych_model/utils')
import IdentityTransformer as IdentityTransformer

class MultipleVectorizer( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__(self, representation_methods_dict={'sig_input': IdentityTransformer.IdentityTransformer, 'psych_input': IdentityTransformer.IdentityTransformer}, representation_params_dict={'sig_input':{}, 'psych_input':{}}, transformer_dict=dict()):
        #Return seIdentityTransformerlf nothing else to do here    
        self.representation_methods_dict = representation_methods_dict
        self.representation_params_dict = representation_params_dict
        self.transformer_dict=transformer_dict
    def fit( self, X, y = None):
        self.transformer_dict = dict()
        for partial_input in ['sig_input', 'psych_input']:
#         for item in self.representation_params:
#             self.representation_params_tuples.add(tuple(sorted((k, v) for k, v in item.items())))
#         for key in self.representation_methods:
#             self.fit_transformers_dict[key] = dict()
#             for item in self.representation_params_tuples:
#                 self.fit_transformers_dict[key][item] = key(**dict(item)).fit(X)
            try:
                self.transformer_dict[partial_input] = self.representation_methods_dict[partial_input] (**self.representation_params_dict[partial_input]).fit([item[0] if type(item[0]).__name__ == 'str' else item for item in X[partial_input]])
            except Exception as e:
                raise type(e)(str(e) + " representation_methods_dict: " + str(self.representation_methods_dict))
    
        return self
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None, matrix_output=False):
#         tensor_output = dict()
#         matrix_output = dict()
#         for key in self.representation_methods:
#             tensor_output[key] = dict()
#             matrix_output[key] = dict()
#             for item in self.representation_params_tuples:
#                 print(key, item)
#                 sparse_matrix = self.fit_transformers_dict[key][item].transform(X).astype(np.float64)
                    
#                 tensor_output[key][item] = convert_sparse_matrix_to_sparse_tensor(sparse_matrix)
#                 matrix_output[key][item] = sparse_matrix
        output_dict = dict()
        try:
            for partial_input in ['sig_input', 'psych_input']:
                output_dict[partial_input] = self.transformer_dict[partial_input].transform([item[0] if type(item[0]).__name__ == 'str' else item for item in X[partial_input]])
                if type(output_dict[partial_input][0][0]).__name__ == 'csr_matrix':
                    output_dict[partial_input] = output_dict[partial_input]
                else:
                    output_dict[partial_input] = np.squeeze(np.asarray(output_dict[partial_input]))
        except Exception as e:
            raise type(e)(str(e) + " in transform() we have representation_methods_dict: " + str(self.representation_methods_dict))
                    
        return output_dict
