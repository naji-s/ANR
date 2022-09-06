    #Custom Transformer that extracts columns passed as argument to its constructor 
from sklearn.base import BaseEstimator, TransformerMixin
class TransformToCOO( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__(self, tocsr=False):
        self.tocsr = tocsr
        pass
#         self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        if self.tocsr:
            if type(X).__name__ == 'dict':
                output_dict = dict()
                for key in X.keys():
                    try:
                        output_dict[key] = X[key].tocsr()
                    except AttributeError:
                        output_dict[key] = X[key]
                return output_dict
            try:
                return X.tocoo().tocsr()
            except AttributeError:
                return X.tocoo()
        else:
            if type(X).__name__ == 'dict':
                output_dict = dict()
                for key_1 in X.keys():
                    output_dict[key_1] = X[key_1]
                return output_dict
            return X.tocoo() 