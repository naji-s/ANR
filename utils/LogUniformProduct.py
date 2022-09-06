from numpy import asarray, float64, logspace, log
from numpy.random import choice
from scipy.stats import loguniform
class LogUniformProduct():
    "Product of LogUniform distributions with rejection sampling"
    def __init__(self, name='test', min_list=[1e-4, 1e-4], max_list=[1e4, 1e4], min_prod=1e-12, max_prod = 1e8):
#         super(LogUniformProduct, self).__init__(name=name, **kwargs)
        self.base_dists = []
        self.max_prod = max_prod
        self.min_prod = min_prod
        for min_elem, max_elem in zip(min_list, max_list):
            self.base_dists.append(loguniform(min_elem, max_elem))
        
    def rvs(self, size=1):
        sample = []
        samples = []
        product = 1.
        try:
            while (product > self.max_prod or product < self.min_prod or len(samples) < size):
                product = 1.
                for dist in self.base_dists:
                    one_dim_sample = dist.rvs()
                    sample.append(one_dim_sample)
                    product *= one_dim_sample
                if product < self.max_prod and product > self.min_prod:
                    samples.append(sample)
                sample = []                   
        except Exception as e:
            raise type(e)(str(e) + str(product)+','+str(self.max_prod)+','+str(self.min_prod)+','+str(size)+','+str(len(samples)))
        return asarray(samples).reshape((size, -1))
    
