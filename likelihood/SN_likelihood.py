import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

class SNLike(Likelihood):
    
    def initialize(self):

        self.dataset_SN = np.load(self.SN_data_path+'.npy',allow_pickle=True).item()


    def get_requirements(self):
        # Requirements are the output of the theory code that you are using
        requirements = {'mB': None}

        return requirements
    
    def logp(self, **params_values): 

        diffvec_SN = (self.provider.get_result('mB')(self.dataset_SN['z']))-(self.dataset_SN['mB'])

        loglike = -0.5*np.dot((diffvec_SN),np.dot(np.linalg.inv(((self.dataset_SN['covmat']))),(diffvec_SN)))

        return loglike
