import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

class SNLike(Likelihood):
    
    def initialize(self):

        self.dataset_SN = pd.read_csv(self.SN_data_path+'_data.txt',sep='\s+',header=0)
        covmat          = pd.read_csv(self.SN_data_path+'_covmat.txt',sep='\s+',header=0)
        self.invcovmat  = np.linalg.inv(covmat)


    def get_requirements(self):
        # Requirements are the output of the theory code that you are using
        requirements = {'mB': None}

        return requirements
    
    def logp(self, **params_values): 

        diffvec_SN = (self.provider.get_result('mB')(self.dataset_SN['z'].values))-(self.dataset_SN['mB'+self.suffix].values)

        loglike = -0.5*np.dot((diffvec_SN),np.dot(self.invcovmat,(diffvec_SN)))

        return loglike
