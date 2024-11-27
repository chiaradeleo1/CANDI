import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

class GWLike(Likelihood):
    
    def initialize(self):

        self.dataset_GW = pd.read_csv(self.GW_data_path+'_data.txt',sep='\s+',header=0)
        covmat          = pd.read_csv(self.GW_data_path+'_covmat.txt',sep='\s+',header=0)
        self.invcovmat  = np.linalg.inv(covmat)


    def get_requirements(self):
        # Requirements are the output of the theory code that you are using
        requirements = {'DL_GW': None}

        return requirements
    
    def logp(self, **params_values): 
        
        diffvec_GW = (self.provider.get_result('DL_GW')(self.dataset_GW['z'].values))-(self.dataset_GW['dL'].values)
        

        loglike = -0.5*np.dot((diffvec_GW),np.dot(self.invcovmat,(diffvec_GW)))

        return loglike
