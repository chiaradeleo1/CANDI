import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

class GWLike(Likelihood):
    
    def initialize(self):

        self.dataset_GW = np.load(self.GW_data_path+'.npy',allow_pickle=True).item()
        self.invcov = np.linalg.inv(((self.dataset_GW['covmat'])))


    def get_requirements(self):
        # Requirements are the output of the theory code that you are using
        requirements = {'DL_GW': None}

        return requirements
    
    def logp(self, **params_values): 
        
        diffvec_GW = (self.provider.get_result('DL_GW')(self.dataset_GW['z']))-(self.dataset_GW['dL'])
        

        loglike = -0.5*np.dot((diffvec_GW),np.dot(self.invcov,(diffvec_GW)))

        return loglike
