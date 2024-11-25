import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

class BAOLike(Likelihood):
    
    def initialize(self):
        
        self.dataset_DHDM = np.load(self.BAO_data_path+'_DHDM.npy',allow_pickle=True).item()
        self.dataset_DV = np.load(self.BAO_data_path+'_DV.npy',allow_pickle=True).item()

        zmax = 4.
        self.z_camb = np.linspace(0.001, zmax, 10000)


    def get_requirements(self):
        # Requirements are the output of the theory code that you are using
        requirements = {'DM': None,
                        'DH': None,
                        'DV': None }

        return requirements
    
    def logp(self, **params_values): 

        diffvec_DH   = self.provider.get_result('DH')(self.dataset_DHDM['z'])-self.dataset_DHDM['DH']
        diffvec_DM   = self.provider.get_result('DM')(self.dataset_DHDM['z'])-self.dataset_DHDM['DM']
        diffvec_DV   = self.provider.get_result('DV')(self.dataset_DV['z'])-self.dataset_DV['DV']
        diffvec_DHDM = np.concatenate((diffvec_DH, diffvec_DM),axis=0)

        chi2  = np.dot((diffvec_DHDM),np.dot(np.linalg.inv((self.dataset_DHDM['covmat'])),(diffvec_DHDM)))
        chi2 += np.dot((diffvec_DV),np.dot(np.linalg.inv(((self.dataset_DV['covmat']))),(diffvec_DV)))

        loglike = -0.5*chi2

        return loglike
