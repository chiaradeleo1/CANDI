import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

class BAOLike(Likelihood):
    
    def initialize(self):
        
        self.dataset_DH = np.load(self.BAO_data_path+'_DH.npy',allow_pickle=True).item()
        self.dataset_DM = np.load(self.BAO_data_path+'_DM.npy',allow_pickle=True).item()
        self.dataset_DV = np.load(self.BAO_data_path+'_DV.npy',allow_pickle=True).item()
        self.dataset_DHDM = np.load(self.BAO_data_path+'_DHDM.npy',allow_pickle=True).item()

        zmax = 4.
        self.z_camb = np.linspace(0.001, zmax, 10000)


    def get_requirements(self):
        # Requirements are the output of the theory code that you are using
        requirements = {'DM': None,
                        'DH': None,
                        'DV': None }

        return requirements
    
    def logp(self, **params_values): 

        diffvec_DH = self.provider.get_result('DH')(self.dataset_DH['z'])-self.dataset_DH['value']
        diffvec_DM = self.provider.get_result('DM')(self.dataset_DM['z'])-self.dataset_DM['value']
        diffvec_DV = self.provider.get_result('DV')(self.dataset_DV['z'])-self.dataset_DV['value']
        diffvec_DHDM=np.concatenate((diffvec_DH, diffvec_DM),axis=0)

        chi_DHDM=np.dot((diffvec_DHDM),np.dot(np.linalg.inv((self.dataset_DHDM['covmat'])),(diffvec_DHDM)));
        chi_DV=np.dot((diffvec_DV),np.dot(np.linalg.inv(((self.dataset_DV['covmat']))),(diffvec_DV)))

        loglike=-0.5*chi_DV-0.5*chi_DHDM

        return loglike
