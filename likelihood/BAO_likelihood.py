import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

class BAOLike(Likelihood):
    
    def initialize(self):
        
        self.dataset_DHDM = pd.read_csv(self.BAO_data_path+'_data_DHDM.txt',sep='\s+',header=0)
        covmat_DHDM       = pd.read_csv(self.BAO_data_path+'_covmat_DHDM.txt',sep='\s+',header=0)
        self.invcov_DHDM  = np.linalg.inv(covmat_DHDM)

        self.dataset_DV = pd.read_csv(self.BAO_data_path+'_data_DV.txt',sep='\s+',header=0)
        covmat_DV       = pd.read_csv(self.BAO_data_path+'_covmat_DV.txt',sep='\s+',header=0)
        self.invcov_DV  = np.linalg.inv(covmat_DV)

        if self.use_noisy_data:
            self.suffix = '_noisy'
        else:
            self.suffix = ''


        zmax = 4.
        self.z_camb = np.linspace(0.001, zmax, 10000)


    def get_requirements(self):
        # Requirements are the output of the theory code that you are using
        requirements = {'DM': None,
                        'DH': None,
                        'DV': None }

        return requirements
    
    def logp(self, **params_values): 

        diffvec_DH   = self.provider.get_result('DH')(self.dataset_DHDM['z'])-self.dataset_DHDM['DH'+self.suffix]
        diffvec_DM   = self.provider.get_result('DM')(self.dataset_DHDM['z'])-self.dataset_DHDM['DM'+self.suffix]
        diffvec_DV   = self.provider.get_result('DV')(self.dataset_DV['z'])-self.dataset_DV['DV'+self.suffix]
        diffvec_DHDM = np.concatenate((diffvec_DH, diffvec_DM),axis=0)

        chi2  = np.dot((diffvec_DHDM),np.dot(self.invcov_DHDM,(diffvec_DHDM)))
        chi2 += np.dot((diffvec_DV),np.dot(self.invcov_DV,(diffvec_DV)))

        loglike = -0.5*chi2

        return loglike
