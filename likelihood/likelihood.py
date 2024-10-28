import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

class DDRLike(Likelihood):
    
    def initialize(self):
        
        self.BAO_data = pd.read_csv(self.BAO_data_path+'_dataset.txt',sep='\s+',header=0)
        covmat       = pd.read_csv(self.BAO_data_path+'_covmat.txt',sep='\s+',header=0) 
        self.invcov_BAO  = np.linalg.inv(covmat)

        self.BAO_DV_data = pd.read_csv(self.BAO_data_path+'_dataset_DV.txt',sep='\s+',header=0)
        covmat_DV       = pd.read_csv(self.BAO_data_path+'_covmat_DV.txt',sep='\s+',header=0)
        self.invcov_BAO_DV  = np.linalg.inv(covmat_DV)

        self.SN_data = pd.read_csv(self.SN_data_path, sep='\t', header=0) #mock_data that are generated with epsilon=0

        zmax = max(self.BAO_data['z'].max(), self.SN_data['z'].max())
        self.z_camb = np.linspace(0.001, zmax, 10000)


    def get_requirements(self):
        # Requirements are the output of the theory code that you are using
        return {'BAOtheory': { 'DM' : None,
                                'DH' : None,
                                'DV' : None },
                'SN_theory':  {'mB_z' : None}}
    
    def logp(self, **params_values): 
        
        BAO_theory = self.theory['BAO_theory'].get_theory()
        SN_theory = self.theory['SN_theory'].get_theory()

        BAO_theory_vec = np.concatenate([BAO_theory['DM'](self.BAO_data['z']),BAO_theory['DH'](self.BAO_data['z']) ])
        BAO_DV_theory_vec = BAO_theory['DV'](self.BAO_DV_data['z'])
        SN_theory_vec = SN_theory['mu_z'](self.SN_data['z'])

        BAO_data_vec = np.concatenate([self.BAO_data['DM'],self.BAO_data['DH'] ])
        BAO_DV_data_vec = self.BAO_data['DV']
        SN_data_vec = self.SN_data['mu']

        BAO_diff_vec = BAO_data_vec-BAO_theory_vec
        BAO_DV_diff_vec = BAO_DV_data_vec-BAO_DV_theory_vec
        SN_diff_vec  = SN_theory_vec-SN_data_vec

        chi2 = np.dot(BAO_diff_vec,np.dot(self.invcov_BAO,BAO_diff_vec))
        chi2 += np.dot(BAO_DV_diff_vec,np.dot(self.invcov_BAO_DV,BAO_DV_diff_vec))

        chi2 += (SN_diff_vec/self.SN_data['err'])**2

        loglike = -0.5*chi2

        return loglike