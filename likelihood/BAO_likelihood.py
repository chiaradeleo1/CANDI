import sys
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

class BAOLike(Likelihood):
    
    def initialize(self):

        if self.DESI_table:
            self.data_table = pd.read_csv(self.BAO_data_path+'.txt',sep='\t',header=0)
            if self.observables == 'distances':
                covmat = pd.read_csv(self.BAO_data_path+'_distances_covmat.txt',sep='\t',header=0)
                covmat.index = covmat.columns
                self.inv_covmat = pd.DataFrame(np.linalg.inv(covmat.values),columns=covmat.columns,index=covmat.index)
            elif self.observables == 'alphas':
                covmat = pd.read_csv(self.BAO_data_path+'_alphas_covmat.txt',sep='\t',header=0)
                covmat.index = covmat.columns
                self.inv_covmat = pd.DataFrame(np.linalg.inv(covmat.values),columns=covmat.columns,index=covmat.index)
        else:
            sys.exit('UNKNOWN DATA FORMAT')
       
        zmax = 4.
        self.z_camb = np.linspace(0.001, zmax, 10000)


    def get_requirements(self):
        # Requirements are the output of the theory code that you are using
        requirements = {'DM': None,
                        'DH': None,
                        'DV': None,
                        'rdrag': None}

        return requirements
    
    def logp(self, **params_values): 

        #MM: add rdrag switch here
        rdrag = params_values['rdrag']

        if self.observables == 'distances':
            DV_only     = self.data_table[self.data_table.isna().any(axis=1)]
            chi2_single = (DV_only['DV/rd']-self.provider.get_result('DV')(DV_only['z'])/params_values['rdrag'])**2/DV_only['DV_err']**2

            allobs = self.data_table[~self.data_table.isna().any(axis=1)]

            split_cols = [col.split('_') for col in self.inv_covmat.columns]
            datavec    = np.array([allobs.iloc[int(ind)-1][obs] for obs,ind in split_cols])
            thvec = []
            for obs,ind in split_cols:
                if obs == 'DV/rd':
                    thvec.append(self.provider.get_result('DV')(allobs.iloc[int(ind)-1]['z'])/rdrag)
                elif obs == 'DM/DH':
                    thvec.append(self.provider.get_result('DM')(allobs.iloc[int(ind)-1]['z'])/self.provider.get_result('DH')(allobs.iloc[int(ind)-1]['z']))

            theoryvec = np.array(thvec)

            chi2_all = np.dot(theoryvec-datavec,np.dot(self.inv_covmat,theoryvec-datavec))

            chi2 = chi2_all+chi2_single

        elif self.observables == 'alphas':
            iso_only     = self.data_table[self.data_table.isna().any(axis=1)]
            chi2_single = (iso_only['alpha_iso']-(self.provider.get_result('DV')(iso_only['z'])/params_values['rdrag'])/iso_only['DV/rd_fid'])**2/iso_only['iso_err']**2

            allobs = self.data_table[~self.data_table.isna().any(axis=1)]

            split_cols = [col.split('_') for col in self.inv_covmat.columns]
            datavec    = np.array([allobs.iloc[int(ind)-1][obs1+'_'+obs2] for obs1,obs2,ind in split_cols])
            thvec = []
            for obs1,obs2,ind in split_cols:
                if obs1+'_'+obs2 == 'alpha_iso':
                    thvec.append((self.provider.get_result('DV')(allobs.iloc[int(ind)-1]['z'])/rdrag)/allobs.iloc[int(ind)-1]['DV/rd_fid'])
                elif obs1+'_'+obs2 == 'alpha_AP':
                    thvec.append((self.provider.get_result('DH')(allobs.iloc[int(ind)-1]['z'])/
                                  self.provider.get_result('DM')(allobs.iloc[int(ind)-1]['z']))/allobs.iloc[int(ind)-1]['DM/DH_fid'])

            theoryvec = np.array(thvec)

            chi2_all = np.dot(theoryvec-datavec,np.dot(self.inv_covmat,theoryvec-datavec))

            chi2 = chi2_all+chi2_single


        loglike = -0.5*chi2

        return loglike
