import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

import sys

class SNLike(Likelihood):
    
    def initialize(self):

        if self.use_Pantheon:
            if self.calibration == 'SH0ES': 
                print('Using calibrated Pantheon data')
            else:
                print('Using mB with free MB')
            self.dataset_SN = self.build_data()
            covmat          = self.build_covariance()
        else:
            try:
                # NH: adding this try/except to catch missing files
                self.dataset_SN = pd.read_csv(self.SN_data_path+'_data.txt',sep='\s+',header=0)
                covmat          = pd.read_csv(self.SN_data_path+'_covmat.txt',sep='\s+',header=0)
            except FileNotFoundError:
                print("SN dataset and covariance matrix files not found at the following path: {}".format(self.SN_data_path))
                sys.exit(1)

        self.invcovmat  = np.linalg.inv(covmat)

    def build_data(self):
        
        data = pd.read_csv(self.SN_data_path+'_data.txt',sep='\s+')
        self.origlen = len(data)
        if self.calibration == 'SH0ES': ## use the calibrated Pantheon data to include the SH0ES prior instead of including it as a gaussian prior
            self.ww = (data['zHD'] > 0.01) | (np.array(data['IS_CALIBRATOR'], dtype=bool))
        else:
            self.ww = (data['zHD']>0.01)

        self.zCMB = data['zHD'][self.ww] #use the vpec corrected redshift for zCMB
        zHEL = data['zHEL'][self.ww]
        m_obs = data['m_b_corr'][self.ww]
        if self.calibration == 'SH0ES':
            self.is_calibrator = np.array(data['IS_CALIBRATOR'][self.ww], dtype=bool)
            self.cepheid_distance = np.array(data['CEPH_DIST'][self.ww])

        dataset = pd.DataFrame({'z': self.zCMB, 'zHEL': zHEL, 'mB': m_obs})

        return dataset

    def build_covariance(self):

        filename = self.SN_data_path+'_covmat.txt' 
        print("Loading covariance from {}".format(filename))

        # The file format for the covariance has the first line as an integer
        # indicating the number of covariance elements, and the the subsequent
        # lines being the elements.
        # This function reads in the file and the nasty for loops trim down the covariance
        # to match the only rows of data that are used for cosmology

        f = open(filename)
        n = int(len(self.zCMB))
        C = np.zeros((n,n))
        ii = -1
        jj = -1
        mine = 999
        maxe = -999
        for i in range(self.origlen):
            jj = -1
            if self.ww[i]:
                ii += 1
            for j in range(self.origlen):
                if self.ww[j]:
                    jj += 1
                val = float(f.readline())
                if self.ww[i]:
                    if self.ww[j]:
                        C[ii,jj] = val
        f.close()

        print('Done')

        return C


    def get_requirements(self):
        # Requirements are the output of the theory code that you are using
        requirements = {'mB': None}

        return requirements
    
    def logp(self, **params_values): 

        if self.calibration != None:
            if self.calibration == 'SH0ES':
                z = self.dataset_SN['z'].values
                mB_theory = np.zeros_like(z)
                mB_theory[self.is_calibrator] = self.cepheid_distance[self.is_calibrator]
                mB_theory[~self.is_calibrator] = self.provider.get_result('mB')(z[~self.is_calibrator])
                diffvec_SN = mB_theory - self.dataset_SN['mB'].values
            elif self.calibration == 'Gaussian':
                diffvec_SN = self.provider.get_result('mB')(self.dataset_SN['z'].values)-self.dataset_SN['mB'].values
            else:
                sys.exit('Unknown SN calibration: {}'.format(self.calibration))

            loglike = -0.5*np.dot((diffvec_SN),np.dot(self.invcovmat,(diffvec_SN)))
        else:
            diffvec_SN = (self.provider.get_result('mB')(self.dataset_SN['z'].values))-(self.dataset_SN['mB'].values)

            unit = np.ones(len(diffvec_SN))

            a = np.dot(diffvec_SN,np.dot(self.invcovmat,diffvec_SN))
            b = np.dot(diffvec_SN,np.dot(self.invcovmat,unit))
            e = np.dot(unit,np.dot(self.invcovmat,unit))

            chi2_marg = a+ np.log(e/(2*np.pi)) - b**2/e

            loglike = -0.5 * chi2_marg

        return loglike
