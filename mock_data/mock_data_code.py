import sys
import numpy  as np
import pandas as pd

from copy import deepcopy
from scipy.interpolate import interp1d


class MockCalcs:
    def __init__(self,settings, obs_settings,params, theory):
        
        ###################
        self.params = params
        self.theory = theory

        if 'BAO' in obs_settings:
            self.settings_BAO = obs_settings['BAO']
            self.data_BAO = self.get_BAO_mock()

        if 'SN' in obs_settings:
            self.settings_SN = obs_settings['SN']
            self.data_SN = self.get_SN_mock()

        if 'GW' in obs_settings:
            self.settings_GW = obs_settings['GW']
            self.data_GW = self.get_GW_mock()
        


    def get_BAO_mock(self):

        if self.settings_BAO['distribution']=='binned':
            zmin_BAO = self.settings_BAO['zmin']
            zmax_BAO = self.settings_BAO['zmax']
            N_bin_BAO = self.settings_BAO['N_bin']
            bin_edges = np.linspace(zmin_BAO, zmax_BAO, N_bin_BAO)
            z_BAO   = (bin_edges[:-1] + bin_edges[1:]) / 2
        else:
            sys.exit('Unknown BAO distribution: {}'.format(self.settings_BAO['distribution']))

        DH = self.theory.DH(z_BAO)
        DV = self.theory.DV(z_BAO)
        DM = self.theory.DM(z_BAO) 

        if type(self.settings_BAO['error_type']) == float:

            DH_error = DH*self.settings_BAO['error_type']
            DV_error = DV*self.settings_BAO['error_type']
            DM_error = DM*self.settings_BAO['error_type']
        else:
            sys.exit('Unknown BAO error type: {}'.format(self.settings_BAO['error_type']))

        if self.settings_BAO['spread_data']:
            DH_noisy = np.random.normal(DH, DH_error)
            DV_noisy = np.random.normal(DV, DV_error)
            DM_noisy = np.random.normal(DM, DM_error)
        else:
            DH_noisy = DH
            DV_noisy = DV
            DM_noisy = DM

        #DH_error = 0.05 * DH_noisy
        #DV_error = 0.05 * DV_noisy
        #DM_error = 0.05 * DM_noisy

        data_BAO_DHDM = {'z' : z_BAO,
                         'DH': DH_noisy,
                         'err_DH': DH_error,
                         'DM': DM_noisy,
                         'err_DM': DM_error}
                
        if self.settings_BAO['correlation'] == False:
            error_BAO= np.concatenate((data_BAO_DHDM['err_DH'], data_BAO_DHDM['err_DM']),axis=0)

            covmat_DHDM = np.zeros((len(error_BAO), len(error_BAO)))

            np.fill_diagonal(covmat_DHDM, error_BAO ** 2)
        else:
            sys.exit('Correlation in BAO measurements not implemented yet')

        data_BAO_DHDM['covmat'] = covmat_DHDM

        np.save(self.settings_BAO['BAO_file_path']+'_DHDM.npy', data_BAO_DHDM)

        data_BAO_DV  = {'z' : z_BAO,
                        'DV': DV_noisy,
                        'err_DV': DV_error,}
        
        covmat_DV = np.zeros((len(DV_error), len(DV_error)))
        np.fill_diagonal(covmat_DV, DV_error ** 2)
        data_BAO_DV['covmat'] = covmat_DV

        np.save(self.settings_BAO['BAO_file_path']+'_DV.npy', data_BAO_DV)

        data_BAO = {'z': z_BAO,
                    'DH': DH_noisy,
                    'err_DH': DH_error,
                    'DM': DM_noisy,
                    'err_DM': DM_error,
                    'DV': DV_noisy,
                    'err_DV': DV_error}
        
        return data_BAO

    def get_SN_mock(self):

        N_SN = self.settings_SN['N_SN']

        if self.settings_SN['distribution'] == 'Euclid':
            dN_dz = lambda z: 1.53e-4 * ((1+z)/1.5)**2.14 * (self.params['H0']/70)**3
            z = np.linspace(self.settings_SN['zmin'],self.settings_SN['zmax'], N_SN*2)
            p_z = dN_dz(z)
            p_z /= np.sum(p_z)
            z_SN = np.sort(np.random.choice(z,size=N_SN, p=p_z, replace = False))
        else:
            sys.exit('Unknown BAO distribution: {}'.format(self.settings_SN['distribution']))
        
        mB = interp1d(z_SN, 5* np.log10(self.theory.DL_EM(z_SN))+ 25 + self.params['MB'])(z_SN)

        if type(self.settings_SN['error_type']) == float:

            mB_error = mB*self.settings_SN['error_type']

        else:
            sys.exit('Unknown SN error type: {}'.format(self.settings_SN['error_type']))

        if self.settings_SN['spread_data']:
            mB_noisy = np.random.normal(mB, mB_error)
        else: 
            mB_noisy = mB

        #mB_error = 0.05 * mB_noisy

        data_SN = {'z' : z_SN,
                   'mB': mB_noisy,
                   'err_mB': mB_error}
                
        if self.settings_SN['correlation'] == False:
            covmat_SN = np.zeros((len(mB_error), len(mB_error)))

            np.fill_diagonal(covmat_SN, mB_error ** 2)
        else:
            sys.exit('Correlation in SN measurements not implemented yet')

        data_SN['covmat'] = covmat_SN

        np.save(self.settings_SN['SN_file_path']+'.npy', data_SN)

        return data_SN
    

    def get_GW_mock(self):

        N_gw = self.settings_GW['N_gw']
        z_calc = np.linspace(self.settings_GW['zmin'], self.settings_GW['zmax'], N_gw*2)

        if self.settings_GW['distribution'] == 'BNS':

            merger_rate = np.array([self.BNS_merger_rate(z) for z in z_calc])

            r_values = np.array([2.99*1e5/self.theory.Hz(z) for z in z_calc])
            p_z      = 4 * np.pi * r_values**2 * merger_rate / (self.theory.Hz(z_calc) * (1 + z_calc))
            p_z     /= np.sum(p_z) 
            z_GW     = np.sort(np.random.choice(z_calc, size = N_gw, p = p_z, replace = False))
        
        else:
            sys.exit('Unknown GW distribution: {}'.format(self.settings_GW['distribution']))

        dL_GW = self.theory.DL_GW(z_GW)

        if type(self.settings_GW['error_type']) == float:

            dL_GW_error = dL_GW*self.settings_GW['error_type']

        elif self.settings_GW['error_type'] == 'observational_error':

            sys.exit('Observational error not available yet')

            #SNR cannot be computed with uniform distribution. It correlates with dL!
            snr = np.random.uniform(10, 100, size=N_gw)  
            sigma_L = 0.05 * z_GW * dL_GW               
            sigma_i = 2 * dL_GW / snr                    
            dL_GW_error = np.sqrt(sigma_L**2 + sigma_i**2) 

        if self.settings_GW['spread_data']:
            dL_GW_noisy = np.random.normal(dL_GW,dL_GW_error)
        else:
            dL_GW_noisy = dL_GW
        

        data_GW = {'z' : z_GW,
                   'dL': dL_GW_noisy,
                   'err_dL': dL_GW_error}
                
        if self.settings_GW['correlation'] == False:
            covmat_GW = np.zeros((len(dL_GW_error), len(dL_GW_error)))

            np.fill_diagonal(covmat_GW, dL_GW_error ** 2)
        else:
            sys.exit('Correlation in GW measurements not implemented yet')

        data_GW['covmat'] = covmat_GW

        np.save(self.settings_GW['GW_file_path']+'.npy', data_GW)

        return data_GW


    def BNS_merger_rate(self,z):
        if z <= 1:
            return 1 + 2 * z
        elif 1 < z < 5:
            return 3/4 * (5-z)
        else:
            return 0
