import sys
import numpy  as np
import pandas as pd

from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.integrate   import trapezoid
from scipy.stats       import rv_continuous,norm

from GWFish.modules.detection    import Network,Detector
from GWFish.modules.fishermatrix import compute_network_errors,compute_detector_fisher,compute_detector_fisher
from GWFish.modules.waveforms    import IMRPhenomD, TaylorF2

class MockCalcs:
    def __init__(self,settings, obs_settings,params, theory):

        
        ###################
        self.params = params
        self.theory = theory

        print('CREATING MOCKS FOR {}'.format(list(obs_settings.keys())))

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
        elif self.settings_BAO['distribution']=='SKAO':
            bin_edges = np.linspace(0.2,2.0,18)
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
        elif self.settings_BAO['error_type'] == 'SKAO':
            DH_error = DH*np.array([1.8,1.17,0.99,0.79,0.7,0.64,0.61,0.57,0.67,0.69,0.8,0.95,1.16,1.51,2.12,3,5.3])*1.e-2
            DM_error = DM*np.array([1.1,0.76,0.59,0.5,0.44,0.42,0.4,0.38,0.42,0.45,0.54,0.63,0.83,1.12,1.55,2.20,3.95])*1.e-2
        else:
            sys.exit('Unknown BAO error type: {}'.format(self.settings_BAO['error_type']))

        DH_noisy = np.random.normal(DH, DH_error)
        if self.settings_BAO['distribution'] != 'SKAO':
            DV_noisy = np.random.normal(DV, DV_error)
        DM_noisy = np.random.normal(DM, DM_error)

        data_BAO_DHDM = {'z' : z_BAO,
                         'DH_noisy': DH_noisy,
                         'DH': DH,
                         'err_DH': DH_error,
                         'DM_noisy': DM_noisy,
                         'DM': DM,
                         'err_DM': DM_error}
                
        if self.settings_BAO['correlation'] == False:
            error_BAO= np.concatenate((data_BAO_DHDM['err_DH'], data_BAO_DHDM['err_DM']),axis=0)

            covmat_DHDM = np.zeros((len(error_BAO), len(error_BAO)))

            np.fill_diagonal(covmat_DHDM, error_BAO ** 2)
        else:
            sys.exit('Correlation in BAO measurements not implemented yet')

        #Creating dataframe to save to file
        data_df   = pd.DataFrame.from_dict(data_BAO_DHDM)
        covmat_df = pd.DataFrame(covmat_DHDM,columns=['DH_{}'.format(i) for i in data_df.index]+['DM_{}'.format(i) for i in data_df.index])
        covmat_df.index = covmat_df.columns

        data_df.to_csv(self.settings_BAO['BAO_file_path']+'_data_DHDM.txt',header=True,index=False,sep='\t')
        covmat_df.to_csv(self.settings_BAO['BAO_file_path']+'_covmat_DHDM.txt',header=True,index=False,sep='\t')

        if self.settings_BAO['distribution'] != 'SKAO':
            data_BAO_DV  = {'z' : z_BAO,
                            'DV_noisy': DV_noisy,
                            'DV': DV,
                            'err_DV': DV_error,}
        
            covmat_DV = np.zeros((len(DV_error), len(DV_error)))
            np.fill_diagonal(covmat_DV, DV_error ** 2)

            #Creating dataframe to save to file
            data_df   = pd.DataFrame.from_dict(data_BAO_DV)
            covmat_df = pd.DataFrame(covmat_DV,columns=['z{}'.format(i) for i in data_df.index])
            covmat_df.index = covmat_df.columns
   
            data_df.to_csv(self.settings_BAO['BAO_file_path']+'_data_DV.txt',header=True,index=False,sep='\t')
            covmat_df.to_csv(self.settings_BAO['BAO_file_path']+'_covmat_DV.txt',header=True,index=False,sep='\t')

        data_BAO = {'z': z_BAO,
                    'DH_noisy': DH_noisy,
                    'DH': DH,
                    'err_DH': DH_error,
                    'DM_noisy': DM_noisy,
                    'DM': DM,
                    'err_DM': DM_error}
        if self.settings_BAO['distribution'] != 'SKAO':
            data_BAO['DV_noisy'] = DV_noisy
            data_BAO['DV'] = DV
            data_BAO['err_DV'] = DV_error

        print('CREATED BAO DATASET')
        
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
        
        mB = self.theory.mB(z_SN)

        if type(self.settings_SN['error_type']) == float:

            mB_error = mB*self.settings_SN['error_type']

        elif self.settings_SN['error_type'] == 'LSST-like':

            sigma_flux = 0.01 
            sigma_scat = 0.025
            sigma_intr = 0.12

            mB_error = np.array([np.sqrt((np.random.normal(loc=0.,scale=0.01)*z)**2+sigma_flux**2+sigma_scat**2.+sigma_intr**2.) for z in z_SN])

        else:
            sys.exit('Unknown SN error type: {}'.format(self.settings_SN['error_type']))

        mB_noisy = np.random.normal(mB, mB_error)

        #mB_error = 0.05 * mB_noisy

        data_SN = {'z' : z_SN,
                   'mB_noisy': mB_noisy,
                   'mB': mB,
                   'err_mB': mB_error}
                
        if self.settings_SN['correlation'] == False:
            covmat_SN = np.zeros((len(mB_error), len(mB_error)))

            np.fill_diagonal(covmat_SN, mB_error ** 2)
        else:
            sys.exit('Correlation in SN measurements not implemented yet')

        #Creating dataframe to save to file
        data_df   = pd.DataFrame.from_dict(data_SN)
        covmat_df = pd.DataFrame(covmat_SN,columns=['z{}'.format(i) for i in data_df.index])
        covmat_df.index = covmat_df.columns

        data_df.to_csv(self.settings_SN['SN_file_path']+'_data.txt',header=True,index=False,sep='\t')
        covmat_df.to_csv(self.settings_SN['SN_file_path']+'_covmat.txt',header=True,index=False,sep='\t')

        print('CREATED SN DATASET')

        return data_SN
    
    def get_GW_mock(self):

        N_gw = self.settings_GW['N_gw']
        z_calc = np.linspace(self.settings_GW['zmin'], self.settings_GW['zmax'], N_gw)#MM: there was *2) mltiplying Ngw here, why?

        if self.settings_GW['distribution'] == 'BNS':

            rate = np.array([self.BNS_merger_rate(z) for z in z_calc])

            unnorm      = rate*((4*np.pi*(self.theory.comoving(z_calc))**2)/
                                (self.theory.Hz(z_calc)*(1+z_calc)))
            integral    = trapezoid(unnorm,x=z_calc)

            norm      = 1/integral
            norm_dist = norm*unnorm

            prob_z = interp1d(z_calc,norm_dist,kind='linear',bounds_error=False,fill_value=0)

            z_GW = self.get_events_redshifts(prob_z,self.settings_GW['zmin'],self.settings_GW['zmax'],N_gw)

        
        else:
            sys.exit('Unknown GW distribution: {}'.format(self.settings_GW['distribution']))

        dL_GW = self.theory.DL_GW(z_GW)

        if type(self.settings_GW['error_type']) == float:

            dL_GW_error = dL_GW*self.settings_GW['error_type']

        elif self.settings_GW['error_type'] == 'GWfish':

            observed    = self.get_realistic_error_GW(z_GW,dL_GW)
            z_GW        = observed['z']
            dL_GW       = observed['luminosity_distance']
            dL_GW_error = observed['err_luminosity_distance']

        else:
            sys.exit('Unknown GW error type: {}'.format(self.settings_GW['error_type']))

        dL_GW_noisy = np.random.normal(dL_GW,dL_GW_error)
        

        data_GW = {'z' : z_GW,
                   'dL_noisy': dL_GW_noisy,
                   'dL': dL_GW,
                   'err_dL': dL_GW_error}
        

        
        #data_GW['SNR'] = data_GW['dL']/data_GW['err_dL']

        if self.settings_GW['correlation'] == False:
            covmat_GW = np.zeros((len(dL_GW_error), len(dL_GW_error)))

            np.fill_diagonal(covmat_GW, dL_GW_error ** 2)
        else:
            sys.exit('Correlation in GW measurements not implemented yet')

        #Creating dataframe to save to file
        data_df   = pd.DataFrame.from_dict(data_GW)
        covmat_df = pd.DataFrame(covmat_GW,columns=['z{}'.format(i) for i in data_df.index])
        covmat_df.index = covmat_df.columns

        data_df.to_csv(self.settings_GW['GW_file_path']+'_data.txt',header=True,index=False,sep='\t')
        covmat_df.to_csv(self.settings_GW['GW_file_path']+'_covmat.txt',header=True,index=False,sep='\t')

        print('CREATED GW DATASET')

        return data_GW
    

    def BNS_merger_rate(self,z):
        if z <= 1:
            return 1 + 2 * z
        elif 1 < z < 5:
            return 3/4 * (5-z)
        else:
            return 0

    def get_events_redshifts(self,pz,zmin,zmax,Nsamp):

        class MyDist(rv_continuous):
            def _pdf(self, x):
                return pz(x)


        mydist = MyDist(a=zmin,b=zmax)

        zs = mydist.rvs(size=Nsamp)

        return zs

    def get_realistic_error_GW(self,z_GW,dL_GW):

        Ngw = len(z_GW)

        th_features = pd.DataFrame.from_dict({'z': z_GW,
                                              'luminosity_distance': dL_GW})

        #Generating random features
        #Sky location
        th_features['dec']  = np.arccos(np.random.uniform(low=-1, high=1,size=Ngw))
        th_features['ra']   = np.random.uniform(low=0, high=2*np.pi,size=Ngw)

        #Polarization
        th_features['psi'] = np.random.uniform(low=0, high=2*np.pi,size=Ngw)

        #Phase
        th_features['phase'] = np.random.uniform(low=0,high=2.*np.pi,size=Ngw)

        #System inclination
        th_features['theta_jn'] = np.arccos(np.random.uniform(low=-1, high=1,size=Ngw))

        #MM: check this!!!
        th_features['geocent_time'] = np.random.uniform(1735257618, 1766793618,size=Ngw)

        #MM: ASSUMING MONOCHROMATIC MASS

        th_features['mass_1'] = 1.4
        th_features['mass_2'] = 1.4
        Mtot = (th_features['mass_1']+th_features['mass_2'])
        eta  = th_features['mass_1']*th_features['mass_2']/(th_features['mass_1']+th_features['mass_2'])**2
        th_features['chirp_mass'] = (1+th_features['z'])*Mtot*eta**(3/5)
        th_features['mass_ratio'] = eta

        #MM: hard coded?
        freepars = ['theta_jn','luminosity_distance']
        SNR_cut  = 8

        detected, snr, errors, sky_localization = compute_network_errors(network = Network(detector_ids = self.settings_GW['survey'],
                                                                                           detection_SNR = (0., 0)),
                                                                                           parameter_values = th_features,
                                                                                           fisher_parameters=freepars,
                                                                                           waveform_model = 'IMRPhenomD',
                                                                                           save_matrices=True)


        for i,par in enumerate(freepars):
            th_features['err_'+par] = errors[:,i]
        
        th_features['SNR'] = snr

        observed = th_features[th_features['SNR']>=SNR_cut]

        return observed
