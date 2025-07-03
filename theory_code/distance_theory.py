import sys
import numpy  as np
import pandas as pd

from copy import deepcopy
from time import time

from scipy.interpolate import interp1d
from scipy.integrate   import trapezoid,quad

from theory_code.DDR_theory import DDRCalcs

clight = 299792.458


class TheoryCalcs:

    def __init__(self,settings,cosmosets,SNmodel,fiducial,DDR=None,feedback=False):
        
        ##################
        #General settings#
        ##################
        self.zmin      = settings['zmin']
        self.zmax      = settings['zmax']
        self.Nz        = settings['Nz']

        self.zcalc = np.linspace(self.zmin,self.zmax,self.Nz)

        ############################
        #Getting baseline cosmology#
        ############################
        if cosmosets['cosmology'] == 'Standard':
            if feedback:
                print('Running standard CAMB...')
            try:
                tini = time()
                cosmo_results = self.call_camb(cosmosets['parameters'])
                tend = time()
                if feedback:
                    print('CAMB done in {:.2f} s'.format(tend-tini))
            except Exception as e:
                sys.exit('CAMB FAILED!\n {}'.format(e))

        elif cosmosets['cosmology'] == 'Custom':
            if feedback:
                print('Running custom cosmology...')
            try:
                tini = time()
                cosmo_results = self.call_custom(cosmosets['parameters'])
                tend = time()
                if feedback:
                    print('Custom cosmology done in {:.2f} s'.format(tend-tini))
            except Exception as e:
                sys.exit('CUSTOM COSMOLOGY FAILED!\n {}'.format(e))

        else:
            sys.exit('UNKNOWN COSMOLOGY: {}'.format(cosmosets['cosmology']))

        #Changing theory dict elements into class attributes
        for k, v in cosmo_results.items():
            setattr(self,k,v)

        #MM!!! Put here switch to use external rdrag
        #self.rdrag = something

        ################
        #BAO quantities#
        ################
        if feedback:
            print('')
            print('Computing LCDM fiducial')

        if type(fiducial) == str:
            fidtable = pd.read_csv(fiducial,header=0,sep='\t')
            fidcosmo = {'DV_rd': interp1d(fidtable['z'],fidtable['DV_rd']),
                        'DH_DM': interp1d(fidtable['z'],fidtable['DH_DM'])}

            self.alpha_iso = lambda x: (self.DV(x)/self.rdrag)/fidcosmo['DV_rd'](x)
            self.alpha_AP  = lambda x: (self.DH(x)/self.DM(x))/fidcosmo['DH_DM'](x)

        elif type(fiducial) == dict:
            fidcosmo = self.call_camb(fiducial)

            self.alpha_iso = lambda x: (self.DV(x)/self.rdrag)/(fidcosmo['DV'](x)/fidcosmo['rdrag']) 
            self.alpha_AP  = lambda x: (self.DH(x)/self.DM(x))/(fidcosmo['DH'](x)/fidcosmo['DM'](x))

        self.DV_rd     = lambda x: self.DV(x)/self.rdrag
        self.DM_DH     = lambda x: self.DM(x)/self.DH(x)
        self.DM_rd     = lambda x: self.DM(x)/self.rdrag
        self.DH_rd     = lambda x: self.DH(x)/self.rdrag

        ###############
        #Computing DDR#
        ###############

        if DDR != None:
            if feedback:
                print('Computing DDR functions...')
            try:
                tini = time()
                ddr_results  = DDRCalcs(DDR,self.zcalc)
                self.eta_EM = ddr_results.eta_EM
                self.eta_GW = ddr_results.eta_GW
                tend = time()
                if feedback:
                    print('DDR done in {:.2f} s'.format(tend-tini))
            except Exception as e:
                sys.exit('DDR FAILED!\n {}'.format(e))

        else:
            self.eta_EM = lambda x: 1.
            self.eta_GW = lambda x: 1.


        self.DL_EM  = self.get_dL(self.dA,self.eta_EM)
        self.DL_GW  = self.get_dL(self.dA,self.eta_GW)


        ########################
        #Computing SN magnitude#
        ########################
        self.mB = self.get_mB(self.DL_EM,SNmodel)

    def call_camb(self,params):

        #MM: path to camb to be made customizable
        import camb
        pars = camb.set_params(**params)
        results = camb.get_background(pars)

        Hz    = results.h_of_z
        comov = (1+self.zcalc)*results.angular_diameter_distance2(self.zmin,np.array([z for z in self.zcalc]))

        theory = {'H_Mpc': Hz,
                  'H_kmsMpc': results.hubble_parameter,
                  'DM': interp1d(self.zcalc,comov),
                  'DH': interp1d(self.zcalc,1/(Hz(self.zcalc))),
                  'DV': interp1d(self.zcalc, (self.zcalc*comov**2/Hz(self.zcalc))**(1/3)),
                  'dA': results.angular_diameter_distance,
                  'comoving': results.comoving_radial_distance,
                  'rdrag': results.get_derived_params()['rdrag'],
                  'omegaL': results.get_Omega('de',z=0)}

        return theory

    def call_custom(self,params):

        zfine  = np.linspace(min(self.zcalc),max(self.zcalc),len(self.zcalc)*10)
        hubble = params['H0']*np.sqrt(params['omegam']*(1+zfine)**(3+params['Delta'])+(1-params['omegam']))
        Hz = interp1d(zfine,hubble/clight)

        comov_vec = []
        for z in self.zcalc:
            zint = np.linspace(0.001,z,10)
            comov_vec.append(trapezoid([1/Hz(zi) for zi in zint],x=zint))
        comov = np.array(comov_vec)

        rdrag = 1. #MM: to be updated

        theory = {'H_Mpc': Hz,
                  'H_kmsMpc': interp1d(zfine,hubble),
                  'DM': interp1d(self.zcalc,comov),
                  'DH': interp1d(self.zcalc,1/(Hz(self.zcalc))),
                  'DV': interp1d(self.zcalc, (self.zcalc*comov**2/Hz(self.zcalc))**(1/3)),
                  'dA': interp1d(self.zcalc,comov/(1+self.zcalc)),
                  'comoving': interp1d(self.zcalc,comov),
                  'rdrag': rdrag,
                  'omegaL': 1-params['omegam']}

        return theory

    def get_dL(self,dA,eta):

        dL = interp1d(self.zcalc,eta(self.zcalc)*(1+self.zcalc)**2*dA(self.zcalc))

        return dL

    def get_mB(self,dL,MBpars):

        if MBpars['model'] == 'constant':
            MB = lambda x: MBpars['MB']
        else:
            sys.exit('UNKNOWN SN MODEL: {}'.format(MBpars['model']))

        mB = interp1d(self.zcalc,5*np.log10(dL(self.zcalc))+MB(self.zcalc)+25,kind='linear')

        return mB
    
    
