import sys
import numpy  as np
import pandas as pd

from copy import deepcopy
from scipy.interpolate import interp1d

from cobaya.theory import Theory

class CalcDist(Theory):

    def initialize(self):
        """called from __init__ to initialize"""

        #MM: eventually to be made options
        #They should be ok in most cases
        self.zmin  = 0.001
        self.zmax  = 5.
        self.Nz    = 1000
        self.zdrag = 1060
        ##################################

        self.zcalc = np.linspace(self.zmin,self.zmax,self.Nz)

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_can_provide(self):

        return ['DM','DH','DV']

    def get_can_provide_params(self):
        return ['rdrag','omegaL']

    def calculate(self, state, want_derived=True, **params_values_dict):

        camb_results = self.call_camb(params_values_dict)

        state['DM'] = camb_results['DM/rd']
        state['DH'] = camb_results['DH/rd']
        state['DV'] = camb_results['DV/rd']
        #MM: this to be generalized
        state['DL_EM'] = self.get_dL(camb_results['dA'],params_values_dict['epsilon0_EM'])
        state['DL_GW'] = self.get_dL(camb_results['dA'],params_values_dict['epsilon0_GW'])
        ###########################
        state['mB'] = self.get_mB(state['DL_EM'],params_values_dict['MB'])
        state['derived'] = {'rdrag': camb_results['rdrag'], 'omegaL': camb_results['omegaL']}

    def call_camb(self,params):

        camb_params = deepcopy(params)
        #MM: neutrinos to be added
        omnuh2 = 0.#camb_params['mnu']/neutrino_mass_fac*(camb_params['nnu']/3.0)**0.75
        camb_params['omch2'] = camb_params.pop('omegam')*(camb_params['H0']/100)**2.-camb_params['ombh2']-omnuh2
        del camb_params['MB']
        del camb_params['epsilon0_EM']
        del camb_params['epsilon0_GW']


        try:
            #MM: path to camb to be made customizable
            import camb
            pars = camb.set_params(**camb_params)
            results = camb.get_background(pars)

            Hz    = results.h_of_z
            comov = (1+self.zcalc)*results.angular_diameter_distance2(self.zmin,np.array([z for z in self.zcalc]))
            rdrag = results.sound_horizon(self.zdrag)

        except Exception as e: 
            sys.exit('SOMETHING HORRIBLE HAPPENED!!\n {}'.format(e))


        theory = {'DM/rd': interp1d(self.zcalc,comov/rdrag),
                  'DH/rd': interp1d(self.zcalc,1/(Hz(self.zcalc)*rdrag)),
                  'DV/rd': interp1d(self.zcalc, (self.zcalc*comov**2/Hz(self.zcalc))**(1/3)/rdrag),
                  'dA': results.angular_diameter_distance,
                  'rdrag': rdrag,
                  'omegaL': results.get_Omega('de',z=0)}


        return theory


    def get_dL(self,dA,epsilon):
        
        dL = interp1d(self.zcalc,(1+self.zcalc)**(2+epsilon)*dA(self.zcalc))

        return dL

    def get_mB(self,dL,MB):

        mB = interp1d(self.zcalc,5*np.log10(dL(self.zcalc))+MB+25,kind='linear')

        return mB
