import sys
import numpy  as np
import pandas as pd

from copy import deepcopy
from scipy.interpolate import interp1d

from theory_code.distance_theory import TheoryCalcs

from cobaya.theory import Theory

class CalcDist(Theory):

    
    def initialize(self):
        """called from __init__ to initialize"""
        

        #MM: eventually to be made options
        #They should be ok in most cases
        
        self.settings = {'zmin': 0.001,
                         'zmax': 5.,
                         'Nz': 1000,
                         'zdrag': 1060,
                         'BBN': self.BBN}

        if self.fiducial == None:
            self.fiducial = {'H0': 67.36,
                             'omch2': 0.1200,
                             'ombh2': 0.02237,
                             'omk': 0.,
                             'mnu': 0.06,
                             'nnu': 3.}
       
        ##################################

        self.zcalc = np.linspace(self.settings['zmin'],self.settings['zmax'],self.settings['Nz'])

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_can_provide(self):

        return ['DM','DH','DV','DL_EM','DL_GW','mB','DV_rd','DM_DH','alpha_iso','alpha_AP']

    def get_can_provide_params(self):

        return ['rdrag','omegaL']

    def calculate(self, state, want_derived=True, **params_values_dict):

        ##MM: to be improved!!!
        cosmopars = ['H0','omch2','omk','ombh2','omnuh2','num_nu_massive','num_nu_massless']
        DDRpars   = ['a_EM','n_EM','epsilon0_EM','a_GW','n_GW','epsilon0_GW']
        
        SNmodel   = {'model': 'constant', 'MB': params_values_dict.pop('MB')}
        cosmosets = {'cosmology': 'Standard',
                     'parameters': {k:v for k,v in params_values_dict.items() if k in cosmopars}}

        cosmosets['parameters']['num_nu_massive'] = int(cosmosets['parameters']['num_nu_massive'])
       
        #MM: DDR to be added here
        try:
            DDR = {'eta_model': 'polynomial',
                   'use_pade': self.use_pade,
                   'parameters': {k:v for k,v in params_values_dict.items() if k in DDRpars}}
        except:
            DDR = None

        theory = TheoryCalcs(self.settings,cosmosets,SNmodel,self.fiducial,DDR=DDR)

        state['DM'] = theory.DM 
        state['DH'] = theory.DH
        state['DV'] = theory.DV
        state['DV_rd'] = theory.DV_rd
        state['DM_DH'] = theory.DM_DH
        state['alpha_iso'] = theory.alpha_iso
        state['alpha_AP'] = theory.alpha_AP
        #MM: this to be generalized
        state['DL_EM']  = theory.DL_EM
        state['DL_GW']  = theory.DL_GW
        ###########################
        state['mB']   = theory.mB
        state['derived'] = {'rdrag': theory.rdrag, 'omegaL': theory.omegaL}
