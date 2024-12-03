import sys
import numpy  as np
import pandas as pd
from bios import read

from copy import deepcopy
from scipy.interpolate import interp1d

from theory_code.distance_theory import TheoryCalcs

from cobaya.theory import Theory

class CalcDist(Theory):

    def initialize(self):
        """called from __init__ to initialize"""
        
        info     = read(sys.argv[1])
        theory_settings = deepcopy(info['settings'])
        #MM: eventually to be made options
        #They should be ok in most cases
        self.settings = {'zmin': 0.001,
                         'zmax': 5.,
                         'Nz': 1000,
                         'zdrag': 1060,
                         'DDR_model': theory_settings['DDR_model'],}
        ##################################

        self.zcalc = np.linspace(self.settings['zmin'],self.settings['zmax'],self.settings['Nz'])

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_can_provide(self):

        return ['DM','DH','DV','DL_EM','DL_GW','mB', 'h_GW']

    def get_can_provide_params(self):
        return ['rdrag','omegaL']

    def calculate(self, state, want_derived=True, **params_values_dict):

        theory = TheoryCalcs(self.settings,params_values_dict)

        state['DM'] = theory.DM 
        state['DH'] = theory.DH
        state['DV'] = theory.DV
        #MM: this to be generalized
        state['DL_EM']  = theory.DL_EM
        state['DL_GW']  = theory.DL_GW
        ###########################
        state['mB']   = theory.mB
        state['derived'] = {'rdrag': theory.rdrag, 'omegaL': theory.omegaL}
