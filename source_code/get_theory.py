import sys
import numpy as np
import pandas as pd


from scipy.interpolate import interp1d
from scipy.integrate   import trapz

from copy      import deepcopy
from itertools import product
from time      import time



class get_theory:

    def __init__(self, params, case):
        self.params = params
        self.case = case
        self.z_camb  = np.linspace(0.001,4.,10000)
        self.dA = self.get_dA(self)
        self.dL = self.get_dL(self)


    def cosmo_dict(self):
        extra_params_keys = ['espilon0', 'epsilon1']
        cosmo_params = {par: params[par] for par in self.params.keys() if par not in extra_params_keys}
        cosmo_dict = {'cosmo_params': cosmo_params}
        extra_params = {par: params[par] for par in self.params.keys() if par in extra_params_keys}
        cosmo_dict['extra'] = extra_params

        return cosmo_dict

    def get_dA(self):
        import camb
        from   camb         import model, initialpower

        
        cosmo = self.cosmo_dict()
        cosmo_pars = cosmo['cosmo_pars']
        pars = camb.set_params(**cosmo_pars)
        results = camb.get_results(pars)
        dA      = results.angular_diameter_distance2(0,self.z_camb)
        return dA
    
    def get_dL(self):
        cosmo = self.cosmo_dict()
        if case = 'constant':
            epsilon0 = cosmo['extra']['epsilon0']
            dL = self.dA*(1+self.z_camb)**(2+epsilon0)
        elif case = 'parametrization' :
            sys.exit('{} case not yet implemented'.format(case))
        else :
            sys.exit('{} case unknown'.format(case))

    def get_theory_BAO