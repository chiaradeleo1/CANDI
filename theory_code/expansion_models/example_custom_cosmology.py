import sys,os

import numpy  as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.integrate   import trapezoid

clight = 299792.458

class CustomExpansion:

    def __init__(self,call_name):

        self.label = 'Custom'
        self.recognized_params = {'omegam': 0.32,
                                  'H0': 67.,
                                  'Delta': 0.,
                                  'ombh2': 0.02218}

        if call_name == self.label:
            self.used = True

        else:
            self.used = False 


    def get_cosmology(self,params,settings):

        self.zmin      = settings['zmin']
        self.zmax      = settings['zmax']
        self.Nz        = settings['Nz']

        self.zcalc = np.linspace(self.zmin,self.zmax,self.Nz)

        unknown = [par for par in params.keys() if par not in self.recognized_params]

        if unknown != []:
            sys.exit('Error in {} cosmology code!\n Unknown parameters: {}'.format(self.label,unknown))

        zfine  = np.linspace(min(self.zcalc),max(self.zcalc),len(self.zcalc)*10)
        hubble = params['H0']*np.sqrt(params['omegam']*(1+zfine)**(3+params['Delta'])+(1-params['omegam']))
        Hz = interp1d(zfine,hubble/clight)

        comov_vec = []
        for z in self.zcalc:
            zint = np.linspace(0.001,z,10)
            comov_vec.append(trapezoid([1/Hz(zi) for zi in zint],x=zint))
        comov = np.array(comov_vec)

        Neff  = 3.044
        rdrag = 147.05*(params['omegam']*(params['H0']/100)**2/0.1432)**(-0.23) * (Neff/3.04)**(-0.1) * (params['ombh2']/0.02236)**(-0.13)

        theory = {'H_Mpc': Hz,
                  'H_kmsMpc': interp1d(zfine,hubble),
                  'comoving': interp1d(self.zcalc,comov),
                  'rdrag': rdrag,
                  'omegaL': 1-params['omegam']}

        return theory
