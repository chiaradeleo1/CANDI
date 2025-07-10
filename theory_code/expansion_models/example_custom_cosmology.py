import sys,os

import numpy  as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.integrate   import trapezoid

clight = 299792.458

class CustomExpansion:

    def __init__(self,call_name,settings):

        label = 'Custom'

        if call_name == label:
            print('')
            print('Selected cosmology: {}'.format(label))
            self.used = True

            self.zmin      = settings['zmin']
            self.zmax      = settings['zmax']
            self.Nz        = settings['Nz']

            self.zcalc = np.linspace(self.zmin,self.zmax,self.Nz)
        else:
            self.used = False 


    def get_cosmology(self,params):

        #Check parameters:
        self.recognized_params = ['omegam','H0','Delta']

        unknown = [par for par in params.keys() if par not in self.recognized_params]

        if unknown != []:
            sys.exit('Error in {} cosmology code!\n Unknown parameters: {}'.format(label,unknown))

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
                  'comoving': interp1d(self.zcalc,comov),
                  'rdrag': rdrag,
                  'omegaL': 1-params['omegam']}

        return theory
