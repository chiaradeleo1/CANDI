#MM: this module is currently empty and sad :(
#Here we could add fancy ways to obtain eta(z) given specific theories

import sys
import numpy  as np
import pandas as pd

from copy import deepcopy
from scipy.interpolate import interp1d


class DDRCalcs:

    def __init__(self,settings,params,zcalc):

        self.zcalc = zcalc
        self.settings = settings
        self.z_LSS = 1100

        if settings['epsilon_model'] == 'polynomial':
            self.eta_EM = self.get_eta(params['epsilon0_EM'],params['n_EM'], params['a_EM'])
            self.eta_GW = self.get_eta(params['epsilon0_GW'], params['n_GW'], params['a_GW'])    
           


                
        else:
            sys.exit('Unknown DDR breaking model: {}'.format(settings['epsilon_model']))

    def get_eta(self,epsilon, n, a):
        if self.settings['pade'] == True:
            if n != 1:
                sys.exit('Pade approximant implemented only for n=1')
            else:
                eta_pade = 1+ (2*self.zcalc*(self.zcalc-self.z_LSS)*epsilon)/(2*(self.zcalc-self.z_LSS)-self.zcalc*self.z_LSS)
                eta = interp1d(self.zcalc, eta_pade)
        else:
            eta = interp1d(self.zcalc,(1+(a*self.zcalc**n))**epsilon)

        return eta
