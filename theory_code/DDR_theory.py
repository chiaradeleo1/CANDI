#MM: this module is currently empty and sad :(
#Here we could add fancy ways to obtain eta(z) given specific theories

import sys
import numpy  as np
import pandas as pd

from copy import deepcopy
from scipy.interpolate import interp1d


class DDRCalcs:

    def __init__(self,model,params,zcalc):

        self.zcalc = zcalc

        if model == 'constant':
            self.eta_EM = self.get_eta_poly(params['epsilon0_EM'])
            
            self.eta_GW = self.get_eta_poly(params['epsilon0_GW'])
        
        elif model == 'padè':
            self.eta_EM = self.get_eta_pade_approximant(params['epsilon0_EM'])
            
            self.eta_GW = self.get_eta_pade_approximant(params['epsilon0_GW'])
            
        else:
            sys.exit('Unknown DDR breaking model: {}'.format(model))

    def get_eta_poly(self,epsilon):

        eta = interp1d(self.zcalc,(1+self.zcalc)**epsilon)

        return eta
    
    def get_eta_pade_approximant(self, epsilon):

        eta = interp1d(self.zcalc, 1 + (epsilon*np.log(1+self.zcalc))/(1-(epsilon*0.5*np.log(1+self.zcalc))+ (epsilon/12 * (np.log(1+self.zcalc))**2) ))
        
        return eta