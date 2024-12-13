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
        if settings['epsilon_model'] == 'constant':
            self.eta_EM = self.get_eta(params['epsilon0_EM'])
            self.eta_GW = self.get_eta(params['epsilon0_GW'])
        else:
            sys.exit('Unknown DDR breaking model: {}'.format(model))

    def get_eta(self,epsilon):
        
        if self.settings['pade'] == 'True':
            eta = interp1d(self.zcalc, 1 + (epsilon*np.log(1+self.zcalc))/(1-(epsilon*0.5*np.log(1+self.zcalc))+ (epsilon/12 * (np.log(1+self.zcalc))**2) ))
        else:
            eta = interp1d(self.zcalc,(1+self.zcalc)**epsilon)

        return eta
