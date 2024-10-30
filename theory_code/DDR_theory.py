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
            self.eta_EM = self.get_eta(params['epsilon0_EM'])
            self.eta_GW = self.get_eta(params['epsilon0_GW'])
        else:
            sys.exit('Unknown DDR breaking model: {}'.format(model))

    def get_eta(self,epsilon):

        eta = interp1d(self.zcalc,(1+self.zcalc)**epsilon)

        return eta
