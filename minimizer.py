import sys
import numpy as np
import pandas as pd

from cobaya.run import run
from bios       import read
from copy       import deepcopy

from utils.utils import Suppressor

from samplers.samplers_interface import nautilus_interface, emcee_interface

import pprint
pp = pprint.PrettyPrinter(indent=4)



info = read(sys.argv[1])

temp_like = deepcopy(info['likelihood'])

#MMmod: VERY WEIRD REWRITING OF THE LIKELIHOOD
#       BUT IF I DON'T DO THIS COBAYA DOES NOT FIND EXTRA PARS
#       TO BE FIXED!!!!
if 'F-DESI' in info['likelihood']:
    from DESI_likelihood.likelihood import fakeDESI
    info['likelihood'] = {'F-DESI': {'external': fakeDESI,
                                     'data_path': temp_like['data_path'],
                                     'debug_mode': temp_like['debug_mode'],
                                     'use_calibration': temp_like['use_calibration'],
                                     'use_thetastar': temp_like['use_thetastar'],
                                     'exclude_LRG': temp_like['exclude_LRG']}}

if 'Pplus' in temp_like:
    from Pantheonplus_likelihood.likelihood import Pplus
    info['likelihood']['Pplus'] =   {'external': Pplus,
                                     'data_path': temp_like['Pplus']['data_path']}

if 'wsettings' in info:
    for node in range(len(info['theory']['camb']['extra_args']['znodes'])):
        info['params']['w_{}'.format(node)] = deepcopy(info['wsettings'])
        info['params']['w_{}'.format(node)]['latex'] = 'w_{}'.format(node)

info['sampler'] = {'minimize': None}

updated_info,sampler = run(info)
