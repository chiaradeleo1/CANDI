import sys
import numpy as np
import pandas as pd

from cobaya.run import run

from bios import read
from copy import deepcopy

from likelihood.BAO_likelihood   import BAOLike
from likelihood.SN_likelihood    import SNLike
from theory_code.distance_theory import CalcDist

import pprint
pp = pprint.PrettyPrinter(indent=4)



info = read(sys.argv[1])

if info['BAO_data'] != None:
    info['likelihood'] = {'BAOLike': {'external': BAOLike,
                                      'BAO_data_path': info['BAO_data']}}

if info['SN_data'] != None:
    info['likelihood'] = {'SNLike': {'external': SNLike,
                                      'SN_data_path': info['SN_data']}}

info['theory'] = {'CalcDist': {'external': CalcDist}}
#                               'extra_args': {'camb_path': info['camb_path']}}}

info['force'] = True

if info['sampler'] == 'MH':
    from cobaya.run import run
    info['sampler'] = {'mcmc': {'max_tries':100000}}
    updated_info,sampler = run(info)

elif info['sampler'] == 'Nautilus':
    sys.exit('NOT YET!!!')
    from samplers.samplers_interface import nautilus_interface
    info['sampler'] = {'nautilus': {'num_threads': 1,
                                    'pool': 1,
                                    'n_live': 4000,
                                    'n_batch': 512,
                                    'n_networks': 16}}
    nautilus = nautilus_interface(info)
else:
    sys.exit('Unknown sampler: {}'.format(info['sampler']))

