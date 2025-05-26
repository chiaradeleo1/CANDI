import sys
import numpy as np
import pandas as pd

from cobaya.run import run

from bios import read
from copy import deepcopy

from likelihood.BAO_likelihood         import BAOLike
from likelihood.SN_likelihood          import SNLike
from likelihood.SNnoprior_likelihood   import SNnopriorLike
from likelihood.GW_likelihood          import GWLike
from theory_code.cobaya_theory_wrapper import CalcDist

import pprint
import time
import os
pp = pprint.PrettyPrinter(indent=4)


info = read(sys.argv[1])

tini = time.time()
#MM: we should find a better and more general way to fill the likelihood

info['likelihood'] = {}
if info['BAO_data'] != None:
    info['likelihood']['BAOLike'] = {'external': BAOLike,
                                     'BAO_data_path': info['BAO_data']['path'],
                                     'use_noisy_data': info['BAO_data']['noisy']}

if info['SN_data'] != None:
    if info['SH0ES_prior']  == True:
        info['likelihood']['SNLike'] =  {'external': SNLike,
                                                'SN_data_path': info['SN_data']['path'],
                                                'use_noisy_data': info['SN_data']['noisy']}
    else:
        info['likelihood']['SNnopriorLike'] =  {'external': SNnopriorLike,
                                                'SN_data_path': info['SN_data']['path'],
                                                'use_noisy_data': info['SN_data']['noisy']}
    
if info['GW_data'] != None:
    info['likelihood']['GWLike'] =  {'external': GWLike,
                                     'GW_data_path': info['GW_data']['path'],
                                     'use_noisy_data': info['GW_data']['noisy']}

if len(list(info['likelihood'].keys())) == 0:
    sys.exit('NO LIKELIHOOD LOADED!!!')
else:
    print('Likelihoods loaded: ',list(info['likelihood'].keys()))

#MMnote: pass theory options
info['theory'] = {'CalcDist': {'external': CalcDist,
                               'DDR_model': info['DDR_model']}}

if 'BBN' in info:
    info['theory']['CalcDist']['BBN'] = info['BBN']
else:
    info['theory']['CalcDist']['BBN'] = False

if 'resume' not in info:    
    info['force'] = True
output_folder = [ ]


for folder in output_folder:    
    if not os.path.exists(folder):
        os.makedirs(folder)

if info['sampler'] == 'profiling':
    print('Profiling started')
    info['sampler'] = {'evaluate': {'override': {par: par_dict['ref']['loc'] for par,par_dict in info['params'].items()
                                                 if type(par_dict) == dict and 'ref' in par_dict}}}
    del info['output']
    updated_info,sampler = run(info)
    print('Profiling ended')
    tend = time.time()
    print('Time elapsed: ',tend-tini)
    sys.exit()

if info['sampler'] == 'MH':
    print('Running with Metropolis-Hastings')
    from cobaya.run import run

    info['sampler'] = {'mcmc': {'max_tries':100000}} #MMnote: MCMC options to be read from input
    info['output']  = info['output']+'_MH'

    updated_info,sampler = run(info)
    
#Nautilus to be added
elif info['sampler'] == 'Nautilus':
    print('Running with Nautilus')
    from samplers.samplers_interface import nautilus_interface

    info['sampler'] = {'nautilus': {'num_threads': 1,
                                    'pool': 1,
                                    'n_live': 4000,
                                    'n_batch': 512,
                                    'n_networks': 16}}
    info['output']  = info['output']+'_nautilus'

    nautilus = nautilus_interface(info)
elif args.sampler == 'Fisher':
    sys.exit('Fisher to be adder yet :(')
else:
    sys.exit('Unknown sampler: {}'.format(info['sampler']))

