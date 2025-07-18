import sys
import numpy as np
import pandas as pd

from cobaya.run import run

from bios import read
from copy import deepcopy

from likelihood.BAO_likelihood         import BAOLike
from likelihood.SN_likelihood          import SNLike
from likelihood.GW_likelihood          import GWLike
from theory_code.cobaya_theory_wrapper import CalcDist

import pprint
import time
import os
pp = pprint.PrettyPrinter(indent=4)


info = read(sys.argv[1])

#####PREPARING OUTPUT#####
if 'resume' not in info:
    info['force'] = True

tini = time.time()
#MM: we should find a better and more general way to fill the likelihood

#####LOAD LIKELIHOODS#####
#MM: this can probably be made way nicer or at least moved under the carpet in a separate module
info['likelihood'] = {}
if info['BAO_data'] != None:
    print('')
    print('BAO DATA INFO')
    print('')
    info['likelihood']['BAOLike'] = {'external': BAOLike,
                                     'BAO_data_path': info['BAO_data']['path'],
                                     'data_format': info['BAO_data']['data_format']}
    if 'observables' in info['BAO_data']:
        info['likelihood']['BAOLike']['observables'] = info['BAO_data']['observables']

if info['SN_data'] != None:
    print('')
    print('SN DATA INFO')
    print('')
    if info['SN_data']['calibration'] == 'SH0ES':
        print('Using SH0ES calibration')
        calibration = 'SH0ES'
    elif type(info['params']['MB']) == dict and info['params']['MB']['prior']['dist'] == 'norm':
        print('Using Gaussian prior on MB')
        calibration = 'Gaussian'
    else:
        calibration = None
        print('')
        print('Using the SN likelihood analytically marginalized for H0 and MB.\n These parameters will be removed from the sampling if present')
        print('')
        info['params']['H0'] = 73.4
        info['params']['MB'] = -19.2435

    info['likelihood']['SNLike'] =  {'external': SNLike,
                                     'SN_data_path': info['SN_data']['path'],
                                     'use_Pantheon': info['SN_data']['use_Pantheon'],
                                     'calibration': calibration}
    
if info['GW_data'] != None:
    print('')
    print('GW DATA INFO')
    print('')
    info['likelihood']['GWLike'] =  {'external': GWLike,
                                     'GW_data_path': info['GW_data']['path']}

if len(list(info['likelihood'].keys())) == 0:
    sys.exit('NO LIKELIHOOD LOADED!!!')
else:
    print('Likelihoods loaded: ',list(info['likelihood'].keys()))

#####LOAD THEORY#####
#MMnote: pass theory options
info['theory'] = {'CalcDist': {'external': CalcDist,
                               'cosmology': info['cosmology'],
                               'fiducial': info['fiducial_path']}}
if 'DDR_options' in info:
    info['theory']['CalcDist']['DDR_options'] = info['DDR_options']

#MM: this is a workaround to make Cobaya aware of the theory
#parameters defined in the model module
#PROBABLY CAN BE DONE BETTER
basic_params = read('theory_code/basic_parameters.yaml')
from theory_code.distance_theory import TheoryCalcs

pre_init = TheoryCalcs(None,{'cosmology': info['cosmology']},None,None,run_all=False)

full_params = deepcopy(basic_params)
for par,val in pre_init.recognized_params.items():
    full_params['params'][par] = val

#Adding derived parameters
for par in pre_init.derived_params:
    full_params['params'][par] = {'derived': True}

info['theory']['CalcDist']['derived_pars'] = pre_init.derived_params

import yaml
f = open('theory_code/CalcDist.yaml', 'w+')
yaml.dump(full_params,f)


#####SAMPLING#####
if info['sampler']['name'] in ['mcmc','minimize','evaluate']:
    print('Running with Metropolis-Hastings')
    from cobaya.run import run

    info['sampler'] = {info['sampler']['name']: info['sampler']['options']}

    updated_info,sampler = run(info)

elif info['sampler']['name'] == 'nautilus':
    print('Running with Nautilus')
    from samplers.samplers_interface import nautilus_interface

    #MM: this is a switch for nautilus options
    #I got sick of switching from one to the other
    if info['sampler']['options'] == 'poor':
        info['sampler']['options'] = {'num_threads': 1,
                                      'pool': 1,
                                      'n_live': 500,
                                      'n_batch': 64,
                                      'n_networks': 2}
    elif info['sampler']['options'] == 'good':
        info['sampler']['options'] = {'num_threads': 1,
                                      'pool': 1,
                                      'n_live': 4000,
                                      'n_batch': 512,
                                      'n_networks': 16}

    info['sampler'] = {'nautilus': info['sampler']['options']}

    nautilus = nautilus_interface(info)
else:
    sys.exit('Unknown sampler: {}'.format(info['sampler']))


os.remove('theory_code/CalcDist.yaml')
