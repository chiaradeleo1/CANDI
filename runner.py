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

#####PREPARING OUTPUT#####
if 'resume' not in info:
    info['force'] = True
output_folder = [ ]


for folder in output_folder:
    if not os.path.exists(folder):
        os.makedirs(folder)



tini = time.time()
#MM: we should find a better and more general way to fill the likelihood

#####LOAD LIKELIHOODS#####
info['likelihood'] = {}
if info['BAO_data'] != None:
    info['likelihood']['BAOLike'] = {'external': BAOLike,
                                     'BAO_data_path': info['BAO_data']['path'],
                                     'data_format': info['BAO_data']['data_format']}
    if 'observables' in info['BAO_data']:
        info['likelihood']['BAOLike']['observables'] = info['BAO_data']['observables']

if info['SN_data'] != None:
    if info['SH0ES_prior']  == True:
        info['likelihood']['SNLike'] =  {'external': SNLike,
                                         'SN_data_path': info['SN_data']['path']}
    else:
        info['likelihood']['SNnopriorLike'] =  {'external': SNnopriorLike,
                                                'SN_data_path': info['SN_data']['path']}
    
if info['GW_data'] != None:
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

pre_init = TheoryCalcs(None,{'cosmology': info['cosmology'],'parameters': None},None,None,run_all=False)

full_params = deepcopy(basic_params)
for par,val in pre_init.recognized_params.items():
    full_params['params'][par] = val

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

    info['sampler'] = {'nautilus': info['sampler']['options']}

    nautilus = nautilus_interface(info)
else:
    sys.exit('Unknown sampler: {}'.format(info['sampler']))

