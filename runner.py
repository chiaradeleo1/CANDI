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
                                     'DESI_table': info['BAO_data']['DESI_table'],
                                     'observables': info['BAO_data']['observables']}

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
                               'DDR_model': info['DDR_model'],
                               'fiducial': info['fiducial_path']}}

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

