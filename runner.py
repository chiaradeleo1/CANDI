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
pp = pprint.PrettyPrinter(indent=4)

from argparse import ArgumentParser

parser = ArgumentParser(description='Possible functions for running script')
parser.add_argument('settings',
                    help='path to settings file.')
parser.add_argument("-p", "--profiling",
                    action='store_true',dest="profiling", default=False,
                    help="if True runs only one iteration of the code on the reference values of the parameters")
parser.add_argument("-v", "--verbose",
                    action="store_true", dest="verbose", default=False,
                    help="if True makes the code more chatty (default=False)")

args = parser.parse_args(sys.argv[1:])


info = read(sys.argv[1])


#MM: we should find a better and more general way to fill the likelihood
info['likelihood'] = {}
if info['BAO_data'] != None:
    info['likelihood']['BAOLike'] = {'external': BAOLike,
                                     'BAO_data_path': info['BAO_data']}

if info['SN_data'] != None:
    info['likelihood']['SNLike'] =  {'external': SNLike,
                                     'SN_data_path': info['SN_data']}

if info['GW_data'] != None:
    info['likelihood']['GWLike'] =  {'external': GWLike,
                                     'GW_data_path': info['GW_data']}

if len(list(info['likelihood'].keys())) == 0:
    sys.exit('NO LIKELIHOOD LOADED!!!')
else:
    print('Likelihoods loaded: ',list(info['likelihood'].keys()))

#MMnote: pass theory options
info['theory'] = {'CalcDist': {'external': CalcDist}}
#                               'extra_args': {'camb_path': info['camb_path']}}}

info['force'] = True

if args.profiling:
    print('Profiling started')
    info['sampler'] = {'evaluate': {'override': {par: par_dict['ref']['loc'] for par,par_dict in info['params'].items()
                                                 if type(par_dict) == dict and 'ref' in par_dict}}}
    del info['output']
    updated_info,sampler = run(info)
    print('Profiling ended')
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

