import sys
import numpy as np
import pandas as pd

from cobaya.run import run

from bios import read
from copy import deepcopy

from likelihood.likelihood import DDRLike
from theory_code.DDRtheory import DDRtheory

sys.path.append('../')

import pprint
pp = pprint.PrettyPrinter(indent=4)



info = read(sys.argv[1])
likesets = deepcopy(info['likelihood']['DDR'])
info['likelihood'] = {'DDR': {'external': DDRLike,
                              'BAO_data_path': likesets['BAO_data_path'],
                              'SN_data_path': likesets['SN_data_path']}}

info['theory'] = {'DDRtheory': {'external': DDRtheory}}


if info['BBN_prior']:
    info['params']['ombh2']['prior'] = {'dist': 'norm',
                                            'loc': 0.02218,
                                            'scale': 0.00055}


if info['SH0ES_prior']:
    info['params']['H0']['prior'] = {'dist': 'norm',
                                    'loc': 73.04,
                                    'scale': 1.04}

info['force'] = True

updated_info,sampler = run(info)
