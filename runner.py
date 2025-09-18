import sys
import numpy as np
import pandas as pd

from cobaya.run import run

from bios import read
from copy import deepcopy

from likelihood.handler  import LikelihoodHandler
from samplers.handler    import SamplingHandler
from theory_code.handler import TheoryHandler

import pprint
import time
import os
pp = pprint.PrettyPrinter(indent=4)


info = read(sys.argv[1])

#####PREPARING OUTPUT#####
if 'resume' not in info:
    info['force'] = True

#####LOAD LIKELIHOODS#####
LikeSets = LikelihoodHandler(info)
info['likelihood'] = LikeSets.like_dict

if len(list(info['likelihood'].keys())) == 0:
    sys.exit('NO LIKELIHOOD LOADED!!!')
else:
    print('Likelihoods loaded: ',list(info['likelihood'].keys()))

#####LOAD THEORY#####
info['theory'] = TheoryHandler(info).theory_dict


#####LOADING SAMPLER#####
sampler = SamplingHandler(info)
info['sampler'] = sampler.sampling_dictionary

#####RUNNING######
sampler.run(info)

#MM: not really needed, but just to keep things clean
os.remove('theory_code/CalcDist.yaml')
