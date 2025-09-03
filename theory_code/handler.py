import sys,os

import yaml

from bios import read
from copy import deepcopy

from theory_code.cobaya_theory_wrapper import CalcDist
from theory_code.distance_theory       import TheoryCalcs

class TheoryHandler:

    def __init__(self,info):

        self.theory_dict = {'CalcDist': {'external': CalcDist,
                                         'cosmology': info['cosmology'],
                                         'fiducial': info['fiducial_path']}}

        if 'DDR_options' in info:
            self.theory_dict['CalcDist']['DDR_options'] = info['DDR_options']

        self.theory_dict['CalcDist']['derived_pars'] = self.include_theory_params(info)


    def include_theory_params(self,info):

        #MM: this is a workaround to make Cobaya aware of the theory
        #parameters defined in the model module
        #PROBABLY CAN BE DONE BETTER

        basic_params = read('theory_code/basic_parameters.yaml')

        pre_init = TheoryCalcs(None,{'cosmology': info['cosmology']},None,None,run_all=False)

        full_params = deepcopy(basic_params)
        for par,val in pre_init.recognized_params.items():
             full_params['params'][par] = val

        #Adding derived parameters
        for par in pre_init.derived_params:
            full_params['params'][par] = {'derived': True}

        derived_pars = pre_init.derived_params

        f = open('theory_code/CalcDist.yaml', 'w+')
        yaml.dump(full_params,f)

        return derived_pars
