import sys,os

import numpy  as np
import pandas as pd

import camb

class StandardExpansion:

    def __init__(self,call_name):

        self.label = 'Standard'
        #MM: this needs to be done better!!!
        #How the fuck do I get a list of CAMB parameters??

        self.recognized_params = {'H0': None,
                                  'omch2': None,
                                  'ombh2': None,
                                  'omk': None,
                                  'w': None,
                                  'wa': None,
                                  'omnuh2': None,
                                  'nnu': None}

        self.derived_params = ['rdrag','omegaL']

        if call_name == self.label:
            self.used = True

        else:
            self.used = False


    def get_cosmology(self,params,settings):

        self.zmin      = settings['zmin']
        self.zmax      = settings['zmax']
        self.Nz        = settings['Nz']

        unknown = [par for par in params.keys() if par not in self.recognized_params]

        if unknown != []:
            sys.exit('Error in {} cosmology code!\n Unknown parameters: {}'.format(self.label,unknown))

        pars = camb.set_params(**params)
        results = camb.get_background(pars)

        Hz    = results.h_of_z

        theory = {'H_Mpc': Hz,
                  'H_kmsMpc': results.hubble_parameter,
                  'comoving': results.comoving_radial_distance,
                  'rdrag': results.get_derived_params()['rdrag'],
                  'omegaL': results.get_Omega('de',z=0)}


        return theory

