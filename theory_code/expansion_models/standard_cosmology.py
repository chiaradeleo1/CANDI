import sys,os

import numpy  as np
import pandas as pd


class StandardExpansion:

    def __init__(self,call_name,settings):

        label = 'Standard'

        if call_name == label:
            print('')
            print('Selected cosmology: {}'.format(label))
            self.used = True

            self.zmin      = settings['zmin']
            self.zmax      = settings['zmax']
            self.Nz        = settings['Nz']

            self.zcalc = np.linspace(self.zmin,self.zmax,self.Nz)
        else:
            self.used = False


    def get_cosmology(self,params):

        unknown = []#Add check on CAMB params par for par in parameters.keys() if par not in self.params]

        if unknown != []:
            sys.exit('Error in {} cosmology code!\n Unknown parameters: {}'.format(label,unknown))

        #MM: path to camb to be made customizable
        import camb
        pars = camb.set_params(**params)
        results = camb.get_background(pars)

        Hz    = results.h_of_z
        comov = (1+self.zcalc)*results.angular_diameter_distance2(self.zmin,np.array([z for z in self.zcalc]))

        theory = {'H_Mpc': Hz,
                  'H_kmsMpc': results.hubble_parameter,
                  'comoving': results.comoving_radial_distance,
                  'rdrag': results.get_derived_params()['rdrag'],
                  'omegaL': results.get_Omega('de',z=0)}


        return theory

