import sys
import numpy  as np
import pandas as pd

from copy import deepcopy
from scipy.interpolate import interp1d

class TheoryCalcs:

    def __init__(self,settings,params):

        self.zmin  = settings['zmin']
        self.zmax  = settings['zmax']
        self.Nz    = settings['Nz']
        self.zdrag = settings['zdrag']

        self.zcalc = np.linspace(self.zmin,self.zmax,self.Nz)

        camb_results = self.call_camb(params)

        self.DM = camb_results['DM/rd']
        self.DH = camb_results['DH/rd']
        self.DV = camb_results['DV/rd']
        #MM: this to be generalized
        self.eta_EM = self.get_eta(params['epsilon0_EM'])
        self.eta_GW = self.get_eta(params['epsilon0_GW'])
        self.DL_EM  = self.get_dL(camb_results['dA'],self.eta_EM)
        self.DL_GW  = self.get_dL(camb_results['dA'],self.eta_GW)
        ###########################
        self.mB = self.get_mB(self.DL_EM,params['MB'])
        self.rdrag = camb_results['rdrag']
        self.omegaL = camb_results['omegaL']


    def call_camb(self,params):

        camb_params = deepcopy(params)
        #MM: neutrinos to be added
        omnuh2 = 0.#camb_params['mnu']/neutrino_mass_fac*(camb_params['nnu']/3.0)**0.75
        camb_params['omch2'] = camb_params.pop('omegam')*(camb_params['H0']/100)**2.-camb_params['ombh2']-omnuh2
        del camb_params['MB']
        del camb_params['epsilon0_EM']
        del camb_params['epsilon0_GW']


        try:
            #MM: path to camb to be made customizable
            import camb
            pars = camb.set_params(**camb_params)
            results = camb.get_background(pars)

            Hz    = results.h_of_z
            comov = (1+self.zcalc)*results.angular_diameter_distance2(self.zmin,np.array([z for z in self.zcalc]))
            rdrag = results.sound_horizon(self.zdrag)

        except Exception as e:
            sys.exit('SOMETHING HORRIBLE HAPPENED!!\n {}'.format(e))


        theory = {'DM/rd': interp1d(self.zcalc,comov/rdrag),
                  'DH/rd': interp1d(self.zcalc,1/(Hz(self.zcalc)*rdrag)),
                  'DV/rd': interp1d(self.zcalc, (self.zcalc*comov**2/Hz(self.zcalc))**(1/3)/rdrag),
                  'dA': results.angular_diameter_distance,
                  'rdrag': rdrag,
                  'omegaL': results.get_Omega('de',z=0)}


        return theory
    def get_eta(self,epsilon):

        eta = interp1d(self.zcalc,(1+self.zcalc)**epsilon)

        return eta

    def get_dL(self,dA,eta):

        dL = interp1d(self.zcalc,eta(self.zcalc)*(1+self.zcalc)**2*dA(self.zcalc))

        return dL

    def get_mB(self,dL,MB):

        mB = interp1d(self.zcalc,5*np.log10(dL(self.zcalc))+MB+25,kind='linear')

        return mB
