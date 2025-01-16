import sys
import numpy  as np
import pandas as pd

from copy import deepcopy
from scipy.interpolate import interp1d

from theory_code.DDR_theory import DDRCalcs

class TheoryCalcs:

    def __init__(self,settings,params):
        
        ###################
        self.zmin      = settings['zmin']
        self.zmax      = settings['zmax']
        self.Nz        = settings['Nz']
        self.zdrag     = settings['zdrag']
        self.DDR_model = settings['DDR_model']
        self.BBN       = settings['BBN']

        
        self.zcalc = np.linspace(self.zmin,self.zmax,self.Nz)

        try:
            camb_results = self.call_camb(params)
            ddr_results  = DDRCalcs(self.DDR_model,params,self.zcalc)
        except Exception as e:
            sys.exit('SOMETHING HORRIBLE HAPPENED!!\n {}'.format(e))


        self.DM = camb_results['DM/rd']
        self.DH = camb_results['DH/rd']
        self.DV = camb_results['DV/rd']
        #MM: this to be generalized
        self.eta_EM = ddr_results.eta_EM 
        self.eta_GW = ddr_results.eta_GW
        self.DL_EM  = self.get_dL(camb_results['dA'],self.eta_EM)
        self.DL_GW  = self.get_dL(camb_results['dA'],self.eta_GW)
        ###########################
        self.mB = self.get_mB(self.DL_EM,params['MB'])
        self.rdrag = camb_results['rdrag']
        self.omegaL = camb_results['omegaL']
        self.Hz = camb_results['Hz']
        self.dA = camb_results['dA']


    
    def call_camb(self,params):

        camb_params = deepcopy(params)
        #MM: neutrinos to be added
        omnuh2 = 0.#camb_params['mnu']/neutrino_mass_fac*(camb_params['nnu']/3.0)**0.75
        camb_params['omch2'] = camb_params.pop('omegam')*(camb_params['H0']/100)**2.-camb_params['ombh2']-omnuh2
        del camb_params['MB']
        del camb_params['epsilon0_EM']
        del camb_params['epsilon0_GW']
        del camb_params['a_EM']
        del camb_params['a_GW']
        del camb_params['n_EM']
        del camb_params['n_GW']
        del camb_params['rd']


        #MM: path to camb to be made customizable
        import camb
        pars = camb.set_params(**camb_params)
        results = camb.get_background(pars)

        Hz    = results.h_of_z
        comov = (1+self.zcalc)*results.angular_diameter_distance2(self.zmin,np.array([z for z in self.zcalc]))
        
        if self.BBN == True:
            rdrag = results.sound_horizon(self.zdrag)
        else:
            rdrag = params['rd']
        

        theory = {'DM/rd': interp1d(self.zcalc,comov/rdrag),
                  'DH/rd': interp1d(self.zcalc,1/(Hz(self.zcalc)*rdrag)),
                  'DV/rd': interp1d(self.zcalc, (self.zcalc*comov**2/Hz(self.zcalc))**(1/3)/rdrag),
                  'dA': results.angular_diameter_distance,
                  'rdrag': rdrag,
                  'omegaL': results.get_Omega('de',z=0)}

        theory['Hz'] = Hz
        return theory

    def get_dL(self,dA,eta):

        dL = interp1d(self.zcalc,eta(self.zcalc)*(1+self.zcalc)**2*dA(self.zcalc))

        return dL

    def get_mB(self,dL,MB):

        mB = interp1d(self.zcalc,5*np.log10(dL(self.zcalc))+MB+25,kind='linear')

        return mB
    
    
