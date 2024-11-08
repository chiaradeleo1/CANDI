import sys
import numpy  as np
import pandas as pd

from copy import deepcopy
from scipy.interpolate import interp1d

from theory_code.DDR_theory import DDRCalcs

class TheoryCalcs:

    def __init__(self,settings,params):
        ###########CONSTANT
        self.G = 6.67430e-11  
        solar_mass = 1.9885e30  
        self.c = 2.99 * 1e8
        self.Mpc_to_m = 3.086 *1e22
        ###################
        self.zmin      = settings['zmin']
        self.zmax      = settings['zmax']
        self.Nz        = settings['Nz']
        self.zdrag     = settings['zdrag']
        self.DDR_model = settings['DDR_model']
        self.m1        = settings['m1'] * solar_mass
        self.m2        = settings['m2'] * solar_mass
        

        self.Mchirp = self.get_Mchirp()

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
        self.h_GW = self.get_h_GW(self.DL_GW)

    def get_Mchirp(self):
        
        Mchirp = (self.m1 * self.m2)**(3/5) / (self.m1 + self.m2)**(1/5)

        return Mchirp
    
    def call_camb(self,params):

        camb_params = deepcopy(params)
        #MM: neutrinos to be added
        omnuh2 = 0.#camb_params['mnu']/neutrino_mass_fac*(camb_params['nnu']/3.0)**0.75
        camb_params['omch2'] = camb_params.pop('omegam')*(camb_params['H0']/100)**2.-camb_params['ombh2']-omnuh2
        del camb_params['MB']
        del camb_params['epsilon0_EM']
        del camb_params['epsilon0_GW']


        #MM: path to camb to be made customizable
        import camb
        pars = camb.set_params(**camb_params)
        results = camb.get_background(pars)

        Hz    = results.h_of_z
        comov = (1+self.zcalc)*results.angular_diameter_distance2(self.zmin,np.array([z for z in self.zcalc]))
        rdrag = results.sound_horizon(self.zdrag)

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
    
    def get_h_GW(self,dL):
        
        h_GW = interp1d(self.zcalc,4*(self.G*self.get_Mchirp())**(5/3)/(self.c**4*dL(self.zcalc)*self.Mpc_to_m),kind='linear')
        
        return h_GW
