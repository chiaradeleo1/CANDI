from cobaya.theory import Theory
import numpy as np
from scipy.interpolate import interp1d

class DDRtheory(Theory):

    def __init__(self, params = None, case = None, z_camb = None,  **kwargs):
        super().__init__(**kwargs)
        self.params = params if params is not None else {}
        self.case = case if case is not None else 0
        self.z_camb = z_camb if z_camb is not None else np.linspace(0.001, 3.0, 100)
        
        
        

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        camb_requirements = {'H0': None, 
                             'omegam': None,
                             'angular_diameter_distance_2': {'z_pairs': [(0., z) for z in self.z_camb]},
                            'Hubble': {'z': self.z_camb, 'units': '1/Mpc'} }
        
        params_requirements = {'rdrag': None,
                               'epsilon0': None }
        requirements = {**camb_requirements, **params_requirements}
        return requirements
    
    def calculate(self):
        dA = self.provider.get_angular_diameter_distance_2([(0.,z) for z in self.z_camb])
        comoving_dist = (1+self.z_camb)* dA
        Hz = self.provider.get_Hubble(self.z_camb, units='1/Mpc')
        rdrag = self.provider.get_param('rdrag')
        DM = interp1d(self.z_camb,comoving_dist/rdrag) 
        DH = interp1d(self.z_camb,1/(Hz*rdrag))
        DV = interp1d(self.z_camb, (self.z_camb*comoving_dist**2/Hz)**(1/3)/rdrag)
        dL = (1+self.z_camb)**(2+self.provider.get_param('epsilon0'))*dA
        mu_z = 5*np.log10(dL) + 5# + self.provider.get_param('M0')
        self.rdrag=rdrag
        state['DM'] = DM
        state['DH'] = DH
        state['DV'] = DV
        state['mu_z'] = mu_z

    def get_theory(self):
        """
        Returns a dictionary with the interpolated functions for DM, DH, DV, mu_z.
        """
        return {
            'DM': self.calculate(state={}).get('DM'),
            'DH': self.calculate_(state={}).get('DH'),
            'DV': self.calculate_(state={}).get('DV'),
            'mu_z':  self.calculate_(state={}).get('mu_z'),
            'rdrag': self.rdrag,
            'epsilon0': self.provider.get_param('epsilon0'),
        }


