from cobaya.theory import Theory
import numpy as np
from scipy.interpolate import interp1d

class SNtheory(Theory):

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
        camb_requirements = { 'H0': None, 
                         'omegam': None,
                         #'M0' : None,
                         'angular_diameter_distance_2': { 'z_pairs': [(0., z) for z in self.z_camb]}}
        derived_requirements = {
            'epsilon0': None  # Explicitly specify this as a derived requirement
        }
        
        # Combine the requirements dictionaries
        requirements = {**camb_requirements, **derived_requirements}
        
        return requirements
    
    def get_theory(self):
        dL = (1+self.z_camb)**(2+self.provider.get_param('epsilon0'))*self.provider.get_angular_diameter_distance_2([(0.,z) for z in self.z_camb])
        mu_z = 5*np.log10(dL) + 5# + self.provider.get_param('M0')
        
        return {'mu_z' : mu_z}

