import sys,os
import importlib.util
import inspect

import numpy  as np
import pandas as pd

from copy import deepcopy
from time import time

from scipy.interpolate import interp1d
from scipy.integrate   import trapezoid,quad

from theory_code.DDR_parametrizations import DDRCalcs

clight = 299792.458


class TheoryCalcs:

    def __init__(self,settings,cosmosets,SNmodel,fiducial,DDR=None,feedback=False,run_all=True):

        self.feedback = feedback
        
        #############################################
        #MM: this is a quite involved method to load
        #all modules contained in the expansion_models folder
        folder = 'theory_code/expansion_models'
        imported_classes = self.import_classes_from_folder(folder)

        cosmo_module = []
        if imported_classes:
            for class_name, class_obj in imported_classes.items():
                test = class_obj(cosmosets['cosmology'])
                if test.used:
                    cosmo_module.append(test)
                    if self.feedback:
                        print('')
                        print('Selected cosmology: {}'.format(class_name))

        if len(cosmo_module) > 1:
            sys.exit('Error in importing possible expansion modules (probably same label for multiple modules)')
        elif len(cosmo_module) == 0:
            sys.exit('UNKNOWN COSMOLOGY: {}'.format(cosmosets['cosmology']))
        else:
            cosmo_module = cosmo_module[0]
        ##############################################

        if not run_all:
            self.recognized_params = cosmo_module.recognized_params
            return

        ##################
        #General settings#
        ##################
        self.settings  = settings
        self.zmin      = settings['zmin']
        self.zmax      = settings['zmax']
        self.Nz        = settings['Nz']

        self.zcalc = np.linspace(self.zmin,self.zmax,self.Nz)

        ############################
        #Getting baseline cosmology#
        ############################

        if 'rd' in cosmosets['parameters']:
            true_rdrag  = cosmosets['parameters'].pop('rd')
        else:
            true_rdrag = None

        try:
            tini = time()
            cosmo_results = cosmo_module.get_cosmology(cosmosets['parameters'],settings)
            tend = time()
            if self.feedback:
                print('Basic cosmology done in {:.2f} s'.format(tend-tini))
        except Exception as e:
            sys.exit('COSMOLOGY CALCULATIONS FAILED!!\n {}'.format(e))


        #Changing theory dict elements into class attributes
        for k, v in cosmo_results.items():
            setattr(self,k,v)


        ###############
        #Computing DDR#
        ###############
        #MM: DDR is only possible in dL, should we change to allow
        #for dA to be broken as well? It might be needed for some 
        #weird torsion thingy

        if DDR != None:
            try:
                self.eta_EM,self.eta_GW = self.get_parameterized_DDR(DDR)
            except Exception as e:
                sys.exit('DDR FAILED!\n {}'.format(e))
        else:
            self.eta_EM = lambda x: 1.
            self.eta_GW = lambda x: 1.

        self.dA     = interp1d(self.zcalc,self.comoving(self.zcalc)/(1+self.zcalc))
        self.DL_EM  = self.get_dL(self.dA,self.eta_EM)
        self.DL_GW  = self.get_dL(self.dA,self.eta_GW)

        if true_rdrag != None:
            self.rdrag  = true_rdrag

        ##########################
        #Computing BAO quantities#
        ##########################
        self.get_BAO_observables(fiducial)


        ########################
        #Computing SN magnitude#
        ########################
        self.mB = self.get_mB(self.DL_EM,SNmodel)

    def get_parameterized_DDR(self,DDR):

        from theory_code.DDR_parametrizations import DDRCalcs

        if self.feedback:
            print('Computing DDR functions...')
        tini = time()
        ddr_results  = DDRCalcs(DDR,self.zcalc)
        eta_EM = ddr_results.eta_EM
        eta_GW = ddr_results.eta_GW
        tend = time()
        if self.feedback:
            print('DDR done in {:.2f} s'.format(tend-tini))

        return eta_EM,eta_GW


    def get_BAO_observables(self,fiducial):

        self.DM = interp1d(self.zcalc,self.comoving(self.zcalc))
        self.DH = interp1d(self.zcalc,1/(self.H_Mpc(self.zcalc)))
        self.DV = interp1d(self.zcalc, (self.zcalc*self.comoving(self.zcalc)**2/self.H_Mpc(self.zcalc))**(1/3))
        self.dA = interp1d(self.zcalc,self.comoving(self.zcalc)/(1+self.zcalc))

        if type(fiducial) == str:
            fidtable = pd.read_csv(fiducial,header=0,sep='\t')
            fidcosmo = {'DV_rd': interp1d(fidtable['z'],fidtable['DV_rd']),
                        'DH_DM': interp1d(fidtable['z'],fidtable['DH_DM'])}

            self.alpha_iso = lambda x: (self.DV(x)/self.rdrag)/fidcosmo['DV_rd'](x)
            self.alpha_AP  = lambda x: (self.DH(x)/self.DM(x))/fidcosmo['DH_DM'](x)

        elif type(fiducial) == dict:
            from theory_code.expansion_models.standard_cosmology import StandardExpansion
            fidmodule = StandardExpansion('Standard')
            fidcosmo = fidmodule.get_cosmology(fiducial,self.settings)
            fidcosmo['DM'] = interp1d(self.zcalc,fidcosmo['comoving'](self.zcalc))
            fidcosmo['DH'] = interp1d(self.zcalc,1/(fidcosmo['H_Mpc'](self.zcalc)))
            fidcosmo['DV'] = interp1d(self.zcalc, (self.zcalc*fidcosmo['comoving'](self.zcalc)**2/fidcosmo['H_Mpc'](self.zcalc))**(1/3))

            self.alpha_iso = lambda x: (self.DV(x)/self.rdrag)/(fidcosmo['DV'](x)/fidcosmo['rdrag'])
            self.alpha_AP  = lambda x: (self.DH(x)/self.DM(x))/(fidcosmo['DH'](x)/fidcosmo['DM'](x))

        self.DV_rd     = lambda x: self.DV(x)/self.rdrag
        self.DM_DH     = lambda x: self.DM(x)/self.DH(x)
        self.DM_rd     = lambda x: self.DM(x)/self.rdrag
        self.DH_rd     = lambda x: self.DH(x)/self.rdrag

        return None

    def get_dL(self,dA,eta):

        dL = interp1d(self.zcalc,eta(self.zcalc)*(1+self.zcalc)**2*dA(self.zcalc))

        return dL

    def get_mB(self,dL,MBpars):

        if MBpars['model'] == 'constant':
            MB = lambda x: MBpars['MB']
        else:
            sys.exit('UNKNOWN SN MODEL: {}'.format(MBpars['model']))

        #MM: ugly fix to avoid log10(0). To be fixed
        mB = interp1d(self.zcalc[1:],5*np.log10(dL(self.zcalc[1:]))+MB(self.zcalc[1:])+25,kind='linear')

        return mB

    def import_classes_from_folder(self,folder_path):
        #MM: this was done by Gemini
        #Thank you, our Lord and Saviour!
    
        if not os.path.isdir(folder_path):
            print(f"Error: Folder '{folder_path}' not found.")
            return {}

        classes = {}
        sys.path.insert(0, folder_path)

        for filename in os.listdir(folder_path):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]  # Remove .py extension
                file_path = os.path.join(folder_path, filename)

                try:
                    # Create a module spec
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None:
                        print(f"Warning: Could not create spec for {file_path}")
                        continue

                    # Load the module
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    # Inspect the module for classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Ensure the class is defined in the current module, not an imported one
                        if obj.__module__ == module_name:
                            classes[name] = obj
                            #print(f"Imported class: {name} from {filename}")

                except Exception as e:
                    print(f"Error importing module {filename}: {e}")

        # Clean up sys.path
        if folder_path in sys.path:
            sys.path.remove(folder_path)

        return classes
    
    
