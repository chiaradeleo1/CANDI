import sys,os
import subprocess
import numpy   as np
import pandas  as pd
import seaborn as sb

from bios import read
from copy import deepcopy
from yaml import dump

from contextlib import redirect_stdout

from theory_code.distance_theory import TheoryCalcs

from cobaya.run import run

from likelihood.handler  import LikelihoodHandler
from samplers.handler    import SamplingHandler
from theory_code.handler import TheoryHandler

RED   = "\033[1;31m"
GREEN = "\033[0;32m"

def test_settings(main_folder,test_likelihood_value):

    for sets in os.listdir(main_folder):
        settings     = main_folder+'/'+sets
        std_settings = read(settings)
        fiducial_params = {par: val['ref']['loc'] for par,val in std_settings['params'].items() if (type(val)==dict) and ('prior' in val)}
        
        test_settings = deepcopy(std_settings)
        test_settings['sampler']['name']    = 'evaluate'
        test_settings['sampler']['options'] = {'override': fiducial_params}

        with open('testing/'+sets, 'w') as yaml_file:
            dump(test_settings, yaml_file, default_flow_style=False)

        try:
            command_to_run = ['python', 'runner.py', 'testing/'+sets]
            result = subprocess.run(command_to_run, capture_output=True, text=True)

            if result.returncode == 0:
                print('\033[0;32m' + f'{main_folder}{sets} running!' + '\033[0m')
                # You can even access the standard output if you need it later
                script_output = result.stdout

                if test_likelihood_value:
                    sys.exit('Likelihood value test not available yet')

            else:
                error_message = result.stderr if result.stderr else result.stdout

                print('\033[1;31m' + f'{main_folder}{sets} failed!' + '\033[0m')
                print("--- Error Captured ---")
                if error_message.strip():
                    log_filename = "testing/"+sets[:-5]+'.err'

                    print('\033[1;31m' + f'Error saved to {log_filename}' + '\033[0m')

                    with open(log_filename, 'w') as log_file:
                        log_file.write(error_message.strip())
                else:
                    print("Script failed with a non-zero exit code but produced no output.")

        except Exception as e:
           print(f"An unexpected error occurred in the test script itself: {e}")


        os.remove('testing/'+sets)

    return None

def test_observables(reference,threshold,save_comparison):

    settings = {'zmin': 0.001,
                'zmax': 5.,
                'Nz': 10000}

    standard_params = {'cosmology': 'Standard',
                       'parameters':{'H0': 67.36,
                                     'omch2': 0.1200,
                                     'ombh2': 0.02237,
                                     'omk': 0.,
                                     'omnuh2': 0.0006442,
                                     'nnu': 3.}}

    custom_params = {'cosmology': 'Custom',
                     'parameters': {'H0': 67.36,
                                    'omegam': 0.3153,
                                    'Delta': 0.1,
                                    'Gamma': 0.,
                                    'ombh2': 0.02237}}

    fiducial = {'H0': 67.36,
                'omch2': 0.1200,
                'ombh2': 0.02237,
                'omk': 0.,
                'omnuh2': 0.0006442,
                'nnu': 3.}


    SNmodel = {'model': 'constant',
               'MB': -19.2435}

    standard_results = TheoryCalcs(settings,standard_params,SNmodel,fiducial,feedback=True)
    custom_results   = TheoryCalcs(settings,custom_params,SNmodel,fiducial,feedback=True)

    zplot = np.linspace(0.1,3,100)

    std_df = pd.DataFrame({'z': zplot,
                           r'$d_C(z)$': standard_results.comoving(zplot),
                           r'$d_L(z)$': standard_results.DL_EM(zplot),
                           r'$d_A(z)$': standard_results.dA(zplot),
                           r'$H(z)$': standard_results.H_kmsMpc(zplot),
                           'Cosmology': 'Standard'})

    std_melt_df = pd.melt(std_df,id_vars=['z','Cosmology'],value_vars=[r'$d_C(z)$',r'$d_L(z)$',r'$d_A(z)$'],
                          var_name='Function',value_name='value')

    cus_df = pd.DataFrame({'z': zplot,
                           r'$d_C(z)$': custom_results.comoving(zplot),
                           r'$d_L(z)$': custom_results.DL_EM(zplot),
                           r'$d_A(z)$': custom_results.dA(zplot),
                           r'$H(z)$': custom_results.H_kmsMpc(zplot),
                           'Cosmology': 'Custom'})

    allres = pd.concat([std_df,cus_df],ignore_index=True)

    reldiff = pd.DataFrame({'z': allres['z'],
                            'Cosmology': allres['Cosmology']})

    for col in allres.columns:
        if col not in ['z','Cosmology']:
            reldiff[col] = 100*(allres[col]-reference[col])/reference[col]


    melt_reldiff = pd.melt(reldiff,id_vars=['z','Cosmology'],value_vars=[r'$d_C(z)$',r'$d_L(z)$',r'$d_A(z)$'],
                           var_name='Function',value_name='value')

    issues = melt_reldiff[abs(melt_reldiff['value'])>threshold]
    if len(issues) == 0:
        print('\033[0;32m'+'All differences with reference within {}%'.format(threshold)+'\033[0m')
    else:
        print('\033[1;31m'+'Mismatch with reference observables!!''\033[1m')
        print(issues)

    if save_comparison:
        import matplotlib
        import matplotlib.pyplot as plt

        from matplotlib import rc

        rc('text', usetex=True)
        rc('font', family='serif')
        matplotlib.rcParams.update({'font.size': 18})

        plt.figure()
        sb.lineplot(melt_reldiff,x='z',y='value',hue='Function',style='Cosmology')
        plt.axhline(y=1,ls=':',color='black')
        plt.axhline(y=-1,ls=':',color='black')
        plt.ylim([-3,3])
        plt.xlabel(r'$z$')
        plt.ylabel(r'Relative difference [\%]')
        plt.legend(ncols=2,frameon=False,loc='best')
        plt.savefig('testing/test_results.pdf',dpi=500)

    return None
