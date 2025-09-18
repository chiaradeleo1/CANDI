import sys,os
import yaml
import numpy  as np
import pandas as pd

def nautilus_interface(info):

    #We thank Guadalupe Cañas-Herrera for the contribution to this script.

    from cobaya.model import get_model
    from scipy.stats  import norm
    from nautilus     import Prior
    from nautilus     import Sampler

    num_threads = info['sampler']['nautilus']['num_threads']

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)

    print('')
    print('RUNNING WITH NAUTILUS SAMPLER')
    print('')

    print('Loading model wrapper of Cobaya')
    model = get_model(info)
    print('model loaded')
    point = dict(zip(model.parameterization.sampled_params(),
                     model.prior.sample(ignore_external=True)[0]))
    logposterior = model.logposterior(point)

    print('Preparing the prior...')
    prior = Prior()

    for par in info['params'].keys():
        if (type(info['params'][par]) is dict) and ('prior' in info['params'][par].keys()):

            if len(info['params'][par]['prior'])==2:
                dist_prior = (info['params'][par]['prior']['min'],
                              info['params'][par]['prior']['max'])

            elif info['params'][par]['prior']['dist'] == 'norm':
                dist_prior = norm(loc=info['params'][par]['prior']['loc'],
                                  scale = info['params'][par]['prior']['scale'])

            prior.add_parameter(par, dist=dist_prior)

    print('Loaded prior into Nautilus with dimension',prior.dimensionality())
    print('Prior keys: ',prior.keys)

    derived_pars = [k for k in info['params'].keys() if type(info['params'][k]) == dict and 'prior' not in info['params'][k]]
    blob_vec     = [(par, float) for par in derived_pars]

    ## Likelihood
    def likelihood_nautilus(param_dict):

        derived_params = model.logposterior(param_dict).derived

        like_tuple    = [model.logposterior(param_dict).loglike]+[par for par in derived_params]
        full_tuple    = tuple(like_tuple)

        return full_tuple 


    print('Starting to sample with Nautilus...')
    nautilus_options = {k:v for k,v in info['sampler']['nautilus'].items() if k != 'num_threads'}
    if 'output' in info and info['output'] != '':
        nautilus_options['filepath'] = info['output']+'.hdf5'

    sampler = Sampler(prior,likelihood_nautilus,**nautilus_options,blobs_dtype=blob_vec)

    sampler.run(verbose=True)
    log_z = sampler.evidence()
    points, log_w, log_l, derived = sampler.posterior(equal_weight=True,return_blobs=True)
    derived_array = np.array([np.array(list(der)) for der in derived])

    nautilus_dict = {par: info['params'][par]['latex'] for par in prior.keys} | {par: info['params'][par]['latex'] for par in derived_pars}

    if 'output' in info and info['output'] != '':
        with open(info['output']+'.params.yaml', 'w') as outfile:
            yaml.dump(nautilus_dict, outfile, default_flow_style=False)

    results = pd.DataFrame(np.c_[points, derived_array, np.exp(log_w), -log_l],columns=list(nautilus_dict.keys())+['weight','minuslogpost'])
    results = results[['weight','minuslogpost']+list(nautilus_dict.keys())]

    if 'output' in info:
        results.to_csv(info['output']+'_chain.txt',sep='\t',header=False,index=False)


    print('NAUTILUS SAMPLING FINISHED')

    return results,nautilus_dict

