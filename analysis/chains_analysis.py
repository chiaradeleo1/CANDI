import sys,os

import numpy  as np
import pandas as pd

from copy    import deepcopy
from bios    import read
from getdist import plots,loadMCSamples,MCSamples

import pprint
#pp = pprint.PrettyPrinter(indent=4)


class Analyzer:

    def __init__(self,settings,fiducial):

        self.fiducial = fiducial.values()

        self.MH       = settings['Metropolis-Hastings']
        self.Nautilus = settings['Nautilus']
        #self.emcee    = settings['emcee']

    def print_dict(self,d,indent=4):
        #Makes nice print of a dictionary
        for k,v in d.items():
            print('')
            print('\x1b[1;31m'+k+'\x1b[0m')
            pprint.PrettyPrinter(indent=indent).pprint(v)

        return None

    def get_chain_info(self,info):
        #This functions reads in all the files associated with a given chain
        #It looks for different files depending on the sampler
        #MH:
        # - chain txt file for different number of chains
        # - .minimum file (not required)
        #Nautilus:
        # - _chain.txt file
        # - _params.npy file

        chain_info = deepcopy(info)

        if info['sampler'] == 'MH':
            #Read parameters
            chain_info['outyaml'] = read(info['path']+'.updated.yaml', file_type='yaml')
            
            columns = open(info['path']+'.1.txt').readline().rstrip().split()
            columns.pop(0)

            raw_chains = [pd.read_csv(info['path']+'.'+str(i+1)+'.txt',sep='\s+',
                                      skiprows=1,header=None,names=columns)
                          for i in range(info['Nchains'])]

            for i,chain in enumerate(raw_chains):
                chain['Chain number'] = i+1

            chain_info['Raw chains'] = pd.concat(raw_chains,ignore_index=True)


        elif info['sampler'] == 'Nautilus':

            chain_info['outyaml'] = read(info['path']+'.params.yaml', file_type='yaml')

            primary_pars = {par: chain_info['outyaml']['params'][par]['latex'] 
                            for par in chain_info['outyaml']['params']
                            if type(chain_info['outyaml']['params'][par]) == dict and 'prior' in chain_info['outyaml']['params'][par]}
            
            derived_pars = {par: chain_info['outyaml']['params'][par]['latex']
                            for par in chain_info['outyaml']['params']
                            if type(chain_info['outyaml']['params'][par]) == dict and 'prior' not in chain_info['outyaml']['params'][par]}

            pars = primary_pars | derived_pars

            raw_chains = pd.read_csv(info['path']+'_chain.txt',sep='\s+',
                                     header=None,names=['weight','minuslogpost']+list(pars.keys()))
            raw_chains['Chain number'] = 1

            chain_info['Nautilus pars'] = pars
            chain_info['Raw chains']    = raw_chains

        

        else:
            sys.exit('ERROR! Unknown sampler {}'.format(info['sampler']))

        return chain_info

    def analyze_chain(self,name,info):
        #This function analyzes the chains depending on sampler.
        #it returns
        # - getdist objects for contour plotting
        # - creates estimators for parameters values and errors (options TBA)
        # - runs CAMB with mean and best fit values (might break)
        # - outputs a chain_report dictionary

        chain_info   = self.get_chain_info(info)
        chain_report = deepcopy(chain_info)

        print('')
        print('\x1b[1;31m Analyzing {} \x1b[0m'.format(name))

        #Here creates MCsamples object from different sampler
        if info['sampler'] == 'MH':
            sample = loadMCSamples(info['path'], settings=self.MH)

            if info['Nchains']>1:
                print('R-1({}) with {:.0f}% of points ignored = {:.3f}'.format(name,100*self.MH['ignore_rows'],
                                                                           sample.getGelmanRubin()))
            else:
                print('Single chain, no R-1 computed. Trust Cobaya and hope for the best')

        elif info['sampler'] == 'Nautilus':
            sample    = MCSamples(samples=chain_info['Raw chains'][list(chain_info['Nautilus pars'].keys())].values,
                                  names=list(chain_info['Nautilus pars'].keys()),
                                  labels=list(chain_info['Nautilus pars'].values()),label=name)

            sample.root = info['path']
        else:
            sys.exit('ERROR! Unknown sampler {}'.format(info['sampler']))

       
        #Common analysis part
        chain_report['MCsamples'] = sample
        chain_report['bounds'] = sample.getTable(limit=1).tableTex()

        print(chain_report['bounds'])
        
        all_pars     = sample.getParamNames().list()
        labels       = sample.getParamNames().labels()
        primary_pars = sample.getParamNames().getRunningNames()
   
    
        means    = {par:val for par,val in zip(all_pars,sample.getMeans())}
        discard  = ['weight','minuslogpost','minuslogprior', 
                    'minuslogprior__0']
        try:
            best_fit = sample.getParamBestFitDict()
        except:
            print('no .minimum file available for {}. Switching to best sample'.format(name))
            try:
                best_fit = sample.getParamBestFitDict(best_sample=True)
            except:
                print('Best sample unavailable for some reason. Using means as best fit (DO NOT TRUST THIS!)')
                best_fit = deepcopy(means)

        try:
            best_fit['chi2'] = 2*best_fit['loglike']
        except:
            print('Best fit chi2 not available')
        chain_report['Estimators'] = pd.DataFrame.from_dict({par: [means[par],best_fit[par]] for par in all_pars if par not in discard})
        chain_report['Estimators']['Type']      = ['Mean','Best-Fit']
        chain_report['Estimators']['Cosmology'] = name
    
    
        chain_report['Sampled points']              = pd.DataFrame(sample.makeSingleSamples(),columns=all_pars)
        chain_report['Sampled points']['Cosmology'] = name

        covmat = pd.DataFrame(sample.getCovMat().matrix,columns=sample.getCovMat().paramNames,index=sample.getCovMat().paramNames)

        chain_report['covmat'] = covmat


        return chain_report


