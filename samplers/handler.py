import sys,os



class SamplingHandler:

    def __init__(self,info):

        if info['sampler']['name'] in ['mcmc','minimize','evaluate']:
            print('Running with Metropolis-Hastings')
            self.sampling_dictionary,self.run = self.cobaya_setup(info['sampler'])

        elif info['sampler']['name'] == 'nautilus':
            print('Running with Nautilus')
            self.sampling_dictionary,self.run = self.nautilus_setup(info['sampler'])

        else:
            sys.exit('Unknown sampler: {}'.format(info['sampler']))


    def cobaya_setup(self,samp_info):

        from cobaya.run import run

        samp_dict        = {info['sampler']['name']: samp_info['options']}
        running_function = run

        return samp_dict,running_function

    def nautilus_setup(self,samp_info):

        from samplers.nautilus import nautilus_interface

        if samp_info['options'] == 'poor':
            sets = {'num_threads': 1,
                    'pool': 1,
                    'n_live': 500,
                    'n_batch': 64,
                    'n_networks': 2}
        elif samp_info['options'] == 'good':
            sets = {'num_threads': 1,
                    'pool': 1,
                    'n_live': 4000,
                    'n_batch': 512,
                    'n_networks': 16}

        samp_dict        = {'nautilus': sets}
        running_function = nautilus_interface

        return samp_dict,running_function
