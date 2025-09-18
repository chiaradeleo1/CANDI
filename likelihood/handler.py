import sys,os

import pprint
import time
import os
pp = pprint.PrettyPrinter(indent=4)


class LikelihoodHandler:

    def __init__(self,info):

        self.info = info

        self.like_dict = {}

        if 'BAO_data' in info and info['BAO_data'] != None:
            print('')
            print('LOADING BAO DATA')
            print('')
            pp.pprint(info['BAO_data'])
            self.like_dict['BAOlike'] = self.setup_BAO(info['BAO_data'])

        if 'SN_data' in info and info['SN_data'] != None:
            print('')
            print('LOADING SN DATA')
            print('')
            pp.pprint(info['SN_data'])
            self.like_dict['SNlike'] = self.setup_SN(info['SN_data'])

        if 'GW_data' in info and info['GW_data'] != None:
            print('')
            print('LOADING GW DATA')
            print('')
            pp.pprint(info['GW_data'])
            self.like_dict['GWlike'] = self.setup_GW(info['GW_data'])


    def setup_BAO(self,BAOinfo):

        from likelihood.BAO_likelihood import BAOLike

        bao_dict = {'external': BAOLike,
                    'BAO_data_path': BAOinfo['path'],
                    'data_format': BAOinfo['data_format']}

        if 'observables' in BAOinfo:
            bao_dict['observables'] = BAOinfo['observables']

        return bao_dict

    def setup_SN(self,SNinfo):

        from likelihood.SN_likelihood import SNLike

        if SNinfo['calibration'] == 'SH0ES':
            print('Using SH0ES calibration')
            calibration = 'SH0ES'
        elif SNinfo['calibration'] == 'Marginalized':
            calibration = 'Marginalized'
            print('')
            print('Using the SN likelihood analytically marginalized for H0 and MB.')
            print('WARNING! If you have these parameters free to vary, you will get no information coming from SN')
            print('')
#            info['params']['H0'] = 73.4
#            info['params']['MB'] = -19.2435
        else:
            calibration = None
            print('')
            #if 'dist' in self.info['params']['MB']['prior'].keys() and self.info['params']['MB']['prior']['dist'] == 'norm':
            #    print('Using the SN likelihood with a prior gaussian prior on MB')
            #else:
            #    print('Using the SN likelihood with a flat prior on MB')


        sn_dict =  {'external': SNLike,
                    'SN_data_path': SNinfo['path'],
                    'use_Pantheon': SNinfo['use_Pantheon'],
                    'calibration': calibration}

        return sn_dict

    def setup_GW(self,GWinfo):

        from likelihood.GW_likelihood import GWLike

        #MM: if we manage to integrate Simone's code, 
        #this is where we should have the switch

        gw_dict = {'external': GWLike,
                   'GW_data_path': GWinfo['path']}

        return gw_dict





