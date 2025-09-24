import sys,os
import numpy   as np
import pandas  as pd
import seaborn as sb

from copy      import deepcopy
from itertools import product

from testing.testing_functions import *

import gdown


threshold             = 1     #Percentage of difference from reference within which we say we are fine
get_plot              = False #Set to True if you want a plot
observables_reference = pd.read_csv('testing/reference_results.txt',sep='\t',header=0)
test_likelihood_value = False


print('')
print('TESTING THEORY CODE')
print('')
obs_test = test_observables(observables_reference,threshold,get_plot)

print('')
print('TESTING RUNNING SETTINGS')

#data_path = 'https://drive.google.com/file/d/1_Q4IKpP5MusLz7SC2rvX_7_wOjgvDD09/view?usp=sharing'
#data_file = 'testing_data.tar.gz'
#gdown.download(data_path,data_file,quiet=False)

#os.system('tar -xzf {}'.format(data_file))

print('')
print('Standard cosmology')
main_folder = 'settings/standard/'
test        = test_settings(main_folder,test_likelihood_value)
print('')
print('DCDM cosmology')
main_folder = 'settings/DCDM/'
test        = test_settings(main_folder,test_likelihood_value)
print('')
print('DDR cosmology')
main_folder = 'settings/DDR/'
test        = test_settings(main_folder,test_likelihood_value)

