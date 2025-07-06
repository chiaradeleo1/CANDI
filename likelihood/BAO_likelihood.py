import sys
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

class BAOLike(Likelihood):
    
    def initialize(self):

        if self.data_format == 'DESI':
            data_table = pd.read_csv(self.BAO_data_path+'.txt',sep='\t',header=0)

            #Single observable points
            single_obs = data_table[data_table.isnull().any(axis=1)]

            data_table = data_table.dropna().reset_index(drop=True)
            self.z_vec = data_table['z'].values

            if self.observables == 'distances':
                distance_covs = self.get_covmats(data_table)
                self.single_data = single_obs[['z','DV_rd','DV_rd_err']]
                self.datavectors = [np.array([row['DV_rd'],row['DM_DH']]) for ind,row in data_table.iterrows()]
                self.inv_covmats = [pd.DataFrame(np.linalg.inv(covmat.values),
                                                 columns=covmat.columns,index=covmat.index) for covmat in distance_covs]
            elif self.observables == 'alphas':
                self.single_data = single_obs[['z','alpha_iso','alpha_iso_err']]
                self.datavectors = [np.array([row['alpha_iso'],row['alpha_AP']]) for ind,row in data_table.iterrows()]
                alpha_covs = self.transform_covmat(data_table)
                self.inv_covmats = [pd.DataFrame(np.linalg.inv(covmat.values),
                                                 columns=covmat.columns,index=covmat.index) for covmat in alpha_covs]
        elif self.data_format == 'SKA':
            data_table = pd.read_csv(self.BAO_data_path+'.txt',sep='\t',header=0)
            self.z_vec = data_table['z'].values
            self.datavectors = [np.array([row['DM_rd'],row['DH_rd']]) for ind,row in data_table.iterrows()]
            self.inv_covmats = []
            for ind,row in data_table.iterrows():

                cov = pd.DataFrame([[row['DM_rd_err']**2,row['r_MH']*row['DM_rd_err']*row['DH_rd_err']],
                                    [row['r_MH']*row['DM_rd_err']*row['DH_rd_err'],row['DH_rd_err']**2]],
                                    columns=['DM_rd_{}'.format(ind),'DH_rd_{}'.format(ind)],
                                    index=['DM_rd_{}'.format(ind),'DH_rd_{}'.format(ind)])

                self.inv_covmats.append(pd.DataFrame(np.linalg.inv(cov.values),columns=cov.columns,index=cov.index))

        else:
            sys.exit('Unknown data format: {}'.format(self.data_format))

        zmax = 4.
        self.z_camb = np.linspace(0.001, zmax, 10000)

    def transform_covmat(self,data):
        #MM to be changed

        z_covs = []
        for ind,row in data.iterrows():
            alpha_cols = ['alpha_iso_{}'.format(ind),'alpha_AP_{}'.format(ind)]

            precov   = pd.DataFrame([[row['DM_rd_err']**2,row['r_MH']*row['DM_rd_err']*row['DH_rd_err']],
                                     [row['r_MH']*row['DM_rd_err']*row['DH_rd_err'],row['DH_rd_err']**2]],
                                    columns=['DM_rd_{}'.format(ind),'DH_rd_{}'.format(ind)],
                                    index=['DM_rd_{}'.format(ind),'DH_rd_{}'.format(ind)])

            x1   = row['DM_rd']
            x2   = row['DH_rd']
            x1_F = row['DM_rd_fid']
            x2_F = row['DH_rd_fid']

            jacobian = [[(1/3)*(x1**2*x2)**(-2/3)*2*x1*x2/(x1_F**2*x2_F)**(1/3),(1/3)*(x1**2*x2)**(-2/3)*x1**2/(x1_F**2*x2_F)**(1/3)],
                        [-(x1_F/x2_F)*(x2/x1**2),(x1_F/x2_F)*1/x1]]

            z_covs.append(pd.DataFrame(jacobian @ precov.values @ np.transpose(jacobian),
                                       columns = alpha_cols,index=alpha_cols))


        return z_covs

    def get_covmats(self,data):

        z_covs = [pd.DataFrame([[row['DV_rd_err']**2,row['r_VMH']*row['DV_rd_err']*row['DM_DH_err']],
                                [row['r_VMH']*row['DV_rd_err']*row['DM_DH_err'],row['DM_DH_err']**2]],
                  columns=['DV_rd_{}'.format(ind),'DM_DH_{}'.format(ind)],
                  index=['DV_rd_{}'.format(ind),'DM_DH_{}'.format(ind)]) for ind,row in data.iterrows()]

        return z_covs

    def get_requirements(self):
        # Requirements are the output of the theory code that you are using
        requirements = {'DM_DH': None,
                        'DV_rd': None,
                        'DM_rd': None,
                        'DH_rd': None,
                        #'DV': None,
                        'alpha_iso': None,
                        'alpha_AP': None,
                        'rdrag': None}

        return requirements
    
    def logp(self, **params_values): 

        if self.data_format == 'DESI':

            if self.observables == 'distances':
                obs = 'DV_rd'
            elif self.observables == 'alphas':
                obs = 'alpha_iso'
            chi2_single = sum([(row[obs]-self.provider.get_result(obs)(row['z']))**2/row[obs+'_err']**2 for ind,row in self.single_data.iterrows()])

        else:
            chi2_single = 0.

        chi2_vec = []
        for z,data,invcov in zip(self.z_vec,self.datavectors,self.inv_covmats):
            obs = ['_'.join(item for item in col.split('_')[:2]) for col in invcov.columns]
            theoryvec = np.array([self.provider.get_result(o)(z) for o in obs])
            chi2_vec.append(np.dot(theoryvec-data,np.dot(invcov,theoryvec-data)))

        chi2_binned = sum(chi2_vec)

        chi2 = chi2_binned+chi2_single

        loglike = -0.5*chi2

        return loglike
