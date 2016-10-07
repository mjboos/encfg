
# coding: utf-8

# In[1]:

import mvpa2.suite as mvpa
import numpy as np
import scipy as sp
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from pandas import read_csv



#mask_fname = os.path.join('/home','mboos','SpeechEncoding','temporal_lobe_mask_brain_subj' + str(subj) + 'bold.nii.gz')

#get openFMRI dataset handle
dhandle = mvpa.OpenFMRIDataset(datapath)
model = 1
task = 1

T3 = False
#get openFMRI dataset handle
dhandle = mvpa.OpenFMRIDataset(datapath)
model = 1
task = 1

datapath = os.path.join('/home','data','psyinf','forrest_gump','anondata')
#boldlist = sorted(glob.glob(os.path.join(datapath,'task002*')))
flavor = 'dico_bold7Tp1_to_subjbold7Tp1'

for subj in xrange(1,20):
    mask_fname = os.path.join('/home','mboos','SpeechEncoding','temporal_lobe_mask_brain_subj%02dbold.nii.gz' % subj)

    #load and save all datasets
    run_datasets = []
    for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
        run_ds = dhandle.get_bold_run_dataset(subj,task,run_id,chunks=run_id-1,mask=mask_fname,flavor=flavor)
        run_datasets.append(run_ds)
    s1ds = mvpa.vstack(run_datasets)
    mvpa.poly_detrend(s1ds,polyord=1,chunks_attr='chunks')
    mvpa.zscore(s1ds)
    s1ds.save(os.path.join('/home','mboos','SpeechEncoding','PreProcessed','FG_subj' + str(subj) + 'pp.gzipped.hdf5'),compression=9)

