from __future__ import division
import mvpa2.suite as mvpa
from joblib import dump
import numpy as np
import sys
from sklearn.cross_validation import KFold

subj = sys.argv[1]
subj = int(subj)

spenc_dir = '/home/mboos/SpeechEncoding/'

subj_preprocessed_path = 'PreProcessed/FG_subj{}.gzipped.hdf5'.format(subj)
s1ds = mvpa.h5load(spenc_dir + subj_preprocessed_path)

duration = np.array([902,882,876,976,924,878,1084,676])

# i did not kick out the first/last 4 samples per run yet
slice_nr_per_run = [dur/2 for dur in duration]

# use broadcasting to get indices to delete around the borders
idx_borders = np.cumsum(slice_nr_per_run[:-1])[:,np.newaxis] + \
              np.arange(-4,4)[np.newaxis,:]

fmri_data = np.delete(s1ds.samples, idx_borders, axis=0)

# and we're going to remove the last fmri slice
# since it does not correspond to a movie part anymore
fmri_data = fmri_data[:-1, :]

# shape of TR samples
fmri_data = fmri_data[3:]


voxel_kfold = KFold(fmri_data.shape[1], n_folds=10)

for i, (_, split) in enumerate(voxel_kfold):
    dump(fmri_data[:, split],
         '/home/data/scratch/mboos/prepro/fmri_subj_{}_split_{}.pkl'.format(subj, i),
         compress=3)
