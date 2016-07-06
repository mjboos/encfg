from __future__ import division
import mvpa2.suite as mvpa
from joblib import load, dump
import numpy as np
import sys
import h5py

subj = sys.argv[1]
subj = int(subj)

spenc_dir = '/home/mboos/SpeechEncoding/'

subj_preprocessed_path = 'PreProcessed/FG_subj{}pp.gzipped.hdf5'.format(subj)
s1ds = mvpa.h5load(spenc_dir + subj_preprocessed_path)

duration = np.array([902,882,876,976,924,878,1084,676])

# i did not kick out the first/last 4 samples per run yet
slice_nr_per_run = [dur/2 for dur in duration]

# use broadcasting to get indices to delete around the borders
idx_borders = np.cumsum(slice_nr_per_run[:-1])[:,np.newaxis] + \
              np.arange(-4,4)[np.newaxis,:]

fmri_data = np.delete(s1ds.samples, idx_borders, axis=0)

patches = load(spenc_dir+'MaThe/'+\
                      'transformed_data/sparse_patches.pkl')

# keep only every 10th sample so there's no overlap between the patches
patches = patches[::10].copy()

# the length of the movie segments without the transition TRs 
# (like they are saved in patches)
movieseg_duration = duration[:]
movieseg_duration[0] -= 8
movieseg_duration[-1] -= 8
movieseg_duration[1:-1] -= 16

mvcs = np.cumsum(movieseg_duration)

# we need to remove the last last 2s of the second to last stimulus
to_delete = (mvcs[-2]-2)*10 + np.arange(20)

patches = np.delete(patches, to_delete, axis=0)

# and we're going to remove the last fmri slice
# since it does not correspond to a movie part anymore
fmri_data = fmri_data[:-1, :]

# shape of TR samples
# note: column ordering is now oldest --> newest in steps of 50
patches = np.reshape(patches, (-1, 50*20))
fmri_data = fmri_data[3:]

strides = (patches.strides[0],) + patches.strides

# rolling window of length 4 samples
shape = (patches.shape[0] - 4 + 1, 4, patches.shape[1])

patches = np.lib.stride_tricks.as_strided(patches[::-1,:].copy(),
                                          shape=shape,
                                          strides=strides)[::-1, :, :]

patches = np.reshape(patches, (patches.shape[0], -1))

# we kick out the most recent sample
patches = patches[:, :-1000]

dump(patches,
     spenc_dir+'MaThe/prepro/BSC_stimuli_subj_{}.pkl'.format(subj))

dump(fmri_data,
     spenc_dir+'MaThe/prepro/BSC_fmri_subj_{}.pkl'.format(subj))
