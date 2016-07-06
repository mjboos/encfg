from __future__ import division
import mvpa2.suite as mvpa
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
import sys
from joblib import dump

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


# With LQMFS Features
lq_mfs_list = glob.glob(spenc_dir + 'AudioStimuli/*.lq_mfs')
feature_list = [np.genfromtxt(lq_mfs_fn,delimiter=',')
                for lq_mfs_fn in sorted(lq_mfs_list)]
ft_freq = feature_list[0].shape[1]

def reduce_LQMFS(a, b):
    '''deletes offset'''
    return np.concatenate([a[:-8*10], b[8*10:]], axis=0)

feature_list = [feat[:10*dur] for feat, dur in zip(feature_list, duration)]
features = reduce(reduce_LQMFS, feature_list)
features = features[:-(features.shape[0] % 10)]


features = np.reshape(features, (-1, ft_freq*20))

strides = (features.strides[0],) + features.strides

# rolling window of length 4 samples
shape = (features.shape[0] - 4 + 1, 4, features.shape[1])

features = np.lib.stride_tricks.as_strided(features[::-1,:].copy(),
                                          shape=shape,
                                          strides=strides)[::-1, :, :]

features = np.reshape(features, (features.shape[0], -1))

# we kick out the most recent sample
features = features[:, :-20*ft_freq]


features = StandardScaler().fit_transform(features)

dump(features,
     spenc_dir+'MaThe/prepro/LQMFS_stimuli_subj_{}.pkl'.format(subj))
