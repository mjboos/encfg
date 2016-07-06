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

# With MFS Features
mfs_list = glob.glob(spenc_dir + 'AudioStimuli/*.mfs')
feature_list = [np.genfromtxt(mfs_fn,delimiter=',')
                for mfs_fn in sorted(mfs_list)]
ft_freq = feature_list[0].shape[1]

chunk_lens = [np.sum(s1ds.sa['chunks'].value==i) for i in xrange(8)]

#the minimum of 2s samples in the features or data is the maximally allowed 
chunk_max= [min(np.trunc(ft.shape[0] / 20),chunk_lens[i])
            for i,ft in enumerate(feature_list)]

#get index of elements which chunk-specific index is larger than
#the corresponding element in chunk_max and delete them 
s1ds = s1ds[np.delete(np.arange(s1ds.shape[0]),
            [np.sum(chunk_lens[:i])+chunk_max[i]
             for i in xrange(8) if chunk_lens[i] > chunk_max[i]]),:]
#cut off what doesn't fit into 2s samples
data = np.reshape(np.vstack([ft[:chunk_max[i]*20,:]
                            for i,ft in enumerate(feature_list)]),
                  (-1,ft_freq*20))


# now lag the audiofeatures

# first add 3 rows of zeros in the beginning 
# (since we are lagging by 3 additional rows)
data = np.vstack((np.zeros((3,20*ft_freq)),data))

# lagging by stride tricks
strides = (data.strides[0],data.strides[0],data.strides[1])
shape = (data.shape[0]-3,4,data.shape[1])
lagged_data= np.lib.stride_tricks.as_strided(data[::-1,:].copy(),
                                             shape=shape, strides=strides)
lagged_data = lagged_data[::-1,:,:]

# kick out the audio features presented at the timepoint to which the index
# corresponds --> too recent for any BOLD
lagged_data = np.reshape(lagged_data[:,1:,:],(lagged_data.shape[0],-1))

#and the ones that still contain zeros
lagged_data = StandardScaler().fit_transform(lagged_data[3:,:])
s1ds = s1ds[3:,:]

dump(lagged_data,
     spenc_dir+'MaThe/prepro/MFS_stimuli_subj_{}.pkl'.format(subj))

voxel_kfold = KFold(s1ds.shape[1], n_folds=10)

for i, (_, split) in enumerate(voxel_kfold):
    dump(s1ds.samples[:, split],
         '/home/data/scratch/mboos/prepro/MFS_fmri_subj_{}_split_{}.pkl'.format(subj, i),
         compress=3)

