# coding: utf-8
from __future__ import division
import mvpa2.suite as mvpa
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
from nilearn.plotting import plot_stat_map
from nilearn.masking import unmask
from nilearn.image import threshold_img
from sklearn.metrics import r2_score
from sklearn.externals import joblib
import pylab as plt
from json import load
import sys

model_name, subj = sys.argv[1:]
subj = int(subj)

spenc_dir = '/home/mboos/SpeechEncoding/'

subj_preprocessed_path = 'PreProcessed/FG_subj{}pp.gzipped.hdf5'.format(subj)
s1ds = mvpa.h5load(spenc_dir + subj_preprocessed_path)

with open(spenc_dir + 'DialogData/german_dialog_20150211.json') as fh:
    dialog = load(fh)
with open(spenc_dir + 'DialogData/german_audio_description_20150211.json') as fh:
    description = load(fh)

dialog_SE = [(anno['begin'],anno['end']) for anno in dialog['annotations']]
description_SE = [(anno['begin'],anno['end'])
                  for anno in description['annotations']]

speech_SE = dialog_SE + description_SE
speech_arr = np.array(speech_SE)
speech_arr = speech_arr[np.argsort(speech_arr[:,0]),:]

#MFS stepsize is 10ms, speech begin/end is in ms, so we divide by 10
speech_arr = speech_arr / 1000

duration = np.array([902,882,876,976,924,878,1084,676])

# i did not kick out the first/last 4 samples per run yet
slice_nr_per_run = [dur/2 for dur in duration]

# use broadcasting to get indices to delete around the borders
idx_borders = np.cumsum(slice_nr_per_run[:-1])[:,np.newaxis] + \
              np.arange(-4,4)[np.newaxis,:]

fmri_data = np.delete(s1ds.samples, idx_borders, axis=0)

patches = joblib.load(spenc_dir+'MaThe/'+\
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
splits = np.linspace(0, fmri_data.shape[1], num=20)

overall_score = []
ridge = Ridge(alpha=1000)
cv = KFold(fmri_data.shape[0], n_folds=5)

for i in xrange(1,20):
    scores = []
    for train,test in cv:
        predictions = ridge.fit(patches[train],
                                fmri_data[train, splits[i-1]:splits[i]]
                                ).predict(patches[test])
        scores.append(r2_score(fmri_data[test, splits[i-1]:splits[i]],
                               predictions, multioutput='raw_values'))

    scores = np.mean(scores, axis=0)
    overall_score.append(scores)

BSC_scores = np.concatenate(overall_score)

# With LQMFS Features
lqmfs_list = glob.glob(spenc_dir + 'AudioStimuli/*.lq_mfs')
feature_list = [np.genfromtxt(lqmfs_fn,delimiter=',')
                for lqmfs_fn in sorted(lqmfs_list)]
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
patches = StandardScaler().fit_transform(lagged_data[3:,:])
s1ds = s1ds[3:,:]
fmri_data = s1ds.samples

ridge = Ridge(alpha=10000)
cv = KFold(fmri_data.shape[0], n_folds=5)
overall_score = []
for i in xrange(1,20):
    scores = []
    for train,test in cv:
        predictions = ridge.fit(patches[train],
                                fmri_data[train, splits[i-1]:splits[i]]
                                ).predict(patches[test])
        scores.append(r2_score(fmri_data[test, splits[i-1]:splits[i]],
                               predictions, multioutput='raw_values'))

    scores = np.mean(scores, axis=0)
    overall_score.append(scores)

MFS_scores = np.concatenate(overall_score)

joblib.dump({'sparse':BSC_scores,'lqmfs':MFS_scores},
            spenc_dir+'MaThe/'+'{}_scores_subj_{}.pkl'.format(model_name, subj),
            compress=3)

