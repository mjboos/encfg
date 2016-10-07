# coding: utf-8
from __future__ import division
import mvpa2.suite as mvpa
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.externals.joblib import Parallel, delayed, load, dump
from encoding_helpers import fit_predict
import sys

model_name, subj = sys.argv[1:]
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

# use KFold as iterator for voxel chunks
voxel_kfold = KFold(fmri_data.shape[1], n_folds=20)

patches_train, patches_test, fmri_train, fmri_test = train_test_split(
        patches, fmri_data, test_size=0.2)

cv = KFold(patches_train.shape[0], n_folds=3)
alpha_grid = np.array([1e2, 1e3, 1e4])
cv_scores = dict()

with Parallel(n_jobs=10) as parallel:
    for alpha in alpha_grid:
        scores = []
        for train,test in cv:
            ridge = Ridge(alpha=alpha)
            predictions = np.hstack(parallel(delayed(fit_predict)(
                ridge, patches_train[train], fmri_train[:, chunk][train],
                patches_train[test]) for (_, chunk) in voxel_kfold))
            scores.append(r2_score(fmri_train[test],
                                       predictions, multioutput='raw_values'))
        cv_scores[alpha] = np.mean(scores, axis=0)

    # find best cv scores
    best_alphas = alpha_grid[np.argmax([cvsc
        for key, cvsc in sorted(cv_scores.iteritems())], axis=0)]

    scores = []
    predictions = np.hstack(parallel(delayed(fit_predict)(
        ridge, patches_train, fmri_train[:, chunk],
        patches_test) for _, chunk in voxel_kfold))

BSC_score = r2_score(fmri_test, predictions, multioutput='raw_values')

dump({'scores':BSC_score, 'cv':best_alphas},
     spenc_dir+'MaThe/scores/BSC_{}_subj_{}.pkl'.format(model_name, subj),
     compress=3)

