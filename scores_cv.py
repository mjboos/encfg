# coding: utf-8
from __future__ import division
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

fmri_data = load(spenc_dir+'MaThe/prepro/'+model_name+\
        '_fmri_subj_{}.pkl'.format(subj), mmap_mode='r+')
stimuli = load(spenc_dir+'MaThe/prepro/'+model_name+\
        '_stimuli_subj_{}.pkl'.format(subj), mmap_mode='r+')

# use KFold as iterator for voxel chunks
voxel_kfold = KFold(fmri_data.shape[1], n_folds=10)

stimuli_train, stimuli_test, fmri_train, fmri_test = train_test_split(
        stimuli, fmri_data, test_size=0.3)

cv = KFold(stimuli_train.shape[0], n_folds=3)
alpha_grid = np.array([1e2, 1e3, 1e4])
cv_scores = dict()

with Parallel(n_jobs=1) as parallel:
    for alpha in alpha_grid:
        scores = []
        for train,test in cv:
            ridge = Ridge(alpha=alpha)
            predictions = np.hstack(parallel(delayed(fit_predict)(
                ridge, stimuli_train[train], fmri_train[:, chunk][train],
                stimuli_train[test]) for (_, chunk) in voxel_kfold))
            scores.append(r2_score(fmri_train[test],
                                       predictions, multioutput='raw_values'))
        cv_scores[alpha] = np.mean(scores, axis=0)

    # find best cv scores
    best_alphas = alpha_grid[np.argmax([cvsc
        for key, cvsc in sorted(cv_scores.iteritems())], axis=0)]

    scores = []
    predictions = np.hstack(parallel(delayed(fit_predict)(
        ridge, stimuli_train, fmri_train[:, chunk],
        stimuli_test) for _, chunk in voxel_kfold))

test_score = r2_score(fmri_test, predictions, multioutput='raw_values')

dump({'scores':test_score, 'cv':best_alphas},
     spenc_dir+'MaThe/scores/{}_subj_{}.pkl'.format(model_name, subj),
     compress=3)

