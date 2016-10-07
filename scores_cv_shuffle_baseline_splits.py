# coding: utf-8
from __future__ import division
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.externals.joblib import load, dump
import sys
import argparse

alpha_grid = [1e2, 1e3, 1e4]

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str)
parser.add_argument('subj', type=int)
parser.add_argument('split', type=int)
parser.add_argument('--alpha', default=alpha_grid, type=float, nargs='*')

arg_namespace = vars(parser.parse_args())

model_name, subj, split, alpha_grid = [arg_namespace[key]
                                       for key in ['model_name',
                                                   'subj',
                                                   'split',
                                                   'alpha']]


spenc_dir = '/home/mboos/SpeechEncoding/'

fmri_data = load('/home/data/scratch/mboos/prepro/'+\
        'fmri_subj_{}_split_{}.pkl'.format(subj, split))

stimuli = load(spenc_dir+'MaThe/prepro/'+model_name+\
        '_stimuli_subj_{}.pkl'.format(subj))

cv = KFold(stimuli.shape[0], n_folds=8)

test_score = []
for train, test in cv:
    train_perm = np.random.permutation(train)
    ridge = RidgeCV(alphas=alpha_grid).fit(stimuli[train_perm], fmri_data[train])
    predictions = ridge.predict(stimuli[test])
    test_score.append(r2_score(fmri_data[test], predictions,
                               multioutput='raw_values'))

#test_score = np.mean(test_score, axis=0)

dump(test_score,
     spenc_dir+'MaThe/scores/shuffled_{}_subj_{}_split_{}.pkl'.format(model_name,
                                                                      subj, split),
     compress=3)

