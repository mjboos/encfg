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
        'fmri_of_subj_{}_split_{}.pkl'.format(subj, split))

stimuli = load(spenc_dir+'MaThe/prepro/'+model_name+\
        '_stimuli.pkl')

cv = KFold(stimuli.shape[0], n_folds=8)

test_predictions = []
for train, test in cv:
    ridge = RidgeCV(alphas=alpha_grid).fit(stimuli[train], fmri_data[train])
    predictions = ridge.predict(stimuli[test])
    test_predictions.append(predictions)

test_predictions = np.vstack(test_predictions)

test_score = np.array([np.corrcoef(test_predictions[:, i], fmri_data[:, i])[0,1] for i in xrange(fmri_data.shape[1])])

dump(test_score,
     spenc_dir+'MaThe/scores/{}_of_subj_{}_split_{}.pkl'.format(model_name,
                                                             subj, split),
     compress=3)

