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
        '_stimuli.pkl'.format(subj))

cv = KFold(stimuli.shape[0], n_folds=8, random_state=500)

test_score = []
predictions_list = [RidgeCV(alphas=alpha_grid
    ).fit(stimuli[train], fmri_data[train]
        ).predict(stimuli[test]).astype('float32')
    for train, test in cv]

dump(predictions_list,
     spenc_dir+'MaThe/predictions/{}_subj_{}_split_{}.pkl'.format(model_name,
                                                             subj, split),
     compress=3)

