# coding: utf-8
from __future__ import division
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.externals.joblib import load, dump
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('subj', type=int)
parser.add_argument('split', type=int)
parser.add_argument('--models', type=str, nargs='*')

arg_namespace = vars(parser.parse_args())

subj, split, models = [arg_namespace[key]
                                       for key in ['subj',
                                                   'split',
                                                   'models']]

spenc_dir = '/home/mboos/SpeechEncoding/'

fmri_data = load('/home/data/scratch/mboos/prepro/'+\
        'fmri_subj_{}_split_{}.pkl'.format(subj, split))

stimuli = [load(spenc_dir+'MaThe/prepro/'+model+\
        '_stimuli.pkl') for model in models]

cv = KFold(stimuli[0].shape[0], n_folds=8)

test_predictions = [[] for model in models]

for train, test in cv:
    for i, model in enumerate(models):
        lireg = LinearRegression().fit(stimuli[i][train], fmri_data[train])
        predictions = lireg.predict(stimuli[i][test])
        test_predictions[i].append(predictions)

test_predictions = [np.vstack(preds) for preds in test_predictions]

correlations = {model:np.array([np.corrcoef(preds[:, i], fmri_data[:, i])[0,1] for i in xrange(fmri_data.shape[1])])
               for model, preds in zip(models, test_predictions)}
for i in xrange(len(models)):
    for j in xrange(i+1, len(models)):
        correlations[models[i]+'_'+models[j]] = np.array([np.corrcoef(test_predictions[i][:, voxel], test_predictions[j][:, voxel]) 
                                                      for voxel in xrange(test_predictions[i].shape[1])])

dump(correlations,
     spenc_dir+'MaThe/scores/many_{}_lm_subj_{}_split_{}.pkl'.format('_'.join(models),
                                                             subj, split),
     compress=3)
