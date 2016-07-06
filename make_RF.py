# coding: utf-8
from __future__ import division
import numpy as np
import seaborn as sns
from joblib import load
import pylab as plt
import argparse
import pandas as pd
from sklearn.linear_model import RidgeCV

alpha_grid = np.array([1e3, 1e4, 1e5])

def get_best_voxels(voxels, n_voxels=1000):
    '''Returns the n_voxels best voxels from ndarray voxels'''
    return np.sort(voxels)[::-1][:n_voxels]

model = 'MFS'
subj_list = [subj for subj in xrange(1, 20)]
num = 20
parser = argparse.ArgumentParser()
parser.add_argument('--subj', type=int, nargs='*', default=subj_list)
parser.add_argument('--number', type=int, default = num)

arg_namespace = vars(parser.parse_args())

subj_list, num = [arg_namespace[key]
                        for key in ['subj', 'number']]

spenc_dir = '/home/mboos/SpeechEncoding/'

for subj in subj_list:
    model_scores = np.concatenate([load(spenc_dir+'MaThe/scores/{}_subj_{}_split_{}.pkl'.format(model, subj, split))
                                   for split in xrange(10)])
    best_voxel = np.argsort(model_scores)[-num:]
    best_voxel_perf = np.sort(model_scores)[-num:]
    fmri_data = np.hstack([load('/home/data/scratch/mboos/prepro/fmri_subj_{}_split_{}.pkl'.format(subj, split)) for split in xrange(10)])
    stimuli = load(spenc_dir+'MaThe/prepro/'+model+\
            '_stimuli_subj_{}.pkl'.format(subj), mmap_mode='r+')
    ridge = RidgeCV(alphas=alpha_grid).fit(stimuli, fmri_data[:, best_voxel])
    RF = ridge.coef_
    plt.figure(figsize=(10, 10))
    for i in xrange(num):
        plt.imshow(np.reshape(RF[i, :], (-1, 48)).T, interpolation='nearest',
                   origin='lower', aspect='auto', cmap='coolwarm',
                   vmin=-max(np.abs(np.min(RF[i, :])), np.max(RF[i, :])),
                   vmax=max(np.abs(np.min(RF[i, :])), np.max(RF[i, :])))
        plt.colorbar()
        plt.gca().set_xticklabels([xt/10-8 for xt in plt.xticks()[0]])
        plt.xlabel('seconds')
        plt.title(r'RF for voxel with $r^2$={0:.2f}'.format(best_voxel_perf[i]))
        plt.savefig(spenc_dir+'MaThe/plots/RF_subj_{}_voxel_{}.svg'.format(subj, i))
        plt.close()



