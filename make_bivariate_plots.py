# coding: utf-8
from __future__ import division
import numpy as np
import seaborn as sns
from nilearn.plotting import plot_stat_map
from nilearn.masking import unmask
from nilearn.image import threshold_img
from joblib import load
import pylab as plt
import argparse
import pandas as pd

def get_best_voxels(voxels, n_voxels=1000):
    '''Returns the n_voxels best voxels from ndarray voxels'''
    return np.sort(voxels)[::-1][:n_voxels]

models = ['BSC', 'MFS']
subj_list = [subj for subj in xrange(1, 20)]
threshold = 1e-2
parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, nargs='*', default=models)
parser.add_argument('--subj', type=int, nargs='*', default=subj_list)
parser.add_argument('--threshold', type=int, default = threshold)
parser.add_argument('--ref', type=str, nargs='*', default=models[:1])
arg_namespace = vars(parser.parse_args())

models, subj_list, threshold, ref = [arg_namespace[key]
                                     for key in ['models', 'subj',
                                                 'threshold', 'ref']]

spenc_dir = '/home/mboos/SpeechEncoding/'

for subj in subj_list:
    model_scores = {model : np.concatenate([load(spenc_dir+'MaThe/scores/{}_subj_{}_split_{}.pkl'.format(model, subj, split))
                                            for split in xrange(10)])
                    for model in models}
    for ref_model in ref:
        # plot the best voxels
        for model, scores in model_scores.iteritems():
            if model != ref_model:
                voxel_limit = np.sum(model_scores[ref_model]>threshold)
                vxl_to_keep = np.argsort(model_scores[ref_model])[::-1][:voxel_limit]
                plt.figure(figsize=(10, 10))
                sns.jointplot(
                        x=ref_model, y=model, data=pd.DataFrame(
                            {ref_model:model_scores[ref_model][vxl_to_keep],
                             model:scores[vxl_to_keep]}),
                        alpha=0.1)
                plt.savefig(spenc_dir+'MaThe/plots/'+\
                        'bivariate_{}_subj_{}.svg'.format(
                            '_'.join((ref_model, model)), subj))
                plt.close()

