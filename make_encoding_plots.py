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

models = ['BSC']
subj_list = [subj for subj in xrange(1, 20)]
threshold = 1e-2
parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, nargs='*', default=models)
parser.add_argument('--subj', type=int, nargs='*', default=subj_list)
parser.add_argument('--threshold', type=float, default = threshold)

arg_namespace = vars(parser.parse_args())

models, subj_list, threshold = [arg_namespace[key]
                                for key in ['models', 'subj', 'threshold']]

spenc_dir = '/home/mboos/SpeechEncoding/'

for subj in subj_list:
    try:
        model_scores = {model : np.concatenate([load(spenc_dir+'MaThe/scores/{}_subj_{}_split_{}.pkl'.format(model, subj, split))
                                                for split in xrange(10)])
                        for model in models}
    except IOError:
        print("IOError for subj {}".format(subj))
        continue
    # cut at zero
    for model in model_scores.keys():
        model_scores[model][model_scores[model]<0] = 0

    max_r2 = max(np.max(value) for value in model_scores.values())
    voxel_limit = max(np.sum(value>threshold) for value in model_scores.values())

    fig = plt.figure(figsize=(8, 6))
    # plot the best voxels
    for model, scores in model_scores.iteritems():
        plt.plot(get_best_voxels(scores, voxel_limit), label=model)
    plt.legend()
    plt.ylim(0, 0.05 + max_r2)
    plt.xlabel('Voxel sorted by explained variance')
    plt.ylabel('Explained variance')
    plt.savefig(spenc_dir+'MaThe/plots/'+\
            '{}_r2_subj_{}.svg'.format('_'.join(model_scores.keys()), subj))
    plt.close()

    # best voxels as a violin plot
    sns.violinplot(pd.DataFrame(
        {model:get_best_voxels(scores, voxel_limit)
            for model, scores in model_scores.iteritems()}), cut=0)
    plt.savefig(spenc_dir+'MaThe/plots/'+\
            '{}_violin_subj_{}.svg'.format('_'.join(model_scores.keys()), subj))
    plt.close()

    subj_mask = spenc_dir+\
            'temporal_lobe_mask_head_subj{0:02}bold.nii.gz'.format(subj)
    background_img = '/home/data/psyinf/forrest_gump/anondata/sub{0:03}/'.format(subj)+\
            'templates/bold7Tp1/head.nii.gz'
    # plotting the statistical map on the brain
    for model, scores in model_scores.iteritems():
        unmasked = unmask(scores, subj_mask)
        # thresholding
        unmasked = threshold_img(unmasked, threshold)
        display = plot_stat_map(
                unmasked, bg_img=background_img, symmetric_cbar=False,
                title='Explained variance per voxel ({})'.format(model),
                vmax=max_r2, draw_cross=False)
        plt.savefig(spenc_dir+'MaThe/plots/'+\
                '{}_r2_map_subj_{}.svg'.format(model, subj))
        plt.close()

