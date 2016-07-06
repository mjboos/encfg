# coding: utf-8
from __future__ import division
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from nilearn.plotting import plot_stat_map
from nilearn.plotting import cm
from nilearn.masking import unmask
from nilearn.image import threshold_img
from joblib import load
import argparse
import pylab as plt

cmap_cycle = [cm.black_red, cm.black_green, cm.black_pink,
              cm.black_purple, cm.black_blue]
cmap_names = ['red/orange', 'green', 'pink', 'purple', 'blue']

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
    model_scores = {model : np.concatenate([load(spenc_dir+'MaThe/scores/{}_subj_{}_split_{}.pkl'.format(model, subj, split))
                                            for split in xrange(10)])
                    for model in models}
    # cut at zero
    for model in model_scores.keys():
        model_scores[model][model_scores[model]<0] = 0

    subj_mask = spenc_dir+\
            'temporal_lobe_mask_head_subj{0:02}bold.nii.gz'.format(subj)
    background_img = '/home/data/psyinf/forrest_gump/anondata/sub{0:03}/'.format(subj)+\
            'templates/bold7Tp1/head.nii.gz'
    nonz_perf = np.logical_not(np.all(
        [model_scores[model]==0 for model in sorted(model_scores.keys())],
        axis=0))
    model_argmax = np.argmax([model_scores[model]
                             for model in sorted(models)], axis=0)

    sorted_score_list = np.sort([model_scores[model]
                                 for model in sorted(models)],
                                axis=0)

    model_diffs = sorted_score_list[-1] - sorted_score_list[-2]
    diff_max = np.max(model_diffs)

    for i, model in enumerate(sorted(models)):
        model_curr_diff = np.zeros_like(model_diffs)
        elements = np.logical_and(nonz_perf, model_argmax==i)
        model_curr_diff[elements] = model_diffs[elements]

        unmasked = unmask(model_curr_diff, subj_mask)
        unmasked = threshold_img(unmasked, threshold)
        if i == 0:
            display = plot_stat_map(
                    unmasked, bg_img=background_img, symmetric_cbar=False,
                    title=' '.join(['{}:{}'.format(m,n) for m,n in  zip(models, cmap_names)]),
                    threshold=threshold, cmap=cmap_cycle[i], vmax=diff_max, draw_cross=False,
                    display_mode='z', cut_coords=6, colorbar=False, dim=-.5)
        else:
            display.add_overlay(unmasked, cmap=cmap_cycle[i], threshold=threshold)

    fig = plt.gcf()
    fig.set_size_inches(16, 12)

    plt.savefig(spenc_dir+'MaThe/plots/'+\
            'diff_comparison_subj_{}_models_{}.svg'.format(subj, '_'.join(models)))
    plt.close()

