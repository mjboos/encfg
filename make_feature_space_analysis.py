# coding: utf-8
from __future__ import division
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
from nilearn.plotting import plot_stat_map
from nilearn.masking import unmask
from nilearn.image import threshold_img
from scipy import corrcoef
import featurespace_fun as fsf

threshold = 0.01
models = ['logMFS']
subj_list = [subj for subj in xrange(1, 20)]

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, nargs='*', default=models)
parser.add_argument('--subj', type=int, nargs='*', default=subj_list)
parser.add_argument('--threshold', type=float, default=threshold)

arg_namespace = vars(parser.parse_args())
models, subj_list, threshold = [arg_namespace[var] for var in
                                ['models', 'subj', 'threshold']]

spenc_dir = '/home/mboos/SpeechEncoding/'

logenergy = joblib.load(spenc_dir+'MaThe/prepro/logMFS_stimuli.pkl').sum(axis=1)

for subj in subj_list:
    fmri_data = np.hstack([joblib.load('/home/data/scratch/mboos/prepro/fmri_subj_{}_split_{}.pkl'.format(subj, i)) for i in xrange(10)]).astype('float32')
    expl_var_pc = {}
    PC_rep = {}
    for model in models:
        model_preds = joblib.load(spenc_dir+'MaThe/predictions/{}_subj_{}_all.pkl'.format(model, subj))
        scores = r2_score(fmri_data, model_preds, multioutput='raw_values').astype('float32')
        r2_above_thresh = np.sum(scores>threshold)
        max_r2 = np.max(scores)
        best_vxl = np.argsort(scores)
        voxels_to_use = best_vxl[-r2_above_thresh:]
        pca = PCA(n_components=10).fit(model_preds[:, voxels_to_use])
        expl_var_pc[model] = pca.explained_variance_ratio_[:10]
        fig = plt.figure(figsize=(6, 6))
        filtered = pca.transform(model_preds[:, voxels_to_use])
        PC_rep[model] = filtered
        scatter = plt.scatter(filtered[:, 0], filtered[:, 1], cmap='YlGnBu',
                              c=logenergy, marker='o', alpha=0.3)
        plt.xlabel('1st PC')
        plt.ylabel('2nd PC')
        plt.title('{0}, 1st PC r w/ log stim energy {1:.2f}'.format(model, corrcoef(filtered[:, 0], logenergy)[0,1]))
        plt.savefig(spenc_dir+'MaThe/plots/feature_space/two_PCs_{}_subj_{}.svg'.format(model, subj))
        plt.close()

        subj_mask = spenc_dir + 'temporal_lobe_mask_head_subj{0:02}bold.nii.gz'.format(subj)
        background_img = '/home/data/psyinf/forrest_gump/anondata/sub{0:03}/'.format(subj) + 'templates/bold7Tp1/head.nii.gz'
        scores[scores<threshold] = 0
        unmasked = unmask(scores, subj_mask)
        unmasked = threshold_img(unmasked, threshold)
        display = plot_stat_map(
                        unmasked, bg_img=background_img, symmetric_cbar=False, aspect=1.25,
                        vmax=max_r2, threshold=threshold, draw_cross=False, dim=-1.)
        fig = plt.gcf()
        fig.set_size_inches(12, 4)

        plt.savefig(spenc_dir+'MaThe/plots/feature_space/PC_model_{}_subj_{}_pcnr_0.svg'.format(model, subj))
        plt.close()
        display = plot_stat_map(
                        unmasked, bg_img=background_img, symmetric_cbar=False, aspect=1.25,
                        vmax=max_r2, threshold=threshold, dim=-1.)

        plt.savefig(spenc_dir+'MaThe/plots/feature_space/PC_model_{}_subj_{}_pcnr_0_cross.svg'.format(model, subj))
        plt.close()
        preds_pc = np.zeros_like(filtered)
        preds_pc[:, 0] = filtered[:, 0]
        preds_pc = pca.inverse_transform(preds_pc)

        scores_1st_pc = np.zeros_like(scores)

        scores_1st_pc[voxels_to_use] = r2_score(fmri_data[:, voxels_to_use],
                                                preds_pc, multioutput='raw_values')
        display = plot_stat_map(
                        unmasked, bg_img=background_img, symmetric_cbar=False, aspect=1.25,
                        vmax=max_r2, threshold=threshold, draw_cross=False, dim=-1.)
        fig = plt.gcf()
        fig.set_size_inches(12, 4)

        plt.savefig(spenc_dir+'MaThe/plots/feature_space/PC_model_{}_subj_{}_pcnr_1.svg'.format(model, subj))
        plt.close()
        display = plot_stat_map(
                        unmasked, bg_img=background_img, symmetric_cbar=False, aspect=1.25,
                        vmax=max_r2, threshold=threshold, dim=-1.)

        plt.savefig(spenc_dir+'MaThe/plots/feature_space/PC_model_{}_subj_{}_pcnr_1_cross.svg'.format(model, subj))
        plt.close()
        preds_pc = np.zeros_like(filtered)
        preds_pc[:, 1] = filtered[:, 1]
        preds_pc = pca.inverse_transform(preds_pc)

        scores_2nd_pc = np.zeros_like(scores)

        scores_2nd_pc[voxels_to_use] = r2_score(fmri_data[:, voxels_to_use],
                                                preds_pc, multioutput='raw_values')
        unmasked = unmask(scores_2nd_pc, subj_mask)

        unmasked = threshold_img(unmasked, threshold)
        display = plot_stat_map(
                        unmasked, bg_img=background_img, symmetric_cbar=False, aspect=1.25,
                        threshold=threshold, draw_cross=False, dim=-1.)
        fig = plt.gcf()
        fig.set_size_inches(12, 4)

        plt.savefig(spenc_dir+'MaThe/plots/feature_space/PC_model_{}_subj_{}_pcnr_2.svg'.format(model, subj))
        plt.close()
        display = plot_stat_map(
                        unmasked, bg_img=background_img, symmetric_cbar=False, aspect=1.25,
                        threshold=threshold, dim=-1.)

        plt.savefig(spenc_dir+'MaThe/plots/feature_space/PC_model_{}_subj_{}_pcnr_2_cross.svg'.format(model, subj))
        plt.close()
    fig, ax = plt.subplots()
    for key in expl_var_pc:
        ax.plot(expl_var_pc[key], marker='o', linestyle='-', label=key)
    plt.legend()
    if len(models)==2:
        for i, pc in enumerate(expl_var_pc[models[0]]):
            ax.annotate('{0:.2f}'.format(corrcoef(PC_rep[models[0]][:, i], PC_rep[models[1]][:, i])[0,1]),
                    (i, pc))
    plt.savefig(spenc_dir+'MaThe/plots/feature_space/PC_comp_{}_subj_{}.svg'.format(' '.join(models), subj))
    plt.close()

