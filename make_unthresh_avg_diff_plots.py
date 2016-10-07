import glob
import numpy as np
import featurespace_fun as fsf
import matplotlib.pyplot as plt
from nilearn.masking import apply_mask
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
from scipy.stats import ttest_1samp
import sys

models = sys.argv[1:]

#models = ['logBSC_H200_ds_conv', 'logMFS_ds']

mask = 'temporal_lobe_mask_grp_7T_test.nii.gz'

scores_bsc = np.arctanh(apply_mask(sorted(glob.glob('MaThe/avg_maps/model_{}_subj_*'.format(models[0]))), mask_img=mask))
scores_mfs = np.arctanh(apply_mask(sorted(glob.glob('MaThe/avg_maps/model_{}_subj_*'.format(models[1]))), mask_img=mask))
scores_bsc[scores_bsc<0] = 0
scores_mfs[scores_mfs<0] = 0
diff_scores = scores_bsc - scores_mfs
mean_diff = diff_scores.mean(axis=0)
tscores, p_values = ttest_1samp(diff_scores, 0, axis=0)
p_values[np.isnan(p_values)] = 1
which_ones = p_values > 0.05
mean_diff[which_ones] = 0
tscores[which_ones] = 0
display = fsf.plot_diff_avg(mean_diff, 0.001)
display.savefig('mean_unthresh_diff_model_{}.svg'.format('_'.join(models)))
display.savefig('mean_unthresh_diff_model_{}.png'.format('_'.join(models)))
fsf.save_map_avg('avg', mean_diff, threshold=None, model='diff_'+'_'.join(models))
display = fsf.plot_diff_avg(tscores, 0.001)
display.savefig('ttest_unthresh_diff_model_{}.svg'.format('_'.join(models)))
display.savefig('ttest_unthresh_diff_model_{}.png'.format('_'.join(models)))
