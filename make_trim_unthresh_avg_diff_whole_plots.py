import glob
import numpy as np
import featurespace_fun as fsf
import matplotlib.pyplot as plt
from nilearn.masking import apply_mask
from nilearn.input_data import MultiNiftiMasker
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
from scipy.stats import ttest_1samp, trim_mean
import sys
from nibabel import load
from scipy.stats.mstats import trimmed_mean_ci

models = sys.argv[1:]

mask = 'brainmask_group_template.nii.gz'

scores_bsc = np.arctanh(apply_mask(sorted(glob.glob('MaThe/avg_maps/model_{}_*whole*'.format(models[0]))), mask_img=mask)).astype('float32')
scores_mfs = np.arctanh(apply_mask(sorted(glob.glob('MaThe/avg_maps/model_{}_*whole*'.format(models[1]))), mask_img=mask)).astype('float32')
scores_bsc[scores_bsc<0] = 0
scores_mfs[scores_mfs<0] = 0
diff_scores = scores_bsc - scores_mfs
mean_diff = trim_mean(diff_scores, 0.08, axis=0)
trim_mean_ci = trimmed_mean_ci(diff_scores, (0.08, 0.08), axis=0)
which_ones = np.logical_not(np.logical_or(trim_mean_ci[0,:] > 0, trim_mean_ci[1,:] < 0))
mean_diff[which_ones] = 0
display = fsf.plot_diff_avg_whole(mean_diff, 0.001)
display.savefig('mean_trim_unthresh_diff_model_whole_{}.svg'.format('_'.join(models)))
display.savefig('mean_trim_unthresh_diff_model_whole_{}.png'.format('_'.join(models)))
fsf.save_map_avg_whole(mean_diff, threshold=None, model='diff_trim_'+'_'.join(models))
