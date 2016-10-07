import glob
import numpy as np
import featurespace_fun as fsf
import matplotlib.pyplot as plt
from nilearn.masking import apply_mask
from nilearn.input_data import MultiNiftiMasker
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
from nilearn.plotting import plot_glass_brain
from scipy.stats import ttest_1samp
import sys
from nibabel import load
from nilearn.image import smooth_img

models = sys.argv[1:]

mask = 'brainmask_group_template.nii.gz'

valid_subjs = [1,2,5,6,7,8,9,11,12,14,15,16,17,18,19]

scores_bsc = np.arctanh(apply_mask(smooth_img(sorted(glob.glob('MaThe/avg_maps/model_{}_*whole*'.format(models[0]))), 3.0), mask_img=mask)).astype('float32')
scores_mfs = np.arctanh(apply_mask(smooth_img(sorted(glob.glob('MaThe/avg_maps/model_{}_*whole*'.format(models[1]))), 3.0), mask_img=mask)).astype('float32')
scores_bsc[scores_bsc<0] = 0
scores_mfs[scores_mfs<0] = 0
diff_scores = scores_bsc - scores_mfs
mean_diff = diff_scores.mean(axis=0)
tscores, p_values = ttest_1samp(diff_scores, 0, axis=0)
p_values[np.isnan(p_values)] = 1
which_ones = p_values > 0.05
mean_diff[which_ones] = 0
tscores[which_ones] = 0
display = fsf.plot_diff_avg_whole(mean_diff, 0.001)
display.savefig('mean_unthresh_diff_model_whole_3mm_{}.svg'.format('_'.join(models)))
display.savefig('mean_unthresh_diff_model_whole_3mm_{}.png'.format('_'.join(models)))
fsf.save_map_avg_whole(mean_diff, threshold=None, model='3mm_diff_'+'_'.join(models))
display = fsf.plot_diff_avg_whole(tscores, 0.001)
display.savefig('ttest_unthresh_diff_model_whole_3mm_{}.svg'.format('_'.join(models)))
display.savefig('ttest_unthresh_diff_model_whole_3mm_{}.png'.format('_'.join(models)))
