import glob
import numpy as np
import featurespace_fun as fsf
import matplotlib.pyplot as plt
from nilearn.masking import apply_mask
from nilearn.input_data import MultiNiftiMasker
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
from nilearn.plotting import plot_glass_brain
from nilearn.image import smooth_img
from scipy.stats import ttest_1samp
import sys
from nibabel import load

model = sys.argv[1]

mask = 'brainmask_group_template.nii.gz'

valid_subjs = [1,2,5,6,7,8,9,11,12,14,15,16,17,18,19]

scores = apply_mask(smooth_img(sorted(glob.glob('MaThe/avg_maps/model_depcor_{}_*'.format(model))), 5.0), mask_img=mask).astype('float32')
mean_scores = scores.mean(axis=0)
tscores, p_values = ttest_1samp(scores, 0, axis=0)
p_values[np.isnan(p_values)] = 1
which_ones = p_values > 0.05
mean_scores[which_ones] = 0
tscores[which_ones] = 0
display = fsf.plot_diff_avg_whole(mean_scores, 0.001)
display.savefig('mean_5mm_unthresh_depcor_model_whole_{}.svg'.format(model))
display.savefig('mean_5mm_unthresh_depcor_model_whole_{}.png'.format(model))
fsf.save_map_avg_whole(mean_scores, threshold=None, model='5mm_depcor_'+model)
display = fsf.plot_diff_avg_whole(tscores, 0.001)
display.savefig('ttest_5mm_unthresh_depcor_model_whole_{}.svg'.format(model))
display.savefig('ttest_5mm_unthresh_depcor_model_whole_{}.png'.format(model))
