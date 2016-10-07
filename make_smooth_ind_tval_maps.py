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
for subj_nr, subj in enumerate(valid_subjs):
    fsf.save_map_avg_whole(scores[subj_nr], threshold=None, model='5mm_depcor_'+model+'_subj_{}'.format(subj))
