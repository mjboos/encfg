import glob
import numpy as np
import featurespace_fun as fsf
import matplotlib.pyplot as plt
from nilearn.masking import apply_mask
from nilearn.input_data import MultiNiftiMasker
from scipy.stats import norm
from nilearn.plotting import plot_glass_brain
from scipy.stats import ttest_1samp
import sys
from nibabel import load

mask = 'brainmask_group_template.nii.gz'

maps = [sorted(glob.glob('MaThe/maps/mni/model_5mm_depcor_{}_subj_*whole_avg*'.format(model))) for model in ['lda', 'speaker', 'emotions']]

valid_subjs = [mp.split('_')[-4] for mp in maps[0]]
for subj, ft_maps in zip(sorted(valid_subjs), zip(*maps)):
    display = plot_glass_brain(None, plot_abs=False, threshold=0.001)
    for ind_ft_map, color in zip(ft_maps, ['r', 'b', 'g']):
        level_thr = np.percentile(apply_mask('MaThe/maps/'+ind_ft_map.split('/')[-1], mask), 99)
        display.add_contours(ind_ft_map, colors=[color], levels=[level_thr], alpha=0.6, linewidths=3.0)
    display.savefig('contours_5mm_99perc_tvals_subj_{}.png'.format(subj))
    plt.close()

