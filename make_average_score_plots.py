import glob
import numpy as np
import featurespace_fun as fsf
import matplotlib.pyplot as plt
from nilearn.masking import apply_mask
from scipy.stats import norm, ttest_1samp
from statsmodels.sandbox.stats.multicomp import fdrcorrection0

models = ['logBSC_H200', 'logMFS']
mask = 'temporal_lobe_mask_grp_7T_test.nii.gz'
threshold = 0.001

for model in models:
#    scores = np.arctanh(apply_mask(glob.glob('MaThe/avg_maps/model_{}_p_adj_subj_*'.format(model)), mask_img=mask)).mean(axis=0)
#    mean_scores = scores.mean(axis=0)
#    t_values, p_values = ttest_1samp(scores, 0, axis=0)
#    corr_p_values = fdrcorrection0(p_values, alpha=0.05)
##    threshold = np.min(mean_scores[corr_p_values<0.05])
#    threshold = 0.001
#    fsf.save_map_avg('avg_unthresh', mean_scores, threshold=threshold, model=model)
#    mean_scores[corr_p_values>=0.05] = 0
#    display = fsf.plot_avg(mean_scores, threshold)
#    fsf.save_map_avg('avg', mean_scores, threshold=threshold, model=model)
#    display.savefig('mean_scores_model_{}.svg'.format(model))
#    display.savefig('mean_scores_model_{}.png'.format(model))
    for pc in xrange(1, 4):
        scores = np.arctanh(apply_mask(glob.glob('MaThe/avg_maps/model_{}_p_adj_pc_{}_subj_*'.format(model, pc)), mask_img=mask))
        mean_scores = scores.mean(axis=0)
        t_values, p_values = ttest_1samp(scores, 0, axis=0)
        corr_p_values = fdrcorrection0(p_values, alpha=0.05)
        mean_scores[corr_p_values>=0.05] = 0
        display = fsf.plot_avg(mean_scores, threshold, vmax=0.27)
        display.savefig('mean_scores_model_{}_pc_{}.svg'.format(model, pc))
        display.savefig('mean_scores_model_{}_pc_{}.png'.format(model, pc))
        fsf.save_map_avg('avg', mean_scores, threshold=threshold, model=model+'_pc_'+str(pc))
