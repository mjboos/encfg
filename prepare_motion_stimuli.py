from __future__ import division
import mvpa2.suite as mvpa
from joblib import dump
import numpy as np
import sys
from sklearn.cross_validation import KFold

subj = sys.argv[1]
subj = int(subj)

spenc_dir = '/home/mboos/SpeechEncoding/'

motion = np.vstack([np.genfromtxt('/home/data/psyinf/forrest_gump/anondata/sub{0:03d}/BOLD/task001_run{1:03d}/bold_dico_moco.txt'.format(subj, run))
                    for run in xrange(1, 9)])

duration = np.array([902,882,876,976,924,878,1084,676])
print(np.sum(duration))
print(motion.shape)
# i did not kick out the first/last 4 samples per run yet
slice_nr_per_run = [dur/2 for dur in duration]

# use broadcasting to get indices to delete around the borders
idx_borders = np.cumsum(slice_nr_per_run[:-1])[:,np.newaxis] + \
              np.arange(-4,4)[np.newaxis,:]

motion = np.delete(motion, idx_borders, axis=0)

# and we're going to remove the last fmri slice
# since it does not correspond to a movie part anymore
motion = motion[:-1, :]

# shape of TR samples
motion = motion[3:]

dump(motion, spenc_dir+'MaThe/prepro/motion_{}_stimuli.pkl'.format(subj))
