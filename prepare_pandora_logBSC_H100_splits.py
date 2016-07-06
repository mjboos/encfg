from __future__ import division
from joblib import load, dump
import numpy as np
import sys

spenc_dir = '/home/mboos/SpeechEncoding/'


patches = load(spenc_dir+'MaThe/'+\
                      'transformed_data/log_sparse_patches_pandora.pkl')


# note: column ordering is now oldest --> newest in steps of 50
patches = np.reshape(patches, (-1,3,100*20))
patches = np.reshape(patches[:,::-1,:], (-1, 3*100*20))


dump(patches,
     spenc_dir+'MaThe/prepro/logBSC_H100_pandora_stimuli.pkl')

