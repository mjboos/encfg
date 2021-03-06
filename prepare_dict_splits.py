from __future__ import division
from joblib import load, dump
import numpy as np
from sklearn.preprocessing import StandardScaler

spenc_dir = '/home/mboos/SpeechEncoding/'

duration = np.array([902,882,876,976,924,878,1084,676])

patches = load(spenc_dir+'MaThe/'+\
                      'transformed_data/dict_learning_patches.pkl')

# keep only every 10th sample so there's no overlap between the patches
patches = patches[::10].copy()

# the length of the movie segments without the transition TRs 
# (like they are saved in patches)
movieseg_duration = duration[:]
movieseg_duration[0] -= 8
movieseg_duration[-1] -= 8
movieseg_duration[1:-1] -= 16

mvcs = np.cumsum(movieseg_duration)

# we need to remove the last last 2s of the second to last stimulus
to_delete = (mvcs[-2]-2)*10 + np.arange(20)

patches = np.delete(patches, to_delete, axis=0)
# shape of TR samples
# note: column ordering is now oldest --> newest in steps of 50
patches = np.reshape(patches, (-1, 200*20))

strides = (patches.strides[0],) + patches.strides

# rolling window of length 4 samples
shape = (patches.shape[0] - 4 + 1, 4, patches.shape[1])

patches = np.lib.stride_tricks.as_strided(patches[::-1,:].copy(),
                                          shape=shape,
                                          strides=strides)[::-1, :, :]

patches = np.reshape(patches, (patches.shape[0], -1))

# we kick out the most recent sample
patches = patches[:, :-4000]

patches = StandardScaler().fit_transform(patches)

dump(patches,
     spenc_dir+'MaThe/prepro/dict_stimuli.pkl')

