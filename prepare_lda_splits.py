from __future__ import division
import mvpa2.suite as mvpa
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
import sys
from joblib import dump, load

spenc_dir = '/home/mboos/SpeechEncoding/'

duration = np.array([902,882,876,976,924,878,1084,676])

feature_list = load('lda_matrices.pkl')

ft_freq = feature_list[0].shape[1]

def reduce_MFS(a, b):
    '''deletes offset'''
    return np.concatenate([a[:-8*10], b[8*10:]], axis=0)

feature_list = [feat[:10*dur] for feat, dur in zip(feature_list, duration)]

features = reduce(reduce_MFS, feature_list)

features = features[:-10]

features = np.reshape(features, (-1, ft_freq*20))

strides = (features.strides[0],) + features.strides

# rolling window of length 4 samples
shape = (features.shape[0] - 4 + 1, 4, features.shape[1])

features = np.lib.stride_tricks.as_strided(features[::-1,:].copy(),
                                          shape=shape,
                                          strides=strides)[::-1, :, :]

features = np.reshape(features, (features.shape[0], -1))

# we kick out the most recent sample
features = features[:, :-20*ft_freq]


features = np.mean(np.reshape(features, (features.shape[0], 1, 60, ft_freq)), axis=-2)
features = np.reshape(features, (-1, 1*ft_freq))


features = StandardScaler().fit_transform(features)

dump(features,
     spenc_dir+'MaThe/prepro/lda_stimuli.pkl')
