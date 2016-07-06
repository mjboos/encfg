import joblib
import numpy as np
import sys

subj, model = sys.argv[1:]
spenc_dir = '/home/mboos/SpeechEncoding/'
preds = [joblib.load(spenc_dir+'MaThe/predictions/{}_subj_{}_split_{}.pkl'.format(model, subj, i)) for i in xrange(10)]

preds = np.vstack([np.hstack([preds[i][j] for i in xrange(10)]).astype('float32') for j in xrange(8)])

joblib.dump(preds, spenc_dir+'MaThe/predictions/{}_subj_{}_all.pkl'.format(model, subj), compress=3)


