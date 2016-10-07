# coding: utf-8
import mvpa2.suite as mvpa
import numpy as np
import os
import glob
import joblib
from sklearn.cross_validation import KFold
import sys
from nilearn.masking import unmask
from nibabel import save, load

def tmp_save_fmri(datapath, task, subj, model):
    dhandle = mvpa.OpenFMRIDataset(datapath)
    #mask_fname = os.path.join('/home','mboos','SpeechEncoding','temporal_lobe_mask_brain_subj' + str(subj) + 'bold.nii.gz')

    flavor = 'dico_bold7Tp1_to_subjbold7Tp1'
    group_brain_mask = '/home/mboos/SpeechEncoding/brainmask_group_template.nii.gz'
    mask_fname = os.path.join(datapath, 'sub{0:03d}'.format(subj), 'templates', 'bold7Tp1', 'brain_mask.nii.gz')
    #mask_fname = '/home/mboos/SpeechEncoding/masks/epi_subj_{}.nii.gz'.format(subj)
    scratch_path = '/home/data/scratch/mboos/prepro/tmp/'
    for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
        run_ds = dhandle.get_bold_run_dataset(subj,task,run_id,chunks=run_id-1,mask=mask_fname,flavor=flavor)
        filename = 'whole_brain_subj_{}_run_{}.nii.gz'.format(subj, run_id)
        tmp_path = scratch_path + filename
        save(unmask(run_ds.samples.astype('float32'), mask_fname), tmp_path)
        os.system('applywarp -i {0} -o {1} -r /home/data/psyinf/forrest_gump/anondata/templates/grpbold7Tp1/brain.nii.gz -w /home/data/psyinf/forrest_gump/anondata/sub{2:03}/templates/bold7Tp1/in_grpbold7Tp1/subj2tmpl_warp.nii.gz --interp=nn'.format(tmp_path, scratch_path+'group_'+filename,subj))
        os.remove(tmp_path)
        run_ds = mvpa.fmri_dataset(scratch_path+'group_'+filename, mask=group_brain_mask, chunks=run_id-1)
        mvpa.poly_detrend(run_ds, polyord=1)
        mvpa.zscore(run_ds)
        joblib.dump(run_ds.samples.astype('float32'),
                    '/home/data/scratch/mboos/prepro/tmp/whole_brain_subj_{}_run_{}.pkl'.format(subj, run_id))
        os.remove(scratch_path+'group_'+filename)
    return run_ds.samples.shape[1]

def split_fmri_memmap(file_mmaps, cv):
    '''
    Generator function for voxel-splits
    IN:
    file_mmaps - list of memmapped runs
    cv         - KFold object with splits
    OUT:
    fmri_data split by voxels specified in cv
    '''
    duration = np.array([902,882,876,976,924,878,1084,676])

    # i did not kick out the first/last 4 samples per run yet
    slice_nr_per_run = [dur/2 for dur in duration]

    # use broadcasting to get indices to delete around the borders
    idx_borders = np.cumsum(slice_nr_per_run[:-1])[:,np.newaxis] + \
                  np.arange(-4,4)[np.newaxis,:]

    for i, (_, voxels) in enumerate(cv):
        fmri_data = np.vstack([fl_mm[:, voxels] for fl_mm in file_mmaps])
        fmri_data = np.delete(fmri_data, idx_borders, axis=0)
        # and we're going to remove the last fmri slice
        # since it does not correspond to a movie part anymore
        fmri_data = fmri_data[:-1, :]

        # shape of TR samples
        fmri_data = fmri_data[3:]
        yield fmri_data

subj = int(sys.argv[1])

datapath = os.path.join('/home','data','psyinf','forrest_gump','anondata')
task = 1
model = 1

voxel_nr = tmp_save_fmri(datapath, task, subj, model)
cv = KFold(n=voxel_nr, n_folds=10)
file_mmaps = [joblib.load('/home/data/scratch/mboos/prepro/tmp/whole_brain_subj_{}_run_{}.pkl'.format(subj, run_id), mmap_mode='r') for run_id in xrange(1, 9)]

for i, fmri_data in enumerate(split_fmri_memmap(file_mmaps, cv)):
        joblib.dump(fmri_data,
             '/home/data/scratch/mboos/prepro/whole_fmri_subj_{}_split_{}.pkl'.format(subj, i),
             compress=3)
for run_id in xrange(1,9):
    os.remove('/home/data/scratch/mboos/prepro/tmp/whole_brain_subj_{}_run_{}.pkl'.format(subj, run_id))

