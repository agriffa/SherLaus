
import datetime
import time
import shutil
import os
import glob
import numpy as np
import scipy.io
import nibabel as nib
import sys
from dipy.viz import fvtk
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.viz import fvtk
from dipy.align.aniso2iso import resample
from dipy.data import get_sphere
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.odf import peaks_from_model
from dipy.io.pickles import save_pickle
from dipy.reconst.odf import gfa
from dsi_pipeline_functions import prepare_q4half_257vol
from dsi_pipeline_functions import prepare_dir
  

# DEFINE PATHS
# ------------
project_dir = '/home/eleftherios/Desktop/DSI_q4half_epfl/DSI/'
ftxt = '/home/eleftherios/Desktop/dsi_grad_514.txt'
fbval = project_dir + 'raw.bval'
fbvec = project_dir + 'raw.bvec'
fimg = project_dir + 'raw.nii.gz'
fmask = project_dir + 'raw_bet_mask.nii.gz'

# SET INPUT PARAMETERS
# --------------------
# Voxel size at which you wish to perform deconvolution (preferentially isotropic)
new_zooms = (2., 2., 2.)
# Flip data along axiAl direction
flipy  =  True
# Set triangulated sphere for ODF sampling ('symmetric362', 'symmetric642' or 'symmetric724')
sphere = get_sphere('symmetric724')

# Prepare data for deconvolution
print('... prepare data for deconvolution')
data, affine, zooms, bvals, bvecs, mask = prepare_q4half_257vol(fimg, ftxt, fbval, fbvec, fmask, flipy)

# Create diffusion MR gradients
gtab = gradient_table(bvals, bvecs)

# Resample diffusion data and mask 
print('... resample data')
data_new, affine_new = resample(data, affine, zooms, new_zooms)
mask_new, affine_new = resample(mask, affine, zooms, new_zooms, order=0)

# Deconvolution
t = time.time()
print('... perform deconvolution')
dsmodel = DiffusionSpectrumDeconvModel(gtab)
dsipeaks = peaks_from_model(model=dsmodel,
                            data=data,
                            sphere=sphere,
                            relative_peak_threshold=.5,
                            min_separation_angle=25,
                            mask=mask,
                            return_odf=True,
                            normalize_peaks=True)

main_dir = project_dir

name = os.path.join(main_dir,'dsideconv_gfa.nii.gz')
nib.save(nib.Nifti1Image(dsipeaks.gfa, affine_new), name)
name = os.path.join(main_dir,'dsideconv_peak_indices.nii.gz')
nib.save(nib.Nifti1Image(dsipeaks.peak_indices, affine_new), name)
name = os.path.join(main_dir,'dsideconv_peak_values.nii.gz')
nib.save(nib.Nifti1Image(dsipeaks.peak_values, affine_new), name)
name = os.path.join(main_dir,'dsideconv_odf.nii.gz')
nib.save(nib.Nifti1Image(dsipeaks.odf, affine_new), name)                 

elapsed = time.time() - t
print('    time %d' %elapsed)

# Convert result to dir format
t = time.time()
print('... convert peaks to dir format')
dsidir = prepare_dir(dsipeaks, sphere, 0, 0, 1)
name = os.path.join(main_dir,'dsideconv_dir.nii.gz')
nib.save(nib.Nifti1Image(dsidir, affine_new), name)
elapsed = time.time() - t
print('    time %d' %elapsed)

# Classic DSI reconstruction
t = time.time()
print('... perform classic DSI reconstruction')
dsmodel_dsi = DiffusionSpectrumModel(gtab)
dsifit = dsmodel_dsi.fit(data)
odfs = dsifit.odf(sphere)
GFA = gfa(odfs)
name = os.path.join(main_dir,'dsidipy_gfa.nii.gz')
nib.save(nib.Nifti1Image(GFA, affine_new), name)
name = os.path.join(main_dir,'dsidipy_odf.nii.gz')
nib.save(nib.Nifti1Image(odfs, affine_new), name)						
elapsed = time.time() - t
print('    time %d' %elapsed)

