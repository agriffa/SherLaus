
import datetime
import time
import shutil
import os
import glob
import networkx as nx
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
project_dir = '/home/agriffa/DATA/COLLABORATION_ELEFTHERIOS/test_datasets'
ftxt = '/home/agriffa/Documents/GRADIENTSandPARCELLATIONS/grad_514.txt'
fbval = '/home/agriffa/Documents/GRADIENTSandPARCELLATIONS/protocollo_MRI_NAC/dsiq4half_from_data.bval'
fbvec = '/home/agriffa/Documents/GRADIENTSandPARCELLATIONS/protocollo_MRI_NAC/dsiq4half_from_data.bvec'

# SET INPUT PARAMETERS
# --------------------
# Voxel size at which you wish to perform deconvolution (preferentially isotropic)
new_zooms = (2., 2., 2.)
# Flip data along axiAl direction
flipy  =  True
# Set triangulated sphere for ODF sampling ('symmetric362', 'symmetric642' or 'symmetric724')
sphere = get_sphere('symmetric724')

# LOOP THROUGH ALL THE SUBJECTS IN PROJECT DIRECTORY 
# --------------------------------------------------
for subj in os.listdir(project_dir):
	if os.path.isdir(os.path.join(project_dir,subj)):

		# SUBJECTS SELECTION!!!
		if subj.find('PH0048') != -1:		

			# LOOP THROUGH ALL THE TIMEPOINTS OF THE CURRENT SUBJECTS
			# -------------------------------------------------------
			main_dir = os.path.join(project_dir, subj)
			print 'SUBJECT ' + subj

			for tp in os.listdir(main_dir):

				# TIME POINTS SELECTION
				if os.path.isdir(os.path.join(main_dir,tp)) and (tp.find('scan1') != -1):

					# SET INPUT DATA PATHS
					dname = os.path.join(main_dir,tp,'CMP')
					fimg = os.path.join(main_dir,tp,'NIFTI','DSI.nii.gz')
					fimgfirst = os.path.join(main_dir,tp,'NIFTI','DSI_first.nii.gz')
					fmask = os.path.join(main_dir,tp,'CMP','fs_output','HR__registered-TO-b0','fsmask_resampled.nii.gz')

					# RISLICE WM MASK LIKE DSI 
					print '... resample WM mask'
					origmask = os.path.join(main_dir,tp,'CMP/fs_output/HR__registered-TO-b0','fsmask_1mm.nii.gz')
					command = 'mri_convert -rl "' + fimgfirst + '" -rt nearest "' + origmask + '" "' + fmask + '"'
					#os.system(command)

					# Prepare data for deconvolution
					print('... prepare data for deconvolution')
					data, affine, zooms, bvals, bvecs, mask = prepare_q4half_257vol(fimg, ftxt, fbval, fbvec, fmask, flipy)

					# Create diffusion MR gradients
					gtab = gradient_table(bvals, bvecs)

					# Resample diffusion data and mask 
					print('... resample data')
					data, affine = resample(data, affine, zooms, new_zooms)
					mask, affine = resample(mask, affine, zooms, new_zooms, order=0)

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
					#save_pickle(os.path.join(main_dir,tp,'CMP','scalars','dsideconv.pkl'), dsipeaks)

					# Set headers for output images
					hdr_gfa = nib.load(fimg).get_header()
					hdr_gfa.set_data_shape(mask.shape)
					hdr_gfa.set_zooms(new_zooms)
					hdr_peaks = nib.load(fimg).get_header()
					hdr_peaks.set_data_shape((mask.shape[0],mask.shape[1],mask.shape[2],5))
					hdr_peaks.set_zooms((new_zooms[0],new_zooms[1],new_zooms[2],1.))
					hdr_odf = nib.load(fimg).get_header()
					hdr_odf.set_data_shape((mask.shape[0],mask.shape[1],mask.shape[2],sphere.x.shape[0]))
					hdr_odf.set_zooms((new_zooms[0],new_zooms[1],new_zooms[2],1.))			

					# Save output images
					name = os.path.join(main_dir,tp,'CMP','scalars','dsideconv_gfa.nii.gz')
					nib.save(nib.Nifti1Image(dsipeaks.gfa, affine, hdr_gfa), name)
					name = os.path.join(main_dir,tp,'CMP','scalars','dsideconv_peak_indices.nii.gz')
					nib.save(nib.Nifti1Image(dsipeaks.peak_indices, affine), name)
					name = os.path.join(main_dir,tp,'CMP','scalars','dsideconv_peak_values.nii.gz')
					nib.save(nib.Nifti1Image(dsipeaks.peak_values, affine), name)
					name = os.path.join(main_dir,tp,'CMP','scalars','dsideconv_odf.nii.gz')
					nib.save(nib.Nifti1Image(dsipeaks.odf, affine), name)					
					elapsed = time.time() - t
					print('    time %d' %elapsed)
					
					# Convert result to dir format
					t = time.time()
					print('... convert peaks to dir format')
					dsidir = prepare_dir(dsipeaks, sphere)
					name = os.path.join(main_dir,tp,'CMP','scalars','dsideconv_dir.nii.gz')
					nib.save(nib.Nifti1Image(dsidir, affine), name)
					elapsed = time.time() - t
					print('    time %d' %elapsed)

					# Classic DSI reconstruction
					t = time.time()
					print('... perform classic DSI reconstruction')
					dsmodel_dsi = DiffusionSpectrumModel(gtab)
					dsifit = dsmodel_dsi.fit(data)
					odfs = dsifit.odf(sphere)
					GFA = gfa(odfs)
					name = os.path.join(main_dir,tp,'CMP','scalars','dsidipy_gfa.nii.gz')
					nib.save(nib.Nifti1Image(GFA, affine), name)
					name = os.path.join(main_dir,tp,'CMP','scalars','dsidipy_odf.nii.gz')
					nib.save(nib.Nifti1Image(odfs, affine), name)						
					elapsed = time.time() - t
					print('    time %d' %elapsed)







