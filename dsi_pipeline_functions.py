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


# DEFINE DSI_PIPELINSE FUNCTIONS
# ------------------------------
def prepare_q4half_257vol(fimg, fbtext, fbval, fbvec, fmask, flipy):

    img = nib.load(fimg)
    data = img.get_data()
    affine = img.get_affine()
    zooms = img.get_header().get_zooms()[:3]

    # Read b-values from bval file
    bvals, temp = read_bvals_bvecs(fbval, fbvec)
    bvals = bvals[0:257]

    # Read Wedeen file grad514.txt (fbtxt)
    bmat = np.loadtxt(fbtext) 
    bvecs = bmat[0:257, 1:]
    # Normalize diffusion directions
    bvecs = bvecs / np.sqrt(np.sum(bvecs ** 2, axis=1))[:, None]
    bvecs[np.isnan(bvecs)] = 0

    if flipy:
        bvecs[:,1] = -bvecs[:,1]      

    # Delete empty DSI volumes
    data = data[:,:,:, 0:257]

    # Read mask
    img_mask = nib.load(fmask)
    mask = img_mask.get_data()

    return data, affine, zooms, bvals, bvecs, mask


def prepare_q5half(fimg, fbtext, fbval, fbvec, fmask, flipy):

    img = nib.load(fimg)
    data = img.get_data()
    affine = img.get_affine()
    zooms = img.get_header().get_zooms()[:3]

    bvals, temp = read_bvals_bvecs(fbval, fbvec)

    bmat = np.loadtxt(fbtext)
    bvecs = bmat[:, 1:]
    # Normalize bvec to unit norm
    bvecs = bvecs / np.sqrt(np.sum(bvecs ** 2, axis=1))[:, None]
    bvecs[np.isnan(bvecs)] = 0

    if flipy:
        bvecs[:,1] = -bvecs[:,1]

    img_mask = nib.load(fmask)
    mask = img_mask.get_data()

    return data, affine, zooms, bvals, bvecs, mask    


def prepare_dir(dsipeaks, sphere, flipx, flipy, flipz):

    values = dsipeaks.peak_values
    indices = dsipeaks.peak_indices
    xdir = sphere.x[indices]
    ydir = sphere.y[indices]
    zdir = sphere.z[indices]

    if len(values.shape) == 3:
        xi, xj, xl = values.shape
        dsidir = np.zeros((xi, xj, xl*4))
        dsidir[:,:, 0:xl*4:4] = values
        if flipx:
            dsidir[:,:, 1:xl*4:4] = -xdir
        else:
            dsidir[:,:, 1:xl*4:4] = xdir
        if flipy:    
            dsidir[:,:, 2:xl*4:4] = -ydir
        else:
            dsidir[:,:, 2:xl*4:4] = ydir
        if flipz:    
            dsidir[:,:, 3:xl*4:4] = -zdir
        else:
            dsidir[:,:, 3:xl*4:4] = zdir
    else:
        xi, xj, xk, xl = values.shape
        dsidir = np.zeros((xi, xj, xk, xl*4))
        dsidir[:,:,:, 0:xl*4:4] = values
        if flipx:
            dsidir[:,:,:, 1:xl*4:4] = -xdir
        else:
            dsidir[:,:,:, 1:xl*4:4] = xdir
        if flipy:    
            dsidir[:,:,:, 2:xl*4:4] = -ydir
        else:
            dsidir[:,:,:, 2:xl*4:4] = ydir
        if flipz:
            dsidir[:,:,:, 3:xl*4:4] = -zdir
        else:
            dsidir[:,:,:, 3:xl*4:4] = zdir
            
    return dsidir