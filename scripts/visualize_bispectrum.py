import numpy as np
import pylab as pl

import matplotlib
if __name__=='__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
import glob
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)
mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=18
mpl.rcParams['ytick.labelsize']=18

import healpy as hp
import os
import scipy
from numba import jit
from planck_data_info import *
    
def plot_bispectrum(b,slices=None,title=None,logplot=True,filename=None):

    
    if logplot:
        y = np.log10( np.abs(b) )
    else:
        y = b
  
    #ypositive = np.zeros(b.shape)
    #ynegative = np.zeros(b.shape)
    #ypositive[ b > 0. ] = np.log(np.abs( b[ b > 0. ] ))
    #ynegative[ b < 0. ] = -1.*np.log(np.abs( b[ b < 0. ] ))
    #y=ypositive+ynegative
    #return ypositive,ynegative
    
    if slices is None:
        plot_3D_bispectrum(y, title=title)
        if filename is not None:
            plt.savefig(filename)

    else:
        slices = np.atleast_1d(slices)
        for s in slices:
            plot_slice_bispectrum(y, s=s, title=title)
            if filename is not None:
                plt.savefig(filename)

def plot_slice_bispectrum(y, s=None, title='',
                          colormap='coolwarm'):
    if s is None:
        s=10

    pl.figure()
    if title is not None:
        title += ' (slice ind={})'.format(s)
    else:
        title = '$\ell_1$={}'.format(s)
    pl.title(title, fontsize=20)

    pl.imshow(y[s], cmap=colormap)
    pl.colorbar()

