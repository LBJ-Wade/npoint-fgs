import numpy as np
import pylab as pl

import logging
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
from scipy.optimize import minimize

from planck_data_info import *
import process_fullsky as pf


def fit_power(l, Dl, guess=(200,-2.4),
              binned=True, nbins=10,
              model='power law', method='Nelder-Mead'):

    npts = len(l) // nbins
    if binned:
        x,y = bin_data(l, Dl, npts)
        #x = 10**x
    else:
        x = l
        y = Dl
    if model=='power law':
        pfit = minimize(neglnlike_powerlaw, guess, args=(x,y), method=method)

    return pfit

def bin_data(x, y, npts):
    """
    A modification of Ruth Angus' function for binning your data.
    Binning is sinning, of course, but if you want to get things
    set up quickly this can be very helpful!
    It takes your data: x, y
    npts (int) is the number of points per bin.
    """
    mod, nbins = len(x) % npts, len(x) / npts
    if mod != 0:
        x, y = x[:-mod], y[:-mod]
    xb, yb = [np.zeros(nbins) for i in range(2)]
    for i in range(npts):
        xb += x[::npts]
        yb += y[::npts]
        
        x, y = x[1:], y[1:]
    return xb/npts, yb/npts



def powerlaw(p, x):
    """
    p are parameters of the model
    
    """
    return p[0]*(x/80.)**(p[1]+2)
    
def lnlike_powerlaw(p, xdata, ydata, yerr=1.):
    mod = powerlaw(p, xdata)
    return (-0.5*(mod - ydata)**2 / yerr**2).sum()
    
def neglnlike_powerlaw(*args, **kwargs):
    return -lnlike_powerlaw(*args, **kwargs)
    
    
#define xdata, ydata you want to fit here (yerr too, if you want)


