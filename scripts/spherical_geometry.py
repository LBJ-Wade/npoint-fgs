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


def get_Fk(J, filename='Fks_1000.txt'):
    Ftable = np.loadtxt(filename)
    Ftable = np.atleast_1d(Ftable)
    return Ftable[J]

def tabulate_Fks(Jmax=1000, filename='Fks_1000.txt'):
    if(os.path.exists(filename)):
        fin = open(filename, 'w')
        fin.close()
    
    for J in np.arange(Jmax):
        print J
        if J==0:
            Fres = 1
        else:
            Fres = np.sqrt(1-1./(2*J)) * get_Fk(J-1, filename)
            
        fout = open(filename, 'a')
        fout.write('{:.20f}\n'.format(Fres))
        fout.close()

@jit
def w3j000(L, l, lp, Fks):
    """this uses Kendrick Smith's recursion formula"""

    J = L + l + lp    
    if (J % 2 == 1) or (l + lp < L) or (np.abs(l - lp) > L):
        return 0.

    res = (-1)**(J/2) * (Fks[J/2 - L] * Fks[J/2 - l] * Fks[J/2 - lp]) / (Fks[J/2] * (J + 1.)**0.5)
    
    return res

def get_hs(filename, Fks_filename='Fks_1000.txt', lmax=100):

    if os.path.exists(filename):
        hs = np.load(filename, 'r')
        return hs

    Fks = np.loadtxt(Fks_filename)
    hs = calc_hs(Fks, lmin=lmin, lmax=lmax)
    np.save(filename, hs)
    return hs

@jit
def calc_hs(Fks, lmin=0, lmax=100):
    ls_array = np.arange(lmin, lmax+1)
    lenls = len(ls_array)
    res = np.zeros((lenls,lenls,lenls))
    for l1 in ls_array:
        for l2 in ls_array:
            for l3 in ls_array:
                res[l1,l2,l3] = w3j000(l1, l2, l3, Fks) * ((2.*l1+1)*(2.*l2+1)*(2.*l3+1)/(4.*np.pi))**0.5

    return res

