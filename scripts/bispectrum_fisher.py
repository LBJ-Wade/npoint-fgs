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
import process_fullsky as pf
from spherical_geometry import get_hs

@jit(nopython=True)
def calc_b_cov_TEB(cltt, clee, clbb):

    N = len(cltt)
    res = np.empty((N,N,N))
    for l1 in np.arange(N):
        for l2 in np.arange(N):
            for l3 in np.arange(N):
                res[l1,l2,l3] = cltt[l1] * clee[l2] * clbb[l3]
    return res
    
def b_cov_T353_E143_B143(cl_file=pf.PLANCK_DATA_PATH+'bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl',lmax=100):

    Imap = hp.read_map(pf.PLANCK_DATA_PATH + 'HFI_SkyMap_353_2048_R2.02_full.fits')
    Tlm = hp.map2alm(Imap,lmax=lmax)
    cltt = hp.alm2cl(Tlm,lmax=lmax)

    mask = pf.get_planck_mask(psky=70)
    Qmap, Umap = hp.read_map(pf.PLANCK_DATA_PATH + 'HFI_SkyMap_143_2048_R2.02_full.fits',field=(1,2))
    Elm, Blm = hp.map2alm_spin( (Qmap*mask,Umap*mask), 2, lmax=lmax )
    clee = hp.alm2cl(Elm,lmax=lmax)
    clbb = hp.alm2cl(Blm,lmax=lmax)

    cov = calc_b_cov_TEB(cltt, clee, clbb)
    return cov

@jit#(nopython=True)
def fisher_T353_E143_B143(template,cov, lmax=100):
    res = 0.
    N = lmax+1
    res = np.zeros((N,N,N))
    for l1 in np.arange(N):
        for l2 in np.arange(N):
            for l3 in np.arange(N):
                if cov[l1,l2,l3] != 0.:
                    res += template[l1,l2,l3]/cov[l1,l2,l3]

    return 1./np.sqrt(res)
    

@jit
def offdiagonal(alm1, alm2, lmax=100):
    """
    
    """
    N = lmax + 1
    cc = np.zeros((N,N),dtype=complex)
    for l1 in range(N):
        m1s = np.arange(-l1, l1+1)
        for l2 in range(N):
            m2s = np.arange(-l2, l2+1)
            for m1 in m1s:
                for m2 in m2s:
                   cc[l1, l2] += alm1[l1,m1] * alm2[l2,m2] / len(m1s)/len(m2s)

    return cc


def b_cov_TEB(lmax=100,frequency=353):
    """this one is map-based"""

    Imap,Qmap, Umap = hp.read_map(pf.PLANCK_DATA_PATH + 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(frequency),field=(0,1,2))
    mask = pf.get_planck_mask()
    Tlm = hp.map2alm(Imap*mask,lmax=lmax)
    cltt = hp.alm2cl(Tlm,lmax=lmax)

    Elm, Blm = hp.map2alm_spin( (Qmap*mask,Umap*mask), 2, lmax=lmax )
    clee = hp.alm2cl(Elm,lmax=lmax)
    clbb = hp.alm2cl(Blm,lmax=lmax)

    #hs = get_hs(lmax=100)
    cov = calc_b_cov_TEB(cltt, clee, clbb)#/hs
    return cov



####
import foregrounds as fg
import spherical_geometry as sg
def b_cov_TEE(lmax=200,frequency=353, psky=70, apo=2):
    """this is theory-based"""

    ls, (cltt, clee, clbb, clte) = fg.get_theory_fg(f=frequency, lmax=lmax,
                                                  psky=psky, apo=apo)
   
    cov = calc_b_cov_TEB(cltt, clee, clee)
    hs = sg.get_hs(lmax=lmax)
    iszero = np.isclose(hs,0.)
    cov[iszero] = 0.
    return cov
