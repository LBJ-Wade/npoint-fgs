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

try:
    PLANCK_DATA_PATH = os.environ['PLANCK_DATA_PATH']
except KeyError:
    logging.warning('PLANCK_DATA_PATH environment variable not defined, I will not be able to find input maps!')

try:
    FGS_PATH = os.environ['FGS_PATH']
except KeyError:
    logging.warning('FGS_PATH environment variable not defined, defaulting to current working directory.')
    FGS_PATH = os.getcwd()

FGS_SIM_PATH = FGS_PATH + 'sims/'
FGS_RESULTS_PATH = FGS_PATH + 'results/'

if not os.path.exists(FGS_SIMS_PATH):
    os.mkdir(FGS_SIMS_PATH)
if not os.path.exists(FGS_RESULTS_PATH):
    os.mkdir(FGS_RESULTS_PATH)
    



def make2d_alm(alm, lmax, ls, ms):
    
    alm2d = []
    for l in range(lmax+1):
        ms2d = np.arange(-l,l+1)
        alm2d.append(np.zeros(2*l + 1, dtype=complex))
        l_inds = ls==l
        for m in ms2d:
            m_inds = ms==np.abs(m)
            ind = m_inds & l_inds
            if m < 0:
                alm2d[l][m] = (-1.)**m * np.conjugate(alm[ind][0])
            else:
                alm2d[l][m] = alm[ind][0]

    return alm2d
        
@jit#(nopython=True)
def make2d_alm_square(alm, lmax, ls, ms):
    alm2d = np.zeros( ( lmax + 1, 2*lmax + 1), dtype=complex )
    for l in range(lmax+1):
        ms2d = np.arange(-l,l+1)

        l_inds = ls==l
        for m in ms2d:
            m_inds = ms==np.abs(m)
            ind = m_inds & l_inds
            if m < 0:
                alm2d[l][m] = (-1.)**m * np.conjugate(alm[ind][0])
            else:
                alm2d[l][m] = alm[ind][0]

    return alm2d


def calc_alm(Imap, Qmap, Umap, mask=None,
               lmax=100,add_beam=None,div_beam=None,
               healpy_format=False,
               filtermap=False, l0=None,
               save=False):
    """computes alms, given
    maps and a mask, corrected for sqrt(fsky), optionally corrected for beams
    """
    fsky = mask.sum() / len(mask)
    Tlm = hp.map2alm(Imap * mask, lmax=lmax) / np.sqrt(fsky)
    Elm, Blm = hp.map2alm_spin( (Qmap*mask,Umap*mask), 2, lmax=lmax )
    Elm /= np.sqrt(fsky)
    Blm /= np.sqrt(fsky)
    if add_beam is not None:
        hp.sphtfunc.almxfl(Tlm, add_beam, inplace=True)
        hp.sphtfunc.almxfl(Elm, add_beam, inplace=True)
        hp.sphtfunc.almxfl(Blm, add_beam, inplace=True)
    if div_beam is not None:
        hp.sphtfunc.almxfl(Tlm, 1./div_beam, inplace=True)
        hp.sphtfunc.almxfl(Elm, 1./div_beam, inplace=True)
        hp.sphtfunc.almxfl(Blm, 1./div_beam, inplace=True)
    if not healpy_format:
        ls, ms = hp.sphtfunc.Alm.getlm(lmax, np.arange(len(alm)))
        Tlm = make2d_alm_square(Tlm, lmax, ls, ms)
        Elm = make2d_alm_square(Elm, lmax, ls, ms)
        Blm = make2d_alm_square(Blm, lmax, ls, ms)

    return Tlm,Elm,Blm



def get_cholesky(frequency=100, mapname=None,
                 save=True,rewrite=False):

    if mapname is None:
        mapname = PLANCK_DATA_PATH + 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(frequency)
    newname = mapname[:-5] + '_cholesky.npy'
    
    if os.path.exists(data_path + newname) and not rewrite:
        L = np.load(data_path + newname, 'r')
        print 'found it! ({})'.format(data_path + newname)
        return L

    covII, covIQ, covIU, covQQ, covQU, covUU = hp.read_map( data_path + mapname,
                                                            field=(4,5,6,7,8,9) )
    npix = len(covII)
    L = calc_cholesky_IQU(covII, covIQ, covIU, covQQ, covQU, covUU, npix)
    if save:
        np.save(newname, L)
    return L

@jit(nopython=True)
def calc_cholesky_IQU(covII, covIQ, covIU, covQQ, covQU, covUU, npix):

    L = np.zeros((npix,3,3))
    for i in np.arange(npix):
        L[i,0,0] = np.sqrt(covII[i])
        L[i,1,0] = covIQ[i] / L[i,0,0]
        L[i,2,0] = covIU[i] / L[i,0,0]
        L[i,1,1] = np.sqrt(covQQ[i] - L[i,1,0]**2)
        L[i,2,1] = (covQU[i] - L[i,2,0] * L[i,1,0]) / L[i,1,1]
        L[i,2,2] = np.sqrt(covUU[i] - (L[i,2,0]**2 + L[i,2,1]**2) )
        

    return L
         
        
def simulate_noise(npix=50331648, frequency=100, experiment='planck',
                      save=True, filename=FGS_SIM_PATH+'noisemap.fits'):

    I = np.random.standard_normal(npix)
    Q = np.random.standard_normal(npix)
    U = np.random.standard_normal(npix)
    if experiment=='planck' or experiment=='Planck' or experiment=='PLANCK':
        L = np.load(PLANCK_DATA_PATH + 'HFI_SkyMap_{}_2048_R2.02_full_cholesky.npy'.format(frequency), 'r')
    I, Q, U = L_dot_rand_map(L,I,Q,U,npix)
    if save:
        hp.write_map(filename,[I,Q,U])
    return I, Q, U

@jit(nopython=True)
def L_dot_rand_map(L,rand_I,rand_Q,rand_U,npix):
    Imap = np.zeros(npix)
    Qmap = np.zeros(npix)
    Umap = np.zeros(npix)
    for i in np.arange(npix):
        Imap[i] = L[i][0,0] * rand_I[i]
        Qmap[i] = L[i][1,0] * rand_I[i] + L[i][1,1] * rand_Q[i]
        Umap[i] = L[i][2,0] * rand_I[i] + L[i][2,1] * rand_Q[i] + L[i][2,2] * rand_U[i]
    return Imap, Qmap, Umap

def simulate_cmb_alms(almfile=FGS_SIM_PATH+'cmb_alms/alm.npy', nside=2048, lmax=4000,
                        cl_file=PLANCK_DATA_PATH+'bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl'):
        
    ls, cltt, clte, clee, clbb = get_theory_cls(lmax=lmax, cl_file=cl_file)
    Tlm, Elm, Blm = hp.synalm( (cltt, clee, clbb, clte), new=True, lmax=lmax)
    np.save(almfile, [Tlm,Elm,Blm])
    return Tlm, Elm, Blm

def simulate_cmb_map(almfile=FGS_SIM_PATH+'cmb_alms/simalm.npy', nside=2048, lmax=3000,
                 frequency=100,smear=False,
                 beam=None, beamP=None,
                 save=False, filename=FGS_SIM_PATH+'testcmb.fits',
                 beam_file=PLANCK_DATA_PATH+'HFI_RIMO_Beams-100pc_R2.00.fits'):
        
    
    Tlm, Elm, Blm = np.load(almfile)
    Tlm = Tlm[:lmax+1]
    Elm = Elm[:lmax+1]
    Blm = Blm[:lmax+1]
    
    if smear:
        if (beam is None) or (beamP is None) :
            hdulist = fits.open(beam_file)
            beam = hdulist[BEAM_INDEX['{}'.format(frequency)]].data.NOMINAL[0][:lmax+1]
            beamP = hdulist[BEAM_INDEX['{}P'.format(frequency)]].data.NOMINAL[0][:lmax+1]
        hp.sphtfunc.almxfl(Tlm, beam, inplace=True)
        hp.sphtfunc.almxfl(Elm, beamP, inplace=True)
        hp.sphtfunc.almxfl(Blm, beamP, inplace=True)

    Tmap = hp.alm2map( Tlm, nside )
    Qmap, Umap = hp.alm2map_spin( (Elm, Blm), nside, 2, lmax=lmax)

    if save:
        hp.write_map([Tmap,Qmap,Umap], filename)
    return Tmap, Qmap, Umap

def get_theory_cls(lmax=3000,
                   cl_file=PLANCK_DATA_PATH+'bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl'):
    """ Read theory cls, remove factor of l*(l+1)/2pi,
        convert to uK_CMB^2 units,
        and pad with zeros, such that l starts at 0.
    """

    cl = np.loadtxt(cl_file)
    ls = np.arange(lmax+1)

    factor = cl[:lmax-1, 0]*(cl[:lmax-1, 0] + 1.) / (2.*np.pi) * 1.e12
    cltt = cl[:lmax-1, 1] / factor
    clte = cl[:lmax-1, 2] / factor
    clee = cl[:lmax-1, 3] / factor
    clbb = cl[:lmax-1, 4] / factor
    
    cltt = np.concatenate((np.array([0,0]),cltt))
    clte = np.concatenate((np.array([0,0]),clte))
    clee = np.concatenate((np.array([0,0]),clee))
    clbb = np.concatenate((np.array([0,0]),clbb))

    return ls, cltt, clte, clee, clbb


def simulate_observed_cmb(return_components=False,
                          save=False, filename=FGS_SIM_PATH+'cmbsky.fits',
                            nside=2048, npix=None, lmax=3000,experiment='planck',
                            frequency=100, beam=None, beamP=None,smear=True,
                            almfile=PLANCK_SIM_PATH+'cmb_sims/simalm.npy'):

    if npix is None:
        npix = hp.nside2npix(nside)
    if nside is None:
        nside = hp.npix2nside(npix)
        
    Tcmb, Qcmb, Ucmb = simulate_cmb_map(nside=nside, lmax=lmax,
                                frequency=frequency,smear=smear,
                                nomap=False, save=False,
                                beam=beam, beamP=beamP,
                                almfile=almfile)
    Tnoise, Qnoise, Unoise = simulate_noise(npix=npix,experiment=experiment,
                                            frequency=frequency,
                                            save=False)

    if save:
        hp.write_map(filename, [Tcmb+Tnoise, Qcmb+Qnoise, Ucmb+Unoise])

    if return_components:
        return Tcmb,Tnoise, Qcmb,Qnoise, Ucmb,Unoise
    return Tcmb+Tnoise, Qcmb+Qnoise, Ucmb+Unoise



def get_planck_mask(mask_percentage=60,
             mask_sources=True,
             apodization='0',
             smask_name='HFI_Mask_PointSrc_2048_R2.00.fits'):

    field = MASK_FIELD[mask_percentage]
    mask = hp.read_map(PLANCK_DATA_PATH + 'HFI_Mask_GalPlane-apo{}_2048_R2.00.fits'.format(apodization),
                       field=field)
    if mask_sources:
        smask = hp.read_map(PLANCK_DATA_PATH + smask_name)
        mask *= smask

    return mask
        
