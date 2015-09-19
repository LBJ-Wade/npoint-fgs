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


from planck_data_info import *


def cl_alm2d(alm1=None, alm2=None, lmax=100):
    """this function is just a test for make2d_alm/make2d_alm_square;
    it computes the cl from given alm2d that
    make2d_alm/make2d_alm_square returns, to make sure they match healpy result.
    """
    if alm2 is None:
        alm2 = alm1
    cl = np.zeros(lmax+1)
    ls = np.arange(lmax+1)
    for l in ls:
        ms = np.arange(-l,l+1)
        
        cl[l] += ((alm1[l][ms]*np.conjugate(alm2[l][ms])).real).sum()/(2.*l+1.)
    return cl



def check_Tlm2d(nu=100, lmax=300,experiment='planck',
                maskfield=2, source_maskfield=0,
                label_loc='lower right', xmax=None):
    if experiment=='planck':
        Imap_name = PLANCK_DATA_PATH+'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nu)
    Imap =hp.read_map(Imap_name)
    mask=hp.read_map(PLANCK_DATA_PATH+'HFI_Mask_GalPlane-apo0_2048_R2.00.fits',
                     field=maskfield)
    smask=hp.read_map(PLANCK_DATA_PATH+'HFI_Mask_PointSrc_2048_R2.00.fits',
                     field=source_maskfield)
    mask *= smask

    hdulist = fits.open(PLANCK_DATA_PATH+'HFI_RIMO_Beams-100pc_R2.00.fits')
    beam = hdulist[BEAM_INDEX['{}'.format(nu)]].data.NOMINAL[0][:lmax+1]
    
    tlm = get_Tlm(lmax=lmax, Imap=Imap, mask=mask,
                  healpy_format=False, recalc=True, div_beam=beam)
    tlm_hp = get_Tlm(lmax=lmax, Imap=Imap, mask=mask,
                  healpy_format=True, recalc=True, div_beam=beam)

    cl = cl_alm2d(alm1=tlm, lmax=lmax)
    l = np.arange(len(cl))
    cl_hp = hp.alm2cl(tlm_hp, lmax=lmax)
    l_hp = np.arange(len(cl_hp))

    clplanck = np.loadtxt(data_path + 'bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl')
    cl_planck = clplanck[:,1]
    l_planck = clplanck[:,0]

    pl.figure()
    pl.title('TT check')
    pl.plot(l, cl*l*(l+1)/2./np.pi*1e12, label='2d')
    pl.plot(l_hp,cl_hp*l_hp*(l_hp+1)/2./np.pi*1e12, label='healpy')
    pl.plot(l_planck, cl_planck, label='planck best fit')
    pl.legend(loc=label_loc)
    if xmax is None:
        pl.xlim(xmax=lmax)
    else:
        pl.xlim(xmax=xmax)



def check_EBlm2d(nu1=100,nu2=143, lmax=300,
                maskfield=2, source_maskfield=0,
                label_loc='lower right', xmax=None):
    
    map_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nu1)
    Q1,U1 =hp.read_map(data_path + map_name, field=(1,2))
    map_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nu2)
    Q2,U2 =hp.read_map(data_path + map_name, field=(1,2))
    mask=hp.read_map(data_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits',
                     field=maskfield)
    smask=hp.read_map(data_path + 'HFI_Mask_PointSrc_2048_R2.00.fits',
                     field=source_maskfield)
    mask *= smask

    hdulist = fits.open(data_path + 'HFI_RIMO_Beams-100pc_R2.00.fits')
    beam1 = hdulist[beam_index['{}P'.format(nu1)]].data.NOMINAL[0][:lmax+1]
    beam2 = hdulist[beam_index['{}P'.format(nu2)]].data.NOMINAL[0][:lmax+1]
    
    elm1,blm1 = get_ElmBlm(lmax=lmax, Qmap=Q1, Umap=U1, mask=mask,
                  healpy_format=False, recalc=True, div_beam=beam1)
    elm_hp1,blm_hp1 = get_ElmBlm(lmax=lmax, Qmap=Q1, Umap=U1, mask=mask,
                  healpy_format=True, recalc=True, div_beam=beam1)
    elm2,blm2 = get_ElmBlm(lmax=lmax, Qmap=Q2, Umap=U2, mask=mask,
                  healpy_format=False, recalc=True, div_beam=beam2)
    elm_hp2,blm_hp2 = get_ElmBlm(lmax=lmax, Qmap=Q2, Umap=U2, mask=mask,
                  healpy_format=True, recalc=True, div_beam=beam2)

    clee = cl_alm2d(alm1=elm1, alm2=elm2, lmax=lmax)
    clbb = cl_alm2d(alm1=blm1,alm2=blm2, lmax=lmax)
    l = np.arange(len(clee))
    clee_hp = hp.alm2cl(elm_hp1,elm_hp2, lmax=lmax)
    clbb_hp = hp.alm2cl(blm_hp1,blm_hp2, lmax=lmax)
    l_hp = np.arange(len(clee_hp))

    clplanck = np.loadtxt(data_path + 'bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl')
    clee_planck = clplanck[:,3]
    clbb_planck = clplanck[:,4]
    l_planck = clplanck[:,0]

    pl.figure()
    pl.title('EE check')
    pl.plot(l, clee*l*(l+1)/2./np.pi*1e12, label='2d')
    pl.plot(l,clee_hp*l_hp*(l_hp+1)/2./np.pi*1e12, label='healpy')
    pl.plot(l_planck, clee_planck, label='planck best fit')
    pl.legend(loc=label_loc)
    if xmax is None:
        pl.xlim(xmax=lmax)
    else:
        pl.xlim(xmax=xmax)

    pl.figure()
    pl.title('BB check')
    pl.plot(l, clbb*l*(l+1)/2./np.pi*1e12, label='2d')
    pl.plot(l_hp,clbb_hp*l_hp*(l_hp+1)/2./np.pi*1e12, label='healpy')
    pl.plot(l_planck, clbb_planck, label='planck best fit')
    pl.legend(loc=label_loc)
    if xmax is None:
        pl.xlim(xmax=lmax)
    else:
        pl.xlim(xmax=xmax)
    
def check_TE2d(nu=100, lmax=300,
                maskfield=2, source_maskfield=0,
                label_loc='lower right', xmax=None):
    
    map_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nu)
    I,Q,U =hp.read_map(data_path + map_name, field=(0,1,2))
    mask=hp.read_map(data_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits',
                     field=maskfield)
    smask=hp.read_map(data_path + 'HFI_Mask_PointSrc_2048_R2.00.fits',
                     field=source_maskfield)
    mask *= smask

    hdulist = fits.open(data_path + 'HFI_RIMO_Beams-100pc_R2.00.fits')
    beamP = hdulist[beam_index['{}P'.format(nu)]].data.NOMINAL[0][:lmax+1]
    beam = hdulist[beam_index['{}'.format(nu)]].data.NOMINAL[0][:lmax+1]

    #tlm = get_Tlm(lmax=lmax, Imap=I, mask=mask,
    #              healpy_format=False, recalc=True, div_beam=beam)
    #elm,blm = get_ElmBlm(lmax=lmax, Qmap=Q, Umap=U, mask=mask,
    #              healpy_format=False, recalc=True, div_beam=beamP)
    tlm_hp = get_Tlm(lmax=lmax, Imap=I, mask=mask,
                  healpy_format=True, recalc=True, div_beam=beam)
    elm_hp,blm_hp = get_ElmBlm(lmax=lmax, Qmap=Q, Umap=U, mask=mask,
                  healpy_format=True, recalc=True, div_beam=beamP)

    #cltt = cl_alm2d(tlm, lmax)
    #clee = cl_alm2d(elm, lmax)
    #clbb = cl_alm2d(blm, lmax)
    #l = np.arange(len(clee))
    clte_hp = hp.alm2cl(tlm_hp, elm_hp, lmax=lmax)
    #clee_hp = hp.alm2cl(elm_hp, lmax=lmax)
    #clbb_hp = hp.alm2cl(blm_hp, lmax=lmax)
    l_hp = np.arange(len(clte_hp))

    clplanck = np.loadtxt(data_path + 'bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl')
    clte_planck = clplanck[:,2]
    #clee_planck = clplanck[:,3]
    #clbb_planck = clplanck[:,4]
    l_planck = clplanck[:,0]

    pl.figure()
    pl.title('TE check')
    #pl.plot(l, clee*l*(l+1)/2./np.pi*1e12, label='2d')
    pl.plot(l_hp,clte_hp*l_hp*(l_hp+1)/2./np.pi*1e12, label='healpy')
    pl.plot(l_planck, clte_planck, label='planck best fit')
    pl.legend(loc=label_loc)
    if xmax is None:
        pl.xlim(xmax=lmax)
    else:
        pl.xlim(xmax=xmax)

         
def check_noise(nmaps=100, npix=50331648,frequency=100):

    mapname = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(frequency)
    covII, covIQ, covIU, covQQ, covQU, covUU = hp.read_map( data_path + mapname,
                                                            field=(4,5,6,7,8,9) )
    
    covii = np.zeros(npix)
    covqq = np.zeros(npix)
    covuu = np.zeros(npix)
    coviq = np.zeros(npix)
    coviu = np.zeros(npix)
    covqu = np.zeros(npix)
    
    for i in np.arange(nmaps):
        print '{}/{}...'.format(i+1,nmaps)
        I,Q,U = simulate_noisemap(npix=npix, frequency=frequency,save=False)

        covii += I*I / nmaps
        covqq += Q*Q / nmaps
        covuu += U*U / nmaps
        coviq += I*Q / nmaps
        coviu += I*U / nmaps
        covqu += U*Q / nmaps

    return covii,covqq,covuu,coviq,coviu,covqu
    hp.mollview(np.abs(covii - covII)/covII)
    pl.title('II')
    hp.mollview(np.abs(covqq - covQQ)/covQQ)
    pl.title('QQ')
    hp.mollview(np.abs(covuu - covUU)/covUU)
    pl.title('UU')
    hp.mollview(np.abs(coviq - covIQ)/covIQ)
    pl.title('IQ')
    hp.mollview(np.abs(coviu - covIU)/covIU)
    pl.title('IU')
    hp.mollview(np.abs(covqu - covQU)/covQU)
    pl.title('QU') 

    

def check_cl_sims(nmaps=1,lmax=1000,nside=2048,
                  read_file=False,
                  filename='testsky.fits',frequency=100,
                  beam=None, beamP=None, smear=True,
                  nonoise=False,
                  cl_file='bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl'):

    if read_file and os.path.exists(data_path + filename):
        Tmap, Qmap, Umap = hp.read_map(data_path + filename, field=(0,1,2))
    else:
        if nonoise:
            Tmap, Qmap, Umap = simulate_cmb(nside=nside, lmax=lmax,save=False,
                                        smear=smear, beam=beam, beamP=beamP,
                                        cl_file=cl_file)
        else:
            Tmap, Qmap, Umap = observe_cmb_sky(save=False, nside=nside, npix=None, lmax=3000,
                                                frequency=frequency, beam=beam, beamP=beamP,
                                                cl_file=cl_file)
            
    Tlm = hp.map2alm(Tmap, lmax=lmax)
    Elm,Blm = hp.map2alm_spin( (Qmap,Umap), 2, lmax=lmax )

    if smear:
        if (beam is None) or (beamP is None) :
            hdulist = fits.open(data_path + 'HFI_RIMO_Beams-100pc_R2.00.fits')
            beam = hdulist[beam_index['{}'.format(frequency)]].data.NOMINAL[0][:lmax+1]
            beamP = hdulist[beam_index['{}P'.format(frequency)]].data.NOMINAL[0][:lmax+1]
        hp.sphtfunc.almxfl(Tlm, 1./beam, inplace=True)
        hp.sphtfunc.almxfl(Elm, 1./beamP, inplace=True)
        hp.sphtfunc.almxfl(Blm, 1./beamP, inplace=True)
    
    ls = np.arange(lmax+1)
    factor = ls * ( ls + 1. ) / (2.*np.pi)
    
    cltt = hp.alm2cl(Tlm) * factor
    clee = hp.alm2cl(Elm) * factor
    clbb = hp.alm2cl(Blm) * factor
    clte = hp.alm2cl(Tlm, Elm) * factor
                                                  
    cl = get_theory_cls()
    ls_theory = cl[0][:lmax+1]
    factor_theory = ls_theory * ( ls_theory + 1. ) / (2.*np.pi)
    
    cltt_theory = cl[1][:lmax+1] * factor_theory
    clte_theory = cl[2][:lmax+1] * factor_theory
    clee_theory = cl[3][:lmax+1] * factor_theory
    clbb_theory = cl[4][:lmax+1] * factor_theory

    plt.figure()
    plt.plot(ls, cltt, label='sims')
    plt.plot(ls_theory, cltt_theory,label='theory TT')
    plt.legend()
    
    plt.figure()
    plt.plot(ls, clte, label='sims')
    plt.plot(ls_theory, clte_theory,label='theory TE')
    plt.legend()

    plt.figure()
    plt.plot(ls, clee, label='sims')
    plt.plot(ls_theory, clee_theory,label='theory EE')
    plt.legend()

    plt.figure()
    plt.plot(ls, clbb, label='sims')
    plt.plot(ls_theory, clbb_theory,label='theory BB')
    plt.legend()


