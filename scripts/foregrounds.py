import numpy as np
import pylab as pl

import pickle
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

def bin_data_errs(x, y, yerr, npts):
    """
    Ruth Angus' function for binning your data.
    Binning is sinning, of course, but if you want to get things
    set up quickly this can be very helpful!
    It takes your data: x, y, yerr
    npts (int) is the number of points per bin.
    """
    mod, nbins = len(x) % npts, len(x) / npts
    if mod != 0:
        x, y, yerr = x[:-mod], y[:-mod], yerr[:-mod]
    xb, yb, yerrb = [np.zeros(nbins) for i in range(3)]
    for i in range(npts):
        xb += x[::npts]
        yb += y[::npts]
        yerrb += yerr[::npts]**2
        x, y, yerr = x[1:], y[1:], yerr[1:]
    return xb/npts, yb/npts, yerrb**.5/npts

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

def powerlaw_cl(p, x):
    """
    p are parameters of the model
    
    """
    norm_corr = x * (x + 1.) /2./np.pi *1.e12
    res = np.zeros(len(x))
    res[x!=0.] = powerlaw_dl(p, x[x!=0.]) / norm_corr[x!=0.]
    return res

def powerlaw_dl(p, x):
    """
    p are parameters of the model
    
    """
    return p[0]*(x/80.)**(p[1]+2)
    
def lnlike_powerlaw(p, xdata, ydata, yerr=1.):
    mod = powerlaw_dl(p, xdata)
    return (-0.5*(mod - ydata)**2 / yerr**2).sum()
    
def neglnlike_powerlaw(*args, **kwargs):
    return -lnlike_powerlaw(*args, **kwargs)
    

def fit_fg_powerlaw(frequency=353,lmin=40,lmax=600,
                     put_mask=True,
                     psky=70,
                     mask_sources=True,
                     apodization=0,
                     guess=(200,-2.4),
                     nbins=25,
                     fsky_correction=True):

    ls, TT, EE, BB, TE, TB, EB = measure_dlcl(mode='dl', frequency=frequency,
                                                lmax=lmax,lmin=lmin,
                                                put_mask=put_mask,
                                                psky=psky,
                                                mask_sources=mask_sources,
                                                apodization=apodization,
                                                fsky_correction=fsky_correction)

    sol = []
    PSa = [TT, EE, BB, TE, TB, EB]
    fit_inds = (ls<500)==(ls>60) #planck collab "fitting window"
    for PS in PSa:
        pfit = fit_power(ls[fit_inds], PS[fit_inds], guess=guess)
        sol.append(pfit.x)

    PS_names = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
    fg_dict = {}
    for i,PS_name in enumerate(PS_names):
            fg_dict[PS_name] = sol[i]
  
    return fg_dict, fg_dict['EE'], fg_dict['BB'], ls, EE, BB

def all_fg_powerlaws(frequency=353,lmin=40,lmax=600,
                     put_mask=True,
                     pskys=[20,40,60,70,80],
                     mask_sources=True,
                     apodization=0,
                     plots=True,
                     guess=(200,-2.4),
                     nbins=25,filetag='',
                     fsky_correction=True):

    tag = '_apo{}{}'.format(apodization,filetag)

    pEEs = []
    pBBs = []
    dlees = []
    dlbbs = []
    for psky in pskys:
        fg_dict, pEE, pBB, ls, EE, BB = fit_fg_powerlaw(frequency=frequency,
                                                            lmin=lmin,lmax=lmax,
                                                            put_mask=put_mask,
                                                            psky=psky,
                                                            mask_sources=mask_sources,
                                                            apodization=apodization,
                                                            guess=guess,
                                                            nbins=nbins,
                                                            fsky_correction=fsky_correction)
        filename = pf.FGS_RESULTS_PATH + "fg_power_{}GHz_{}psky{}.pkl".format(frequency,psky,tag)
        pickle.dump( fg_dict, open( filename, "wb" ) )

        l,dlee = bin_data(ls, EE, nbins)
        l,dlbb = bin_data(ls, BB, nbins)
        
        pEEs.append(pEE)
        pBBs.append(pBB)
        dlees.append(dlee)
        dlbbs.append(dlbb)

    plot_fg_power(pEEs, pBBs, l, dlees, dlbbs, pskys,
                  lmin=lmin, lmax=lmax, frequency=frequency,tag=tag)

    plot_alphas(pEEs,pBBs,pskys,tag=tag)

def plot_alphas(pEEs, pBBs, pskys, tag=''):
    plt.figure()
    xmin = 0.2
    xmax = 0.9
    ymax = -2.
    ymin = -2.8
    xs = np.linspace(xmin, xmax, 100)
    yEE = xs/xs * (-2.4)
    yBB = xs/xs * (-2.45)
    plt.plot(xs, yEE, '--',color='red')
    plt.plot(xs, yBB, '--', color='blue')
    for i,psky in enumerate(pskys):
        plt.plot(psky/100.,pEEs[i][1],'o',ms=8,color='red')
        plt.plot(psky/100.,pBBs[i][1],'o',ms=8,color='blue')

    plt.xlabel(r'$f_\text{sky}$',fontsize=20)
    plt.ylabel(r'$\alpha_{EE,BB}$',fontsize=20)
    plt.xlim(xmin=xmin,xmax=xmax)
    plt.ylim(ymin=ymin,ymax=ymax)
    plt.savefig(pf.FGS_RESULTS_PATH + 'planck_alphas{}.pdf'.format(tag))
        

def plot_fg_power(pEEs, pBBs, l, dlees, dlbbs, pskys,
                  lmin=40, lmax=600, frequency=353,tag=''):

    
    ls = np.arange(lmin, lmax+1)
    title = 'percent sky (from bottom):'
    for ps in pskys:
        title += ' {}'.format(ps)

    plt.figure()
    for i,p in enumerate(pEEs):
        plt.loglog(l, dlees[i],'o', color='black')
        plt.loglog(ls, powerlaw_dl(p,ls), color='Maroon')

    plt.xlim(xmin=lmin,xmax=lmax)
    plt.legend(loc='bottom left',fontsize=20, frameon=False)
    plt.xlabel('$\ell$',fontsize=20)
    plt.ylabel('$\mathcal{D}^{EE}_\ell [\mu K^2_{CMB}]$',fontsize=20)
    plt.title(title, fontsize=15)
    
    plt.savefig(pf.FGS_RESULTS_PATH + 'EE_foreground_power_{}GHz{}.pdf'.format(frequency,tag))


    plt.figure()
    for i,p in enumerate(pBBs):
        plt.loglog(l, dlbbs[i],'o', color='black')
        plt.loglog(ls, powerlaw_dl(p,ls), color='Blue')

    plt.xlim(xmin=lmin,xmax=lmax)
    plt.legend(loc='bottom left',fontsize=20, frameon=False)
    plt.xlabel('$\ell$',fontsize=20)
    plt.ylabel('$\mathcal{D}^{BB}_\ell [\mu K^2_{CMB}]$',fontsize=20)
    plt.title(title, fontsize=15)
    
    plt.savefig(pf.FGS_RESULTS_PATH + 'BB_foreground_power_{}GHz{}.pdf'.format(frequency,tag))

def print_beam_files(f=353):
    old_beam_file = pf.PLANCK_DATA_PATH + 'HFI_RIMO_Beams-100pc_R2.00.fits'
    hdulist = pf.fits.open(old_beam_file)
    beam_file = pf.PLANCK_DATA_PATH + 'beam_{}.txt'.format(f)
    bf = open(beam_file, 'w')
    beam_fileP = pf.PLANCK_DATA_PATH + 'beamP_{}.txt'.format(f)
    bfP = open(beam_fileP, 'w')

    beam = hdulist[pf.BEAM_INDEX['{}'.format(f)]].data.NOMINAL[0]
    beamP = hdulist[pf.BEAM_INDEX['{}P'.format(f)]].data.NOMINAL[0]
    ls = np.arange(len(beam))

    for l in ls:
        bf.write('{}    {:.12f}\n'.format(l,beam[l]))
        bfP.write('{}    {:.12f}\n'.format(l,beamP[l]))
    bf.close()
    bfP.close()


def rewrite_mask_files(pskys=[40,60,70,80],
                       mask_sources=True,
                       apodization=2,
                       smask_name='HFI_Mask_PointSrc_2048_R2.00.fits'):
    for psky in pskys:
        maskfile = pf.PLANCK_DATA_PATH + 'mask_psky{}_apo{}.fits'.format(psky,apodization)
        mask = pf.get_planck_mask(psky=psky,
                              mask_sources=mask_sources,
                              apodization=apodization,
                              smask_name=smask_name)

        hp.write_map(maskfile, mask)

import subprocess
    
def spice_cl(lmax=600, f=353,
             psky=70, apo=0,
             subdipole='NO',
             decouple='YES',
             apodizesigma='NO'):

    clfile = pf.FGS_RESULTS_PATH + 'cl_spice_lmax{}_{}GHz_apo{}_psky{}.txt'.format(lmax,f,apo,psky)
             
    mapfile = pf.PLANCK_DATA_PATH + 'HFI_SkyMap_{}_2048_R2.02_halfmission-1.fits'.format(f)
    mapfile2 = pf.PLANCK_DATA_PATH + 'HFI_SkyMap_{}_2048_R2.02_halfmission-2.fits'.format(f)
    maskfile = pf.PLANCK_DATA_PATH + 'mask_psky{}_apo{}.fits'.format(psky,apo)
    beam_file = pf.PLANCK_DATA_PATH + 'beam_{}.txt'.format(f)

    command = 'spice -nlmax {} -beam_file {} -beam_file2 {} -clfile {} -mapfile {} -mapfile2 {} -maskfile {} -maskfile2 {} -maskfilep {} -maskfilep2 {} -polarization YES -subdipole {} -decouple {} -kernelsfileout YES -apodizesigma {}'.format(lmax, beam_file, beam_file, clfile, mapfile, mapfile2, maskfile, maskfile, maskfile, maskfile, subdipole,decouple,apodizesigma)
    #print command
    subprocess.call(command, shell=True)


def measure_dlcl(mode='dl',frequency=353,
                 mask=None,
              lmax=600,lmin=40,
              put_mask=True,
              psky=70,
              mask_sources=True,
              apodization=2,
              beam=None,beamP=None,
              Imap=None,Qmap=None,Umap=None,
              Imap2=None,Qmap2=None,Umap2=None,
              fsky_correction=True):

    """
       If mode=='dl', returns D quantity = Cl *ls*(ls+1)/2./np.pi*1e12 [units: uK_CMB^2]
    """

    
    if put_mask:
        if mask is None:
            print 'reading masks...'
            mask = pf.get_planck_mask(psky=psky,
                                mask_sources=mask_sources,
                                apodization=apodization)
    else:
        mask = map.copy()*0. + 1.

    if (beam is None) or (beamP is None):
        print 'reading beams...'
        beam_file = pf.PLANCK_DATA_PATH+'HFI_RIMO_Beams-100pc_R2.00.fits'
        hdulist = pf.fits.open(beam_file)
        beam = hdulist[pf.BEAM_INDEX['{}'.format(frequency)]].data.NOMINAL[0][:lmax+1]
        beamP = hdulist[pf.BEAM_INDEX['{}P'.format(frequency)]].data.NOMINAL[0][:lmax+1]
        beam = beam[lmin:lmax+1]
        beamP = beamP[lmin:lmax+1]

    fsky = mask.sum() / len(mask)
    ls = np.arange(lmin, lmax+1)
    if mode == 'dl':
        factor =  ls * (ls+1) / (2.*np.pi) * 1e12 / fsky
    if mode == 'cl':
        factor =  1. / fsky

    if fsky_correction:
        fcfilename = pf.FGS_RESULTS_PATH + 'fskycorr_fg_psky{}_apo{}_lmax1000_TT_EE_BB.npy'.format(psky,apodization)
        if os.path.exists(fcfilename):
            fcorr = np.load(fcfilename)
            fcorr_TT = fcorr[0][lmin:lmax+1]
            fcorr_EE = fcorr[1][lmin:lmax+1]
            fcorr_BB = fcorr[2][lmin:lmax+1]
        else:
            fcorr = None
    else:
        fcorr = None
    
    
    if (Imap is None) or (Qmap is None) or (Umap is None):
        print 'reading maps...'
        mapname1 = pf.PLANCK_DATA_PATH + 'HFI_SkyMap_{}_2048_R2.02_halfmission-1.fits'.format(frequency)
        mapname2 = pf.PLANCK_DATA_PATH + 'HFI_SkyMap_{}_2048_R2.02_halfmission-2.fits'.format(frequency)
        Imap = hp.read_map(mapname1,field=0)       
        Imap2 = hp.read_map(mapname2,field=0)
        Qmap, Umap = hp.read_map(mapname1,field=(1,2))
        Qmap2, Umap2 = hp.read_map(mapname2,field=(1,2))
   
    Tlm1 = hp.map2alm(Imap*mask, lmax=lmax)
    Elm1, Blm1 = hp.map2alm_spin( (Qmap*mask, Umap*mask), 2, lmax=lmax )
    if (Imap2 is None) or (Qmap2 is None) or (Umap2 is None):
        Tlm2 = Tlm1
        Elm2 = Elm1
        Blm2 = Blm1       
    else:
        Tlm2 = hp.map2alm(Imap2*mask, lmax=lmax)
        Elm2, Blm2 = hp.map2alm_spin( (Qmap2*mask,Umap2*mask), 2, lmax=lmax )


    TT = hp.alm2cl(Tlm1, Tlm2)
    EE = hp.alm2cl(Elm1, Elm2)
    BB = hp.alm2cl(Blm1, Blm2)

    EE = EE[lmin:] * factor / beamP**2
    TT = TT[lmin:] * factor / beam**2
    BB = BB[lmin:] * factor / beamP**2
    
    TE = hp.alm2cl(Tlm1, Elm2)
    EB = hp.alm2cl(Blm1, Elm2)
    TB = hp.alm2cl(Blm1, Tlm2)
    TE = TE[lmin:] * factor / beam / beamP
    TB = TB[lmin:] * factor / beam / beamP
    EB = EB[lmin:] * factor / beamP**2

    if fcorr is not None:
        TT *= fcorr_TT
        EE *= fcorr_EE
        BB *= fcorr_BB


    return ls, TT, EE, BB, TE, TB, EB

def fg_cl_thfit(f=353, lmax=1000, psky=70, apo=2):

    dic = pickle.load( open( pf.FGS_RESULTS_PATH + 'fg_power_{}GHz_{}psky_apo{}.pkl'.format(f,psky,apo), 'rb' ) )

    pTT = dic['TT']
    pEE = dic['EE']
    pBB = dic['BB']
    pTE = dic['TE']

    ls = np.arange(lmax+1)
    cltt = powerlaw_cl(pTT, ls)
    clee = powerlaw_cl(pEE, ls)
    clbb = powerlaw_cl(pBB, ls)
    clte = powerlaw_cl(pTE, ls)

    return ls, (cltt, clee, clbb, clte)


def calibrate_fsky(mode, nsim=1,
                   smear=True,apo=2,
                    psky=70, f=353, 
                    nside=2048,lmax=1000,
                    visual_check=False,
                    beam_file=pf.PLANCK_DATA_PATH+'HFI_RIMO_Beams-100pc_R2.00.fits',
                    mask_sources=True, put_mask=True):

    """No noise treatment here"""

    fsky_correction = []
    TTm = np.zeros(lmax + 1)
    EEm = np.zeros(lmax + 1)
    BBm = np.zeros(lmax + 1)
    
    if put_mask:
        print 'reading mask...'
        mask = pf.get_planck_mask(psky=psky,
                                mask_sources=mask_sources,
                                apodization=apo)
    else:
        mask = None
    print 'reading beams...'
    hdulist = pf.fits.open(beam_file)
    beam = hdulist[pf.BEAM_INDEX['{}'.format(f)]].data.NOMINAL[0][:lmax+1]
    beamP = hdulist[pf.BEAM_INDEX['{}P'.format(f)]].data.NOMINAL[0][:lmax+1]
    beam = beam[:lmax+1]
    beamP = beamP[:lmax+1]

    if mode == 'fg':
        ls, cls_theory = fg_cl_thfit(f=f, lmax=lmax, psky=psky, apo=apo)
    if mode == 'cmb':
        ls, cltt, clte, clee, clbb = pf.get_theory_cls(lmax=lmax, mode='cl')
        factor = ls*(1+ls)
        cls_theory = (cltt, clee, clbb, clte)
  
    for i in np.arange(nsim):
        print 'sim #{}...'.format(i+1)
        I, Q, U = pf.simulate_cmb_map(nside=nside, lmax=lmax,
                                    frequency=f,smear=smear,
                                    cls_theory=cls_theory,
                                    beam=beam, beamP=beamP,
                                    save=False,
                                    beam_file=None)
        print 'Cl #{}...'.format(i+1)
        Tlm, Elm, Blm = pf.calc_alm(I, Q, U, mask=mask,
                                    lmax=lmax,
                                    add_beam=None,add_beamP=None,
                                    div_beam=beam,div_beamP=beamP,
                                    healpy_format=True)
        TT = hp.alm2cl(Tlm)
        EE = hp.alm2cl(Elm)
        BB = hp.alm2cl(Blm)
 
        #ls, TT, EE, BB, TE, TB, EB = measure_dlcl(mask=mask, Imap=I, Qmap=Q, Umap=U,
        #                                          beam=beam, beamP=beamP,
        #                                            mode='cl',frequency=f,
        #                                            lmax=lmax,lmin=0,
        #                                            put_mask=put_mask,
        #                                            psky=psky,
        #                                            mask_sources=mask_sources,
        #                                            apodization=apo)
        TTm += TT/nsim; EEm += EE/nsim; BBm += BB/nsim
        
        if visual_check:
            hp.mollview(I)
            hp.mollview(Q)
            hp.mollview(U)

    fsky_correction.append(cls_theory[0] / TTm)
    fsky_correction.append(cls_theory[1] / EEm)
    fsky_correction.append(cls_theory[2] / BBm)

    fsky_correction = np.array(fsky_correction)
    fsky_correction[np.isnan(fsky_correction)] = 0.
    fsky_correction[fsky_correction==np.inf] = 0.

    return ls, fsky_correction, TTm, EEm, BBm, cls_theory
