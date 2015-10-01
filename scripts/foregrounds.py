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
    

def calc_fg_powerlaw(frequency=353,lmin=40,lmax=600,
                     put_mask=True,
                     psky=70,
                     mask_sources=True,
                     apodization='0',
                     plots=True,
                     guess=(200,-2.4),
                     nbins=25):

    ls, TT, EE, BB, TE, TB, EB = pf.measure_planck_Dl(frequency=353,
                                                        lmax=600,lmin=40,
                                                        put_mask=put_mask,
                                                        psky=psky,
                                                        mask_sources=mask_sources,
                                                        apodization=apodization)

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
    
    if plots:    
        plt.figure()
        #plt.loglog(ls, TT)
        plt.loglog(ls, EE, color='red')
        plt.loglog(ls, BB, color='blue')

        l,dlee = bin_data(ls, EE, nbins)
        plt.loglog(l,dlee,'o')
        plt.loglog(ls, powerlaw(sol[1],ls), color='Maroon', label='EE')
        l,dlbb = bin_data(ls, BB, nbins)
        plt.loglog(l,dlbb,'o')
        plt.loglog(ls, powerlaw(sol[2],ls), color='cyan', label='BB')
        #for i,PS_name in enumerate(PS_names):
        #    l,dl = bin_data(ls, PSa[i], nbins)
        #    plt.loglog(l,dl,'o')
        #    plt.loglog(ls, powerlaw(sol[i],ls), label=PS_name)
        plt.xlim(xmin=lmin,xmax=lmax)
        plt.legend(loc='lower left',fontsize=20)
        plt.xlabel('$\ell$',fontsize=20)
        plt.ylabel('$\mathcal{D}_\ell [\mu K^2_{CMB}]$',fontsize=20)
        plt.savefig(pf.FGS_RESULTS_PATH + 'foreground_power_{}GHz_{}psky.pdf'.format(frequency,psky))

        

    return fg_dict, fg_dict['EE'], fg_dict['BB'], ls, EE, BB

def all_fg_powerlaws(frequency=353,lmin=40,lmax=600,
                     put_mask=True,
                     pskys=[20,40,60,70,80],
                     mask_sources=True,
                     apodization='0',
                     plots=True,
                     guess=(200,-2.4),
                     nbins=25):

    pEEs = []
    pBBs = []
    dlees = []
    dlbbs = []
    for psky in pskys:
        fg_dict, pEE, pBB, ls, EE, BB = calc_fg_powerlaw(frequency=frequency,
                                                            lmin=lmin,lmax=lmax,
                                                            put_mask=put_mask,
                                                            psky=psky,
                                                            mask_sources=mask_sources,
                                                            apodization=apodization,
                                                            plots=False,
                                                            guess=guess,
                                                            nbins=nbins)
        filename = pf.FGS_RESULTS_PATH + "fg_power_{}GHz_{}psky.pkl".format(frequency,psky)
        pickle.dump( fg_dict, open( filename, "wb" ) )

        l,dlee = bin_data(ls, EE, nbins)
        l,dlbb = bin_data(ls, BB, nbins)
        
        pEEs.append(pEE)
        pBBs.append(pBB)
        dlees.append(dlee)
        dlbbs.append(dlbb)

    plot_fg_power(pEEs, pBBs, l, dlees, dlbbs, pskys,
                  lmin=lmin, lmax=lmax, frequency=frequency)

    plot_alphas(pEEs,pBBs,pskys)

def plot_alphas(pEEs, pBBs, pskys):
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
    plt.savefig(pf.FGS_RESULTS_PATH + 'planck_alphas.pdf')
        

def plot_fg_power(pEEs, pBBs, l, dlees, dlbbs, pskys,
                  lmin=40, lmax=600, frequency=353):

    
    ls = np.arange(lmin, lmax+1)
    title = 'percent sky (from bottom):'
    for ps in pskys:
        title += ' {}'.format(ps)

    plt.figure()
    for i,p in enumerate(pEEs):
        plt.loglog(l, dlees[i],'o', color='black')
        plt.loglog(ls, powerlaw(p,ls), color='Maroon')

    plt.xlim(xmin=lmin,xmax=lmax)
    plt.legend(loc='bottom left',fontsize=20, frameon=False)
    plt.xlabel('$\ell$',fontsize=20)
    plt.ylabel('$\mathcal{D}^{EE}_\ell [\mu K^2_{CMB}]$',fontsize=20)
    plt.title(title, fontsize=15)
    
    plt.savefig(pf.FGS_RESULTS_PATH + 'EE_foreground_power_{}GHz.pdf'.format(frequency))


    plt.figure()
    for i,p in enumerate(pBBs):
        plt.loglog(l, dlbbs[i],'o', color='black')
        plt.loglog(ls, powerlaw(p,ls), color='Blue')

    plt.xlim(xmin=lmin,xmax=lmax)
    plt.legend(loc='bottom left',fontsize=20, frameon=False)
    plt.xlabel('$\ell$',fontsize=20)
    plt.ylabel('$\mathcal{D}^{BB}_\ell [\mu K^2_{CMB}]$',fontsize=20)
    plt.title(title, fontsize=15)
    
    plt.savefig(pf.FGS_RESULTS_PATH + 'BB_foreground_power_{}GHz.pdf'.format(frequency))

"""
FG_POWER = {
    353:,
    143:,
    100:,
    217:,
}
"""
