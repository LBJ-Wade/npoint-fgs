import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
#from __future__ import print
import healpy as hp
import os
import scipy
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, FileTransferSpeed, FormatLabel, Percentage, ProgressBar, ReverseBar, RotatingMarker, SimpleProgress, Timer
from numba import jit
from scipy.special import sph_harm


data_path = '/Users/verag/Research/Repositories/npoint-fgs/data/'

def prepare_map(mapname='HFI_SkyMap_143_2048_R2.02_full.fits',
                maskname='HFI_Mask_GalPlane-apo0_2048_R2.00.fits',
                field = (0,1,2),
                fwhm=0.06283185307179587,
                nside_out=16,
                rewrite_map=False):

    
    newname = mapname[:-5] + '_new_fwhm_{:.3f}rad_nside_{}.fits'.format(fwhm, nside_out)
    if not os.path.exists(data_path + newname) or rewrite_map:
        print 'reading mask...'
        mask = hp.read_map(data_path + maskname)
        masknside = hp.get_nside(mask)
        print 'processing map...'
        Imap, Qmap, Umap = hp.read_map( data_path + mapname,
                                        hdu=1, field=(0,1,2) )
        mapnside = hp.get_nside( Imap )
        if mapnside != masknside:
            mask = hp.pixelfunc.ud_grade(mask,
                                         nside_out=mapnside)
   
        masked_Imap = Imap * mask
        masked_Qmap = Qmap * mask
        masked_Umap = Umap * mask

        if np.isclose(fwhm, 0.):
            smoothed_Imap = masked_Imap
            smoothed_Qmap = masked_Qmap
            smoothed_Umap = masked_Umap
        else:
            smoothed_Imap = hp.sphtfunc.smoothing( masked_Imap,fwhm=fwhm )
            smoothed_Qmap = hp.sphtfunc.smoothing( masked_Qmap,fwhm=fwhm )
            smoothed_Umap = hp.sphtfunc.smoothing( masked_Umap,fwhm=fwhm )

        if mapnside == nside_out:
            reduced_Imap = smoothed_Imap
            reduced_Qmap = smoothed_Qmap
            reduced_Umap = smoothed_Umap
        else:
            reduced_Imap = hp.pixelfunc.ud_grade( smoothed_Imap,
                                                  nside_out=nside_out )
            reduced_Qmap = hp.pixelfunc.ud_grade( smoothed_Qmap,
                                                  nside_out=nside_out )
            reduced_Umap = hp.pixelfunc.ud_grade( smoothed_Umap,
                                                  nside_out=nside_out )
        reduced_map = [reduced_Imap, reduced_Qmap, reduced_Umap]
        
        hp.fitsfunc.write_map( data_path + newname,
                               reduced_map )

    print 'reading map...'
    maps = hp.read_map( data_path + newname,
                        field=field )
    return maps
        
                 
def plot_hists(nus=[143,353],
               nside=2048,
               fwhm=0.0,
              bins=100,normed=True,
              atol=1e-6):

    map1_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nus[0])
    map2_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nus[1])
   
    map1 = prepare_map( map1_name, field=0,
                        nside_out=nside, fwhm=fwhm )
    map2 = prepare_map( map2_name, field=0,
                        nside_out=nside, fwhm=fwhm )
        
    plt.figure()
    y,x = pl.histogram(map1[np.where(np.negative(np.isclose(map1,0.,atol=atol)))],
                       bins=bins,normed=normed)
    bin = (x[:-1] + x[1:]) / 2.
    plt.semilogy(bin, y, lw=3, label='{}'.format(nus[0]))

    y,x = pl.histogram(map2[np.where(np.negative(np.isclose(map2,0.,atol=atol)))],
                       bins=bins,normed=normed)
    bin = (x[:-1] + x[1:]) / 2.
    plt.semilogy(bin, y, lw=3, label='{}'.format(nus[1]))
        

    plt.legend()

def correlation2pt(map1, map2=None, npoints=100):

    if map2 is None:
        map2 = map1
    alm1 = hp.map2alm(map1)
    alm2 = hp.map2alm(map2)
    clcross = hp.alm2cl(alm1, alm2)

    
    thetas = np.linspace(0., np.pi, npoints)
    corr = np.zeros(npoints)
    
    for i,theta in enumerate(thetas):
        for l,cl in enumerate(clcross):
            lfactor = (2*l + 1.)/(4.*np.pi)  * scipy.special.lpmv(0, l, np.cos(theta))
            corr[i] += lfactor * cl

    return thetas, corr



def calc_IP2_equil(Imap, Qmap, Umap,
                   lmax=100):

    Tlm = hp.map2alm( Imap )
    Elm,Blm = hp.map2alm_spin( (Qmap,Umap), 2 )
    
    TEE = hp.alm2cl( Tlm, Elm**2 )
    TBB = hp.alm2cl( Tlm, Blm**2 )
    TEB = hp.alm2cl( Tlm, Elm*Blm )

    ls = np.arange( len(TEE) )
    return ls, TEE, TBB, TEB

def plot_IP2_equil(Imap_name='HFI_SkyMap_353_2048_R2.02_full.fits',
             Pmap_name='HFI_SkyMap_353_2048_R2.02_full.fits',
             nus=None, fwhm=0.063, nside=16):

    title = ''
    if nus is not None:
        Imap_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nus[0])
        Pmap_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nus[1])
        title = '$I_{%i} P^2_{%i}$  (equilateral)' % (nus[0],nus[1])
        
    Imap = prepare_map( Imap_name, field=0,
                        nside_out=nside, fwhm=fwhm )
    Qmap, Umap = prepare_map( Pmap_name, field=(1,2),
                              nside_out=nside, fwhm=fwhm )
    
    ls, TEE, TBB, TEB = calc_IP2_equil(Imap, Qmap, Umap)
     
    pl.figure()
    pl.title(title, fontsize=20)
    pl.plot( ls, TEE, lw=3, label='TEE' )
    pl.plot( ls, TBB, lw=3, label='TBB' )
    pl.plot( ls, TEB, lw=3, label='TEB' )
    pl.xlabel( '$\ell$', fontsize=20 )
    pl.legend()
        
def calc_TEB(Imap_name='HFI_SkyMap_353_2048_R2.02_full.fits',
             Pmap_name='HFI_SkyMap_353_2048_R2.02_full.fits',
             nus=None, fwhm=0.063, nside=16, lmax=100,
             lmaps_only=False, filename=None):
    """Master function for computing the bispectrum TEB
    """

    # read from file if it's there
    if filename is None:
        filename = 'bispectrum_lmax{}'.format(lmax)
        if nus is not None:
            filename += '_{}-{}-{}GHz.npy'.format(nus[0],nus[1],nus[1])
        else:
            filename += '_{}'.format(Imap_name[-5])
            if Imap_name != Pmap_name:
                filename += '_{}.npy'.format(Pmap_name[-5])
            else:
                filename += '.npy'
        print 'looking for {} ...'.format(filename)
    if os.path.exists(filename) and not lmaps_only:
        bispectrum = np.load(filename)
        return bispectrum

    # compute it, if the file doesn't exist
    if nus is not None:
        Imap_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nus[0])
        Pmap_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nus[1])
        title = '$I_{%i} P^2_{%i}$  (equilateral)' % (nus[0],nus[1])
        
    Imap = prepare_map( Imap_name, field=0,
                        nside_out=nside, fwhm=fwhm )
    Tlm = hp.map2alm( Imap, lmax=lmax )

    Qmap, Umap = prepare_map( Pmap_name, field=(1,2),
                              nside_out=nside, fwhm=fwhm )
    
    
    Elm,Blm = hp.map2alm_spin( (Qmap,Umap), 2, lmax=lmax )

    if lmax is None:
        lmax = hp.sphtfunc.Alm.getlmax(len(Tlm))
    ls, ms = hp.sphtfunc.Alm.getlm(lmax,np.arange(len(Tlm)))
    lmin = ls.min()
    mapsize = len(Imap)
    pixelsize = hp.pixelfunc.nside2pixarea(nside)
    
    Ylm = calc_Ylm(Imap, ls, ms)


    #return Ylm, Tlm, ls, ms
    print 'calculating Tl,El,Bl ...'
    Tl = sum_over_m(Tlm, Ylm, ls,
                        lmax=lmax, lmin=lmin, mapsize=mapsize)
    El = sum_over_m(Elm, Ylm, ls,
                        lmax=lmax, lmin=lmin, mapsize=mapsize)
    Bl = sum_over_m(Blm, Ylm, ls,
                        lmax=lmax, lmin=lmin, mapsize=mapsize)

    if lmaps_only:
        return Tl,El,Bl
    
    hs = get_hs(lmin=lmin, lmax=lmax)

    print 'calculating bispectrum ...'
    bispectrum = calc_bispectrum(Tl, El, Bl, hs,
                                 pixelsize,
                                 lmax=lmax, lmin=lmin,
                                 mapsize=mapsize)
    clean_bispectrum_of_naninf(bispectrum, hs, inplace=True)
    np.save(filename, bispectrum)
    return bispectrum


def clean_bispectrum_of_naninf(bispectrum, hs, inplace=True):
    """Turns all bispectrum entries to zero,
    if hs is zero at that spot (because b~1/hs^2...)
    """
    if inplace:
        bispectrum[np.isclose(hs,0.)] = 0.
    else:
        b = bispectrum.copy()
        b[np.isclose(hs,0.)] = 0.
        return b

@jit
def calc_bispectrum(Tl, El, Bl, hs,
                    pixelsize,
                    lmax=100, lmin=0,
                    mapsize=3072):

    result = np.zeros((lmax+1,lmax+1,lmax+1)) 
    ls_array = np.arange(lmin, lmax+1)
    for l1 in ls_array:
        print 'slice #{}/{}...'.format(l1,lmax+1)
        for l2 in ls_array:
            for l3 in ls_array:
                for m in np.arange(mapsize):
                    result[l1,l2,l3] += Tl[l1,m] * El[l2,m] * Bl[l3,m]

                result[l1,l2,l3] *= ( pixelsize / hs[l1,l2,l3]**2 )
                
    return result
                    
@jit
def sum_over_m(alm, Ylm, ls,
                   lmax=100, lmin=0, mapsize=3072):
    """The assumption is that the field is real, so that
        the sum over m produces a real map, "Al",
        which is the return of the function.

        Note: m=0 term is taken twice, which is a slight inconsistency
    """
    ls_array = np.arange(lmin, lmax+1)
    result = np.zeros((len(ls_array), mapsize), dtype=complex)
    
    for i in np.arange(len(ls_array)):
        for j in np.arange(len(ls)):
            if ls[j] == ls_array[i]:
                for k in np.arange(mapsize):
                    result[i,k] += alm[j] * Ylm[j,k] / ( 2.*ls_array[i] + 1. )

    # multiply by 2 because alm's in healpy only contain m>=0 terms
    return 2.*result.real


def calc_Ylm(mapa, ls, ms):
    
    lmax = ls.max()
    nside = hp.pixelfunc.get_nside(mapa)
    mapsize = len(mapa)
    numls = len(ls)
    
    filename = ('Ylm_map_nside{}_lmax{}.npy'.format(nside,lmax))
    if not os.path.exists(filename):
        ylm = []
        theta,phi = hp.pixelfunc.pix2ang(nside,np.arange(mapsize))

        i=0
        for l,m in zip(ls,ms):
            print 'calculating #{}/{}...'.format(i+1,numls)
            i+=1
            ylm.append(sph_harm(m, l, theta, phi))

        np.save(filename,ylm)
    ylm = np.load(filename)
    return ylm


#@jit
#def compute_3lm(Tlm,Elm,Blm,length):
#    cube = np.zeros((length,length,length),dtype=complex)
#    for i,tlm in enumerate(Tlm):
#        for j,elm in enumerate(Elm):
#            for k,blm in enumerate(Blm):
#                cube[i,j,k] = tlm * elm * blm
#    return cube
    


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

def get_hs(lmin=0, lmax=100):
    filename = 'hs_lmax100.npy'
    if os.path.exists(filename):
        hs = np.load(filename)
        return hs

    Fks = np.loadtxt('Fks_1000.txt')
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



def plot_bispectrum(b,slices=None,title=None):

    y = np.log10( np.abs(b) )
    #y [ b==0. ] = 0.
  
    #ypositive = np.zeros(b.shape)
    #ynegative = np.zeros(b.shape)
    #ypositive[ b > 0. ] = np.abs( y[ b > 0. ] )
    #ynegative[ b < 0. ] = -1.*np.abs( y[ b < 0. ] )
    #return ypositive,ynegative
    
    if slices is None:
        plot_3D_bispectrum(y, title=title)
    else:
        slices = np.atleast_1d(slices)
        for s in slices:
            plot_slice_bispectrum(y, s=s, title=title)

def plot_slice_bispectrum(y, s=None, title='',
                          colormap='coolwarm'):
    if s is None:
        s=10

    pl.figure()
    if title is not None:
        title += ' (slice ind={})'.format(s)
    else:
        title = '(slice ind={})'.format(s)
    pl.title(title, fontsize=20)

    pl.imshow(y[s], cmap=colormap)
    pl.colorbar()
    
