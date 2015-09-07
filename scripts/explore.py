import numpy as np
import pylab as pl

import matplotlib
if __name__=='__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


#from __future__ import print
import healpy as hp
import os
import scipy
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, FileTransferSpeed, FormatLabel, Percentage, ProgressBar, ReverseBar, RotatingMarker, SimpleProgress, Timer
from numba import jit
from scipy.special import sph_harm


data_path = '/home/verag/Projects/Repositories/npoint-fgs/data/'
#'/Users/verag/Research/Repositories/npoint-fgs/data/'

def prepare_map(mapname='HFI_SkyMap_143_2048_R2.02_full.fits',
                maskname='HFI_Mask_GalPlane_2048_R1.10.fits',
                field = (0,1,2),
                fwhm=0.0,
                nside_out=128,
                rewrite_map=False,
                masktype=None):

    newname = mapname[:-5] + '_fwhm_{:.3f}rad_nside_{}_mask_{}.fits'.format(fwhm, nside_out,masktype)
    if not os.path.exists(data_path + newname) or rewrite_map:
        
        print 'reading mask...'
        mask = hp.read_map(data_path + maskname, field=2)
        masknside = hp.get_nside(mask)
        
        if masknside != nside_out:
            print 'matching mask to map resolution...'
            mask = hp.pixelfunc.ud_grade(mask, nside_out=nside_out)
        print 'done'

            
        print 'processing map...'
        Imap, Qmap, Umap = hp.read_map( data_path + mapname,
                                        hdu=1, field=(0,1,2) )
        mapnside = hp.get_nside(Imap)


        if not np.isclose(fwhm, 0.):
            Imap = hp.sphtfunc.smoothing( Imap,fwhm=fwhm )
            Qmap = hp.sphtfunc.smoothing( Qmap,fwhm=fwhm )
            Umap = hp.sphtfunc.smoothing( Umap,fwhm=fwhm )

        if mapnside != nside_out:
            Imap = hp.pixelfunc.ud_grade( Imap,nside_out=nside_out )
            Qmap = hp.pixelfunc.ud_grade( Qmap,nside_out=nside_out )
            Umap = hp.pixelfunc.ud_grade( Umap,nside_out=nside_out )

        Imap *= mask
        Qmap *= mask
        Umap *= mask
        
        hp.fitsfunc.write_map( data_path + newname, [Imap, Qmap, Umap])

        print 'done'

    print 'reading map...'
    maps = hp.read_map( data_path + newname,
                        field=field )
    return maps
        
                 
def plot_hists(nus=[143,353],
               map1_name=None,
               map2_name=None,
               maskname='wmap_temperature_kq85_analysis_mask_r10_9yr_v5.fits',
               nside=2048,
               fwhm=0.0,
              bins=100,normed=True,
              atol=1e-6, ymin=0.01, ymax=None,
              xmin=-0.001, xmax=0.005):

    if map1_name is None:
        map1_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nus[0])
    label1 = '{} GHz'.format(nus[0])
    if map2_name is None:
        map2_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nus[1])
    label2 = '{} GHz'.format(nus[1])
   
    map1 = prepare_map( map1_name, field=0,
                        maskname=maskname,
                        nside_out=nside, fwhm=fwhm )
    map2 = prepare_map( map2_name, field=0,
                        maskname=maskname,
                        nside_out=nside, fwhm=fwhm )

    y1,x1 = pl.histogram(map1[np.where(np.negative(np.isclose(map1,0.,atol=atol)))],
                       bins=bins,normed=normed)
    bin1 = (x1[:-1] + x1[1:]) / 2.

    y2,x2 = pl.histogram(map2[np.where(np.negative(np.isclose(map2,0.,atol=atol)))],
                       bins=bins,normed=normed)
    bin2 = (x2[:-1] + x2[1:]) / 2.
    #return bin1,y1,bin2,y2
        

    fig = plt.figure()
    ax = plt.gca()
    
    ax.semilogy(bin1, y1, lw=3, label=label1,color='red')
    ax.semilogy(bin2, y2, lw=3, label=label2,color='gray')
    ax.set_xlim(xmin=xmin,xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)

    #ax.set_yscale('log')
    
    ax.set_xlabel('$\mu K$', fontsize=20)
    ax.set_yticks([])
    
    plt.draw()
    plt.legend(frameon=False, fontsize=20)

    plt.savefig('pdfs_{}GHz_{}GHz_fwhm{:.3}rad.pdf'.format(nus[0],nus[1],fwhm))

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

                if hs[l1,l2,l3] == 0.:
                     result[l1,l2,l3] = 0.
                else:
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
            print 'calculating Ylm #{}/{}...'.format(i+1,numls)
            i+=1
            ylm.append(sph_harm(m, l, theta, phi))

        np.save(filename,ylm)
    ylm = np.load(filename)
    return ylm



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

def plot_bispectrum(b,slices=None,title=None,filename=None):

    y = np.log10( np.abs(b) )
    #y = b
  
    #ypositive = np.zeros(b.shape)
    #ynegative = np.zeros(b.shape)
    #ypositive[ b > 0. ] = np.abs( y[ b > 0. ] )
    #ynegative[ b < 0. ] = -1.*np.abs( y[ b < 0. ] )
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

def npy_to_dat(filename='bispectrum_lmax100_353-353-353GHz.npy',
               side=None, returntable=False):
    ar = np.load(filename)
    x = np.zeros(ar.shape)
    y = np.zeros(ar.shape)
    z = np.zeros(ar.shape)
    if side is None:
        side = ar.shape[0]
    fout = open(filename[:-3] + 'dat', 'w')
    
    for l1 in np.arange(side):
        for l2 in np.arange(side):
            for l3 in np.arange(side):
                x[l1,l2,l3] = l1
                y[l1,l2,l3] = l2
                z[l1,l2,l3] = l3
                fout.write('{} {} {} {}\n'.format(l1,l2,l3,ar[l1,l2,l3]))
    fout.close()

    if returntable:
        return x, y, z, ar


def simulate_noisemap(template_name='HFI_SkyMap_353_2048_R2.02_full.fits',
                      nu=None,
                      newname='noisesim_353.fits'):
    if nu is not None:
        template_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nu)
        newname = 'noisesim_{}.fits'.format(nu)

    covII, covIQ, covIU, covQQ, covQU, covUU = hp.read_map( data_path + template_name,
                                                            field=(4,5,6,7,8,9) )
    Npix = covII.shape[0]

    Imap = create_I_noisemap(Npix, covII)
    Qmap, Umap = create_QU_noisemap(Npix, covQQ, covUU, covQU)

    hp.fitsfunc.write_map( data_path + newname,
                           [Imap, Qmap, Umap])

    return [Imap, Qmap, Umap]
def create_I_noisemap(Npix, cov):

    nmap = np.random.standard_normal(Npix) * np.sqrt(cov)

    return nmap

def create_QU_noisemap(Npix, covQQ, covUU, covQU):
    r1 = np.random.standard_normal(Npix)
    r2 = np.random.standard_normal(Npix)

    nmapQ = r1 * np.sqrt(covQQ)
    nmapU = r1 * covQU / np.sqrt(covQQ) + r2 * np.sqrt(covUU - covQU**2/covQQ)

    return nmapQ, nmapU


def get_alms(maps=None,
            mask=None,
            maplabel='353',
            showI=False,
            pol=True,
            intensity=True,
            rewrite=False,
            writemap=False,
            savealms=True,
            masktype='PowerSpectra',#'GalPlane2',
            lmax=100):

    """Each written map file must contain I,Q,U, and each alms file
            must contain Tlm, Elm, and Blm.
    """

    newname = 'alms_lmax{}_mask_{}__'.format(lmax, masktype) + maplabel + '.npy'
    
    if not os.path.exists(data_path + newname) or rewrite:
        print 'alms file {} does not exist; calculating alms...'.format(newname)
        if mask is None:
            if masktype == 'PowerSpectra':
                maskname = 'HFI_PowerSpect_Mask_2048_R1.10.fits'
                maskfield = 0
            elif masktype == 'GalPlane60':
                maskname = 'HFI_Mask_GalPlane_2048_R1.10.fits',
                maskfield = 2
            elif masktype == 'no':
                maskname = 'HFI_PowerSpect_Mask_2048_R1.10.fits'
                maskfield = 0
            mask = hp.read_map(data_path + maskname, field=maskfield)
            if masktype == 'no':
                mask = mask*0. + 1.
        masknside = hp.get_nside(mask)
        if maps is None:
            Imap,Qmap,Umap = hp.read_map( data_path + 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(maplabel),hdu=1, field=(0,1,2) )
            mapnside = hp.get_nside(Imap)
        else:
            if intensity and pol:
                Imap = maps[0]
                Qmap = maps[1]
                Umap = maps[2]
                mapnside = hp.get_nside(Imap)
            elif intensity and not pol:
                Imap = maps[0]
                mapnside = hp.get_nside(Imap)
            elif pol and not intensity:
                Qmap = maps[0]
                Umap = maps[1]
                mapnside = hp.get_nside(Qmap)
                
        if masknside != mapnside:
            print 'adjusting mask to match map resolution...'
            mask = hp.pixelfunc.ud_grade(mask, nside_out=mapnside)

        if showI:
            hp.mollview(Imap*mask)

        alms = []
        if intensity:
            Imap = Imap*mask
            Tlm = hp.map2alm(Imap, lmax=lmax)
            alms.append(Tlm)
        if pol:
            Qmap *= mask
            Umap *= mask
            Elm,Blm = hp.map2alm_spin( (Qmap,Umap), 2, lmax=lmax )
            alms.append(Elm)
            alms.append(Blm)

        #this will only work if get_intensity and get_pol
        if writemap and intensity and pol:
            hp.fitsfunc.write_map( data_path + newname, [Imap, Qmap, Umap])
            
        if savealms and intensity and pol:
            np.save(data_path + newname, alms)

        return alms


    else:
        alms = np.load(data_path + newname)
        if intensity and pol:
            return alms[0], alms[1], alms[2]
        else:
            if intensity:
                return alms[0]
            if pol:
                return alms[1], alms[2]
            

#import pywigxjpf as wig
#lmax = 200
#wig.wig_table_init(2*lmax,9)
#wig.wig_temp_init(2*lmax)

@jit
def get_w3j(l1,l2,l3,m1,m2,m3):
    return wig.wig3jj([2*l1, 2*l2, 2*l3, 2*m1, 2*m2, 2*m3])

    

def get_TEB_prerequisites(maps=None,
                Imap_label='353',Pmap_label='353',
                Imap_name=None,Pmap_name=None,
                lmax=100,masktype='PowerSpectra',
                rewrite=False,
                iso=True):
    """maps is a list of I,Q,U maps
    """

    if iso:
        isolabel = '_iso'
    filename = 'bispectrum{}__lmax{}_mask_{}_I{}_P{}.npy'.format(isolabel,
                                                                 lmax,
                                                                 masktype,
                                                                 Imap_label,
                                                                 Pmap_label)
    
    
    
    #if os.path.exists(filename) and not rewrite:
    #    bispectrum = np.load(filename)
    #    return bispectrum


    Tlm, Elm, Blm = get_alms(maps=maps,mask=None,masktype=masktype,
                             maplabel=Imap_label,
                             showI=False,rewrite=False,
                             pol=True,intensity=True,
                             writemap=False,savealms=True,
                             lmax=lmax)

    if Imap_label != Pmap_label:
        Elm, Blm = get_alms(maps=maps,mask=None,masktype=masktype,
                             maplabel=Pmap_label,
                             showI=False,rewrite=False,
                             pol=True,intensity=False,
                             writemap=False,savealms=True,
                             lmax=lmax)
  
    ls, ms = hp.sphtfunc.Alm.getlm(lmax,np.arange(len(Tlm)))
    lmin = ls.min()


    hs = get_hs(lmin=lmin, lmax=lmax)
    return Tlm,Elm,Blm,ls,ms,hs

    #print 'calculating bispectrum ...'
   
    #if iso:
    #    bispectrum = calc_b_iso(Tlm.real, Elm.real, Blm.real, ls, ms,
    #                                hs, lmax=lmax)
    #else:
    #    bispectrum = calc_b(Tlm, Elm, Blm, ls, ms,
    #                            hs, lmax=lmax)

    #np.save(filename, bispectrum)
    #return bispectrum

@jit#(nopython=True)
def calc_b_iso(Tlm, Elm, Blm,
               ls, ms, hs, lmax=100):
    result = np.zeros((lmax+1,lmax+1)) 
    for i in np.arange(len(ls)):
        l1 = ls[i]
        m1 = ms[i]
        for j in np.arange(len(ls)):
            l2 = ls[j]
            m2 = ms[j]
            for k in np.arange(len(ms)):
                l3 = ls[j]
                m3 = ms[k]
                factor = 2.
                if l1==l2:
                    factor = 1.

                w3jfactor = wig.wig3jj([2*l1, 2*l2, 2*l2, 2*m1, 2*m2, 2*m3])#get_w3j(l1,l2,l2,m1,m2,m2)
                result[l1,l2] += w3jfactor#Tlm[i] * Elm[j] * Blm[j] * factor * w3jfactor

            #if hs[l1,l2,l2] == 0.:
            #    result[l1,l2] = 0.
            #else:
            #    result[l1,l2] /= hs[l1,l2,l2]
                
    return result

def calc_b():
    result = np.zeros((lmax+1,lmax+1,lmax+1)) 
    ls_array = np.arange(lmin, lmax+1)
    for l1 in ls_array:
        print 'slice #{}/{}...'.format(l1,lmax+1)
        for l2 in ls_array:
            for l3 in ls_array:
                for m in np.arange(mapsize):
                    result[l1,l2,l3] += Tl[l1,m] * El[l2,m] * Bl[l3,m]

                if hs[l1,l2,l3] == 0.:
                     result[l1,l2,l3] = 0.
                else:
                    result[l1,l2,l3] *= ( pixelsize / hs[l1,l2,l3]**2 )
                
    return result


######################
######################

def get_prerequisites(maps=None,
                Imap_label='353',Pmap_label='353',
                Imap_name=None,Pmap_name=None,
                lmax=100,masktype='PowerSpectra',
                rewrite=False,
                iso=True):
   

    if iso:
        isolabel = '_iso'
    filename = 'bispectrum{}__lmax{}_mask_{}_I{}_P{}.npy'.format(isolabel,
                                                                 lmax,
                                                                 masktype,
                                                                 Imap_label,
                                                                 Pmap_label)
    
    

    Tlm, Elm, Blm = get_alms(maps=maps,mask=None,masktype=masktype,
                             maplabel=Imap_label,
                             showI=False,rewrite=False,
                             pol=True,intensity=True,
                             writemap=False,savealms=True,
                             lmax=lmax)

    if Imap_label != Pmap_label:
        Elm, Blm = get_alms(maps=maps,mask=None,masktype=masktype,
                             maplabel=Pmap_label,
                             showI=False,rewrite=False,
                             pol=True,intensity=False,
                             writemap=False,savealms=True,
                             lmax=lmax)

    ls, ms = hp.sphtfunc.Alm.getlm(lmax,np.arange(len(Tlm)))
    lmin = ls.min()

    Tlm = make2d_alm(Tlm, lmax, ls, ms)
    Elm = make2d_alm(Elm, lmax, ls, ms)
    Blm = make2d_alm(Blm, lmax, ls, ms)

    hs = get_hs(lmin=lmin, lmax=lmax)
    return Tlm,Elm,Blm,hs

@jit
def make2d_alm(alm, lmax, ls, ms):
    
    #len_tot = sum([2*l + 1 for l in range(lmax + 1)])
    alm2d = []
    for l in range(lmax+1):
        #print l
        ms2d = range(l+1)
        ms2d_neg = list(-1*np.array(list(reversed(range(1,l+1)))))
        ms2d = ms2d + ms2d_neg
        #print ms2d
        alm2d.append(np.zeros(2*l + 1, dtype=complex))
        l_inds = ls == l
        for m in ms2d:
            m_inds = ms == np.abs(m)
            inds = m_inds & l_inds
            #print m, (-1.)**m * np.conjugate(alm[inds])
            if m < 0:
                alm2d[l][m] = (-1.)**m * np.conjugate(alm[ms == np.abs(m)][0])
            else:
                alm2d[l][m] = alm[ms == m][0]

    return alm2d
        
