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


#from __future__ import print
import healpy as hp
import os
import scipy
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, FileTransferSpeed, FormatLabel, Percentage, ProgressBar, ReverseBar, RotatingMarker, SimpleProgress, Timer
from numba import jit
from scipy.special import sph_harm


#data_path = '/home/verag/Projects/Repositories/npoint-fgs/data/'
data_path = '/Users/verag/Research/Repositories/npoint-fgs/data/'

def prepare_map(mapname='HFI_SkyMap_143_2048_R2.02_full.fits',
                maskname='HFI_Mask_GalPlane-apo0_2048_R2.00.fits',
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
        bispectrum = np.load(filename, 'r')
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
    ylm = np.load(filename, 'r')
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

def get_hs(filename, lmax=100):

    if os.path.exists(filename):
        hs = np.load(filename, 'r')
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

def plot_bispectrum(b,slices=None,title=None,logplot=True,filename=None):

    if logplot:
        y = np.log10( np.abs(b) )
    else:
        y = b
  
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
    ar = np.load(filename, 'r')
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


def simulate_noisemap_old(template_name='HFI_SkyMap_353_2048_R2.02_full.fits',
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
                maskname = 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits',
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
        alms = np.load(data_path + newname, 'r')
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
                almfilename=None):
   


    filename = 'bispectrum__lmax{}_mask_{}_I{}_P{}.npy'.format(lmax,
                                                                 masktype,
                                                                 Imap_label,
                                                                 Pmap_label)
    
    

    if almfilename is not None:
        Tlm, Elm, Blm = np.load(data_path + almfilename, 'r')
    else:
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

    hs = get_hs('hs_lmax100.npy',lmax=lmax)
    return Tlm,Elm,Blm,hs




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
        
beam_index = {
    '100': 3,
    '143': 4,
    '217': 5,
    '353': 6,
    '100P': 7,
    '143P': 8,
    '217P': 9,
    '353P': 10,
    }

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

def check_Tlm2d(nu=100, lmax=300,
                maskfield=2, source_maskfield=0,
                label_loc='lower right', xmax=None):
    
    Imap_name = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(nu)
    Imap =hp.read_map(data_path + Imap_name)
    mask=hp.read_map(data_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits',
                     field=maskfield)
    smask=hp.read_map(data_path + 'HFI_Mask_PointSrc_2048_R2.00.fits',
                     field=source_maskfield)
    mask *= smask

    hdulist = fits.open(data_path + 'HFI_RIMO_Beams-100pc_R2.00.fits')
    beam = hdulist[beam_index['{}'.format(nu)]].data.NOMINAL[0][:lmax+1]
    
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
            
#########%%%%%%%%%%%%%^^^^^^^^^^^^^^^&&&&&&**********************
def get_Tlm(filename='test_tlm.npy',
            Imap=None, mask=None,
            add_beam=None,div_beam=None,
            healpy_format=False,
            lmax=100, recalc=True,
            filtermap=False, l0=None,
            save=False):
    """computes and saves 2d alms from a given
    map and mask, corrected for sqrt(fsky), beams added or divided by choice.
    """
    if not recalc and os.path.exists(data_path + filename):
        Tlm2d = np.load(data_path + filename, 'r')
        return Tlm2d
    fsky = mask.sum() / len(mask)
    Tlm = hp.map2alm(Imap * mask, lmax=lmax) / np.sqrt(fsky)
    if add_beam is not None:
        hp.sphtfunc.almxfl(Tlm, add_beam, inplace=True)
    if div_beam is not None:
        hp.sphtfunc.almxfl(Tlm, 1./div_beam, inplace=True)

    if not healpy_format:
        ls, ms = hp.sphtfunc.Alm.getlm(lmax, np.arange(len(Tlm)))
        Tlm = make2d_alm(Tlm, lmax, ls, ms)
    if save:
        np.save(data_path + filename, Tlm)
    return Tlm

def get_ElmBlm(filename='test_elmblm.npy',
               Qmap=None, Umap=None, mask=None,
               lmax=100,add_beam=None,div_beam=None,
               healpy_format=False,
               recalc=True,
               filtermap=False, l0=None,
               save=False):
    """computes and saves 2d (Elms, Blms) from given
    Q and U maps, corrected for sqrt(fsky)
    """
    if not recalc and os.path.exists(data_path + filename):
        Elm2d, Blm2d = np.load(data_path + filename, 'r')
        return Elm2d, Blm2d
    fsky = mask.sum() / len(mask)
    Elm, Blm = hp.map2alm_spin( (Qmap*mask,Umap*mask), 2, lmax=lmax )
    Elm /= np.sqrt(fsky)
    Blm /= np.sqrt(fsky)
    if add_beam is not None:
        hp.sphtfunc.almxfl(Elm, add_beam, inplace=True)
        hp.sphtfunc.almxfl(Blm, add_beam, inplace=True)
    if div_beam is not None:
        hp.sphtfunc.almxfl(Elm, 1./div_beam, inplace=True)
        hp.sphtfunc.almxfl(Blm, 1./div_beam, inplace=True)
    if not healpy_format:
        ls, ms = hp.sphtfunc.Alm.getlm(lmax, np.arange(len(Elm)))
        Elm = make2d_alm(Elm, lmax, ls, ms)
        Blm = make2d_alm(Blm, lmax, ls, ms)
    if save:
        np.save(data_path + filename, [Elm,Blm])
    return Elm, Blm



def get_cholesky_noise(frequency=100, mapname=None,
                 save=True,rewrite=False):

    if mapname is None:
        mapname = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(frequency)
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
        np.save(data_path + newname, L)
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

    


        
def simulate_noise(npix=50331648, frequency=100,
                      save=True, filename='test_noise100.fits'):

    I = np.random.standard_normal(npix)
    Q = np.random.standard_normal(npix)
    U = np.random.standard_normal(npix)
    L = np.load(data_path + 'HFI_SkyMap_{}_2048_R2.02_full_cholesky.npy'.format(frequency), 'r')
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


def simulate_cmb(nside=2048, lmax=3000,
                 frequency=100,smear=False,
                 nomap = False, beam=None, beamP=None,
                 save=False, filename='testcmb.fits',
                 cl_file='bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl'):
        
    ls, cltt, clte, clee, clbb = get_theory_cls(lmax=lmax, cl_file=cl_file)
 
    Tlm, Elm, Blm = hp.synalm( (cltt, clee, clbb, clte), new=True, lmax=lmax)

    
    if smear:
        if (beam is None) or (beamP is None) :
            hdulist = fits.open(data_path + 'HFI_RIMO_Beams-100pc_R2.00.fits')
            beam = hdulist[beam_index['{}'.format(frequency)]].data.NOMINAL[0][:lmax+1]
            beamP = hdulist[beam_index['{}P'.format(frequency)]].data.NOMINAL[0][:lmax+1]
        hp.sphtfunc.almxfl(Tlm, beam, inplace=True)
        hp.sphtfunc.almxfl(Elm, beamP, inplace=True)
        hp.sphtfunc.almxfl(Blm, beamP, inplace=True)

    if nomap:
        return Tlm,Elm,Blm
    
    Tmap = hp.alm2map( Tlm, nside )
    Qmap, Umap = hp.alm2map_spin( (Elm, Blm), nside, 2, lmax=lmax)

    if save:
        hp.write_map([Tmap,Qmap,Umap],data_path + filename)
    return Tmap, Qmap, Umap

def get_theory_cls(lmax=3000,cl_file='bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl'):
    """ Read theory cls, remove factor of l*(l+1)/2pi,
        convert to uK_CMB^2 units,
        and pad with zeros, such that l starts at 0.
    """

    cl = np.loadtxt(data_path + cl_file)
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

def observe_cmb_sky(save=False, filename='testsky.fits',
                    nside=2048, npix=None, lmax=3000,
                    frequency=100, beam=None, beamP=None,
                    cl_file='bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl'):

    if npix is None:
        npix = hp.nside2npix(nside)
    if nside is None:
        nside = hp.npix2nside(npix)
        
    Tcmb, Qcmb, Ucmb = simulate_cmb(nside=nside, lmax=lmax,
                                frequency=frequency,smear=True,
                                nomap=False, save=False,
                                beam=beam, beamP=beamP,
                                cl_file=cl_file)
    Tnoise, Qnoise, Unoise = simulate_noise(npix=npix,
                                            frequency=frequency,
                                            save=False)

    if save:
        hp.write_map(data_path + filename, [Tcmb+Tnoise, Qcmb+Qnoise, Ucmb+Unoise])

    return Tcmb+Tnoise, Qcmb+Qnoise, Ucmb+Unoise



def observe_alms(save=True, filetag='test',
                     mask=None,mask_percentage=60,mask_sources=True,
                     apodization='0',
                    nside=2048, npix=None, lmax=3000,
                    frequency=100, beam=None, beamP=None,
                    simulation=True,return_alms=True,
                    cl_file='bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl'):

    if mask is None:
        mask = get_mask(mask_percentage=mask_percentage,
                        mask_sources=mask_sources,
                        apodization=apodization)

    if simulation:
        Imap, Qmap, Umap = observe_cmb_sky(save=False,
                    nside=nside, npix=None, lmax=lmax,
                    frequency=frequency,
                    cl_file=cl_file)
    else:
        mapname = 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(frequency)
        Imap, Qmap, Umap = hp.read_map(data_path + mapname, field=(0,1,2))

    if save:
        Pfilename = data_path + 'ElmBlm_{}.npy'.format(filetag)
        Tfilename = data_path + 'Tlm_{}.npy'.format(filetag)
        
    Elm, Blm = get_ElmBlm(filename=Pfilename,
                            Qmap=Qmap, Umap=Umap, mask=mask,
                            lmax=lmax,add_beam=None,div_beam=None,
                            healpy_format=False,
                            recalc=recalc,
                            filtermap=False, l0=None,
                            save=save)

    Tlm = get_Tlm(filename=Tfilename,
                            Imap=Imap, mask=mask,
                            lmax=lmax,add_beam=None,div_beam=None,
                            healpy_format=False,
                            recalc=recalc,
                            filtermap=False, l0=None,
                            save=save)

    if return_alms:
        return Tlm,Elm,Blm

    

MASK_FIELD = {
    60: 2,

}

def get_mask(mask_percentage=60,
             mask_sources=True,
             apodization='0'):

    field = MASK_FIELD[mask_percentage]
    mask = hp.read_map(data_path + 'HFI_Mask_GalPlane-apo{}_2048_R2.00.fits'.format(apodization),
                       field=field)
    if mask_sources:
        smask = hp.read_map(data_path + 'HFI_Mask_PointSrc_2048_R2.00.fits')
        mask *= smask

    return mask
        
