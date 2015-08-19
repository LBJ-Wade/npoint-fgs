import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
#from __future__ import print
import healpy as hp
import os
import scipy

data_path = '/Users/verag/Research/Repositories/fgs/data/'

def prepare_map(mapname='HFI_SkyMap_353_2048_R2.02_full.fits',
                maskname='HFI_Mask_GalPlane-apo0_2048_R2.00.fits',
                field = (0,1,2),
                fwhm=0.06283185307179587,
                nside_out=16,
                rewrite_map=False):

    print 'reading mask...'
    mask = hp.read_map(data_path + maskname)
    masknside = hp.get_nside(mask)
    
    newname = mapname[:-5] + '_new_fwhm_{:.3f}rad_nside_{}.fits'.format(fwhm, nside_out)
    if not os.path.exists(data_path + newname) or rewrite_map:
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
        
        smoothed_Imap = hp.sphtfunc.smoothing( masked_Imap,fwhm=fwhm )
        smoothed_Qmap = hp.sphtfunc.smoothing( masked_Qmap,fwhm=fwhm )
        smoothed_Umap = hp.sphtfunc.smoothing( masked_Umap,fwhm=fwhm )
        
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
        
                 
def plot_hists(mapnames=['HFI_SkyMap_353_2048_R2.02_full.fits','HFI_SkyMap_143_2048_R2.02_full.fits'],
              maskname='HFI_Mask_GalPlane-apo0_2048_R2.00.fits',
              fwhm=0.0628318530717958,
              labels=['353 GHz','143 GHz'],
              bins=1000,normed=True,atol=1e-6):

    plt.figure()
    processed_maps = prepare_maps(mapnames=mapnames,
                                  maskname=maskname,
                                  fwhm=fwhm)

    for i,map in enumerate(processed_maps):
        y,x = pl.histogram(map[np.where(np.negative(np.isclose(map,0.,atol=atol)))],bins=bins,normed=normed)
        bin = (x[:-1] + x[1:]) / 2.
        plt.semilogy(bin, y, lw=3, label=labels[i])
        

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



def calc_IP2_equil(Imap, Qmap, Umap):

    Tlm = hp.map2alm( Imap )
    Elm,Blm = hp.map2alm_spin( (Qmap,Umap), 2)
    
    TEE = hp.alm2cl( Tlm, Elm**2 )
    TBB = hp.alm2cl( Tlm, Blm**2 )
    TEB = hp.alm2cl( Tlm, Elm*Blm )

    ls = np.arange( len(Elm) )
    return ls, TEE, TBB, TEB

def plot_IP2_equil(Imap_name='HFI_SkyMap_353_2048_R2.02_full.fits',
             Pmap_name='HFI_SkyMap_143_2048_R2.02_full.fits'):

    Imap = prepare_map( Imap_name )
    Qmap, Umap = prepare_map( Pmap_name, field=(1,2) )
    
    ls, TEE, TBB, TEB = calc_IP2_equil(Imap, Qmap, Umap)
    
    pl.figure()
    pl.plot( ls, TEE, lw=3, label='TEE' )
    pl.plot( ls, TBB, lw=3, label='TBB' )
    pl.plot( ls, TEB, lw=3, label='TEB' )
    pl.legend()
        


    
