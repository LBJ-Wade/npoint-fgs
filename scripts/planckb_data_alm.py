#!/usr/bin/env python
import sys
import numpy as np
from process_fullsky import PLANCK_DATA_PATH,calc_alm,get_planck_mask
import healpy as hp

frequencies = [353]#[100,143,353,217]

smear = True
mask_sources = True
nside=2048
psky = 70
apodization = 2

lmax_bispectrum = 199


mask = get_planck_mask(psky=psky,
                mask_sources=mask_sources,
                apodization=apodization,
                smask_name='HFI_Mask_PointSrc_2048_R2.00.fits')

for f in frequencies:
    outfilename = PLANCK_DATA_PATH + 'bispectrum_alms/' + 'data_planck_{}GHz_lmax{}'.format(f,lmax_bispectrum)
    mapname = PLANCK_DATA_PATH + 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(f)
    print 'reading map from file: {} ...'.format(mapname)
    I,Q,U = hp.read_map(mapname, field=(0,1,2))
    Tlm,Elm,Blm = calc_alm(I, Q, U, mask=mask,
                        lmax=lmax_bispectrum,add_beam=None,div_beam=None,
                        healpy_format=False)

    np.save(outfilename + '_Tlm.npy', Tlm)
    np.save(outfilename + '_Elm.npy', Elm)
    np.save(outfilename + '_Blm.npy', Blm)
