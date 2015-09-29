#!/usr/bin/env python
import sys
import numpy as np
from process_fullsky import PLANCK_DATA_PATH,FGS_SIM_PATH,simulate_observed_cmb,calc_alm,get_planck_mask
import healpy as hp

frequencies = [100,143,353,217]

lmax_bis = 200

mask = get_planck_mask(mask_percentage=60,
                mask_sources=True,
                apodization='0',
                smask_name='HFI_Mask_PointSrc_2048_R2.00.fits')

for f in frequencies:
    outfilename = FGS_SIM_PATH + 'planck_bispectrum_alms/' + 'data_planck_{}GHz'.format(f)
    mapname = PLANCK_DATA_PATH + 'HFI_SkyMap_{}_2048_R2.02_full.fits'.format(f)
    print 'reading map from file: {} ...'.format(mapname)
    I,Q,U = hp.read_map(mapname, field=(0,1,2))
    Tlm,Elm,Blm = calc_alm(I, Q, U, mask=mask,
                        lmax=lmax_bis,add_beam=None,div_beam=None,
                        healpy_format=False)
    print 'Calculated alms for {} GHz'.format(f)
    np.save(outfilename + '_tlm.npy', Tlm)
    np.save(outfilename + '_elm.npy', Elm)
    np.save(outfilename + '_blm.npy', Blm)
