#!/usr/bin/env python
import sys
import numpy as np
from process_fullsky import PLANCK_DATA_PATH,FGS_SIM_PATH,simulate_observed_cmb,calc_alm,get_planck_mask

frequencies = [100,143,353,217]

filebase = sys.argv[1]
infilename = FGS_SIM_PATH + 'cmb_alms/' + '{}_alm_planck.npz'.format(filebase)

smear = True
save = False
nside=2048
lmax = 2000
lmax_bis = 100

mask = get_planck_mask(mask_percentage=60,
                mask_sources=True,
                apodization='0',
                smask_name='HFI_Mask_PointSrc_2048_R2.00.fits')

for f in frequencies:
    outfilename = FGS_SIM_PATH + 'planck_bispectrum_alms/' + '{}_planck_{}GHz'.format(filebase,f)
    I,Q,U = simulate_observed_cmb(frequency=f, almfile=infilename,
                      smear=smear, save=save, experiment='planck',
                      nside=nside, npix=None, lmax=lmax)
    print 'simulated maps for {}, {} GHz'.format(filebase, f)
    Tlm,Elm,Blm = calc_alm(I, Q, U, mask=mask,
                        lmax=lmax_bis,add_beam=None,div_beam=None,
                        healpy_format=False)
    print 'Calculated alms for {}, {} GHz'.format(filebase, f)
    np.save(outfilename + '_tlm.npy', Tlm)
    np.save(outfilename + '_elm.npy', Elm)
    np.save(outfilename + '_blm.npy', Blm)
