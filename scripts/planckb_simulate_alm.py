#!/usr/bin/env python
import sys
import numpy as np
from process_fullsky import PLANCK_DATA_PATH,FGS_SIM_PATH
from process_fullsky import simulate_observed_map,calc_alm,get_planck_mask

frequencies = [353]#[100,143,353,217]

filebase = sys.argv[1]

smear = True
save = False
nside=2048
lmax = 2000
lmax_bispectrum = 20

mask = get_planck_mask(psky=70,
                mask_sources=True,
                apodization=2,
                smask_name='HFI_Mask_PointSrc_2048_R2.00.fits')



#get the raw alms:
filename_cmb = FGS_SIM_PATH + 'alms/' + '{}_cmb_alm_lmax{}.npz'.format(filebase,lmax)
filename_fg = FGS_SIM_PATH + 'alms/' + '{}_fg_alm_lmax{}.npz'.format(filebase,lmax)
d = np.load(filename_cmb)
Tlm_cmb = d['Tlm']
Elm_cmb = d['Elm']
Blm_cmb = d['Blm']

alms_cmb = np.array([Tlm_cmb,Elm_cmb,Blm_cmb])

d = np.load(filename_fg)
Tlm_fg = d['Tlm']
Elm_fg = d['Elm']
Blm_fg = d['Blm']

alms_fg = np.array([Tlm_fg,Elm_fg,Blm_fg])

for f in frequencies:

    outfilename = PLANCK_DATA_PATH + 'bispectrum_alms/' + '{}_planck_{}GHz_lmax{}'.format(filebase,f,
                                                                                             lmax_bispectrum)
    I,Q,U = simulate_observed_map(alms_cmb=alms_cmb, alms_fg=alms_fg,
                                  frequency=f,
                                  smear=smear, experiment='planck',
                                  nside=nside, npix=None, lmax=lmax)
    print 'simulated maps for {}, {} GHz'.format(filebase, f)
    
    Tlm,Elm,Blm = calc_alm(I, Q, U, mask=mask,
                        lmax=lmax_bispectrum,add_beam=None,div_beam=None,
                        healpy_format=False)
    print 'Calculated alms for {}, {} GHz'.format(filebase, f)

    np.save(outfilename + '_Tlm', Tlm)
    np.save(outfilename + '_Elm', Elm)
    np.save(outfilename + '_Blm', Blm)
