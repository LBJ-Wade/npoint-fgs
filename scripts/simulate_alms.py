#!/usr/bin/env python
import sys
from process_fullsky import simulate_cmb_alms,PLANCK_DATA_PATH,FGS_SIM_PATH

filebase = sys.argv[1]
filename = FGS_SIM_PATH + 'cmb_alms/' + '{}_alm_planck.npy'.format(filebase)
simulate_cmb_alms(almfile=filename, nside=2048, lmax=4000,
                  cl_file=PLANCK_DATA_PATH+'bf_base_cmbonly_plikHMv18_TT_lowTEB_lmax4000.minimum.theory_cl')
