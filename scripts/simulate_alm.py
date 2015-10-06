#!/usr/bin/env python
import sys
import numpy as np
from process_fullsky import simulate_alms,get_theory_cmb,PLANCK_DATA_PATH,FGS_SIM_PATH
from foregrounds import get_theory_fg

lmax = 2000
nside = 2048

filebase = sys.argv[1]
filename_cmb = FGS_SIM_PATH + 'alms/' + '{}_cmb_alm_lmax{}'.format(filebase,lmax)
filename_fg = FGS_SIM_PATH + 'alms/' + '{}_fg_alm_lmax{}'.format(filebase,lmax)

ls, cls_theory_cmb = get_theory_cmb(lmax=lmax)
ls, cls_theory_fg = get_theory_fg(lmax=lmax)

Tlm_cmb, Elm_cmb, Blm_cmb = simulate_alms(cls_theory_cmb, nside=nside, lmax=lmax)
Tlm_fg, Elm_fg, Blm_fg = simulate_alms(cls_theory_fg, nside=nside, lmax=lmax)

np.savez(filename_cmb, Tlm=Tlm_cmb, Elm=Elm_cmb, Blm=Blm_cmb)
np.savez(filename_fg, Tlm=Tlm_fg, Elm=Elm_fg, Blm=Blm_fg)
