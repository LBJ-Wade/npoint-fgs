#!/usr/bin/env python

# This script computes the m-averaged bispectrum b. 
# inputs: names of alm1, alm2, and alm3 files, and lmax

from mpi4py import MPI
import ctypes as ct
import numpy as np
import os, sys, os.path
import argparse
from numba import jit

from spherical_geometry import get_hs
from process_fullsky import FGS_SIM_PATH, FGS_RESULTS_PATH, PLANCK_DATA_PATH
from test import inner_loops

alms_sims_path = PLANCK_DATA_PATH + 'bispectrum_alms/'

# Use default communicator. No need to complicate things.
COMM = MPI.COMM_WORLD

parser = argparse.ArgumentParser()
parser.add_argument('--lmax',default=200, type=int)
parser.add_argument('--alm1',default='alm.npy')
parser.add_argument('--alm2',default=None)
parser.add_argument('--alm3',default=None)
parser.add_argument('--filename',default='bispectrumtest.npy')

args = parser.parse_args()
LMAX = args.lmax

filename = FGS_RESULTS_PATH + 'bispectra/' + args.filename

# fetch alms:
alm1 = np.load(alms_sims_path+args.alm1)
if args.alm2 is not None:
    alm2 = np.load(alms_sims_path+args.alm2)
else:
    alm2 = alm1
if args.alm3 is not None:
    alm3 = np.load(alms_sims_path+args.alm3)
else:
    alm3 = alm1

#assert len(alm1)==len(alm2) and len(alm1)==len(alm3) and len(alm1)==LMAX+1, 'problem: alm size(s) and lmax mismatch.'
    
# fetch w3j's for ms=(0,0,0)s
hs = get_hs(lmax=LMAX)


def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

#########
##########
###########
N = LMAX + 1
if COMM.rank == 0:
    ns = range(N)
    # Split into however many cores are available.
    ns = split(ns, COMM.size)
else:
    ns = None

# Scatter jobs across cores.
ns = COMM.scatter(ns, root=0)

                       
#initialize bispectrum to be empty
bispectrum = np.zeros((LMAX+1,LMAX+1,LMAX+1), dtype=complex)

for i in ns:
    print('on rank {}: i={}'.format(COMM.rank, i))
    inner_loops(i, args.lmax, bispectrum, alm1, alm2, alm3, hs=hs)

# Gather results on rank 0.
bispectrum = COMM.gather(bispectrum, root=0)

if COMM.rank == 0:
    bispectrum = np.array(bispectrum).sum(axis=0)
    nonzero = ~np.isclose(hs,0.)
    bispectrum[nonzero] = bispectrum[nonzero] / hs[nonzero]
    np.save(filename,bispectrum.real)




