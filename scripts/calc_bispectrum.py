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
from process_fullsky import FGS_SIM_PATH, FGS_RESULTS_PATH

alms_sims_path = FGS_SIM_PATH + 'planck_bispectrum_alms/'

# Use default communicator. No need to complicate things.
COMM = MPI.COMM_WORLD

parser = argparse.ArgumentParser()
parser.add_argument('--lmax',default=100, type=int)
parser.add_argument('--alm1',default='alm.npy')
parser.add_argument('--alm2',default=None)
parser.add_argument('--alm3',default=None)
parser.add_argument('--hfile',default='hs_lmax100.npy')
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
hs = get_hs(args.hfile, lmax=LMAX)


##### set up w3j calculation
lmaximum = 100
_wigxjpf_dir = os.path.expanduser('~/wigxjpf-1.0/lib')
_libwigxjpf = np.ctypeslib.load_library('libwigxjpf_shared', _wigxjpf_dir)
# Define argument and result types
_libwigxjpf.wig_table_init.argtypes = [ct.c_int]*2
_libwigxjpf.wig_table_init.restype = ct.c_void_p
_libwigxjpf.wig_table_free.argtypes = []
_libwigxjpf.wig_table_free.restype = ct.c_void_p
_libwigxjpf.wig_temp_init.argtypes = [ct.c_int]
_libwigxjpf.wig_temp_init.restype = ct.c_void_p
_libwigxjpf.wig_temp_free.argtypes = []
_libwigxjpf.wig_temp_free.restype = ct.c_void_p
_libwigxjpf.wig3jj.argtypes = [ct.c_int]*6
_libwigxjpf.wig3jj.restype = ct.c_double
wig3jj_c = _libwigxjpf.wig3jj
#set up memory for wig3j computations
_libwigxjpf.wig_table_init(2*lmaximum,3)
_libwigxjpf.wig_temp_init(2*lmaximum)
###### done with w3j setup

####### functions
@jit(nopython=True)
def wig3j(l1, l2, l3, m1, m2, m3):
    return wig3jj_c(l1*2, l2*2, l3*2,
                    m1*2, m2*2, m3*2)
                  

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


@jit(nopython=True)
def inner_loops(i, bispectrum):
    """
    bispectrum is (lmax+1)^3 array 
    """
    l1 = i
    m1s = np.arange(-l1, l1+1)
    for l2 in range(N):
        m2s = np.arange(-l2, l2+1)
        for l3 in range(N):
            m3s = np.arange(-l3, l3+1)
            for m1 in m1s:
                for m2 in m2s:
                    for m3 in m3s:  
                        if hs[l1, l2, l3] != 0. and m1 + m2 + m3 == 0:
                            bispectrum[l1, l2, l3] += wig3j(l1, l2, l3, m1, m2, m3) * alm1[l1,m1] * alm2[l2,m2] * alm3[l3,m3] #/ hs[l1, l2, l3]


                            
#initialize bispectrum to be empty
bispectrum = np.zeros((LMAX+1,LMAX+1,LMAX+1), dtype=complex)

for i in ns:
    print('on rank {}: i={}'.format(COMM.rank, i))
    inner_loops(i, bispectrum)

# Gather results on rank 0.
bispectrum = COMM.gather(bispectrum, root=0)

if COMM.rank == 0:
    bispectrum = np.array(bispectrum).sum(axis=0)
    np.save(filename,bispectrum)


#_libwigxjpf.wig_temp_free()
#_libwigxjpf.wig_table_free()
