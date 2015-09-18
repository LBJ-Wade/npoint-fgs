#!/usr/bin/env python

from mpi4py import MPI
import ctypes as ct
import numpy as np
import os, sys, os.path

from numba import jit

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
LMAX = 100

_libwigxjpf.wig_table_init(2*LMAX,3)
_libwigxjpf.wig_temp_init(2*LMAX)


@jit(nopython=True)
def wig3j(l1, l2, l3, m1, m2, m3):
    return wig3jj_c(l1*2, l2*2, l3*2,
                    m1*2, m2*2, m3*2)
                  
# Use default communicator. No need to complicate things.
COMM = MPI.COMM_WORLD

def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]


#########
#########
lmax = 100
frequency = 353

#assert type(id) is IntType, "id is not an integer: %r" % id
########
########

from explore import get_hs,observe_alms#, get_Tlm,get_ElmBlm
import healpy as hp

hs = get_hs('hs_lmax100.npy')
Tlm,Elm,Blm = observe_alms(filetag='test{}'.format(frequency),
                           frequency=frequency,simulate=False,
                           lmax_save=lmax)

#Tlm = get_Tlm('Tlm_lmax{}_test143.npy'.format(lmax))
#Elm,Blm = get_ElmBlm('ElmBlm_lmax{}_test143.npy'.format(lmax))

#N = len(Tlm)
N = lmax

#sys.exit()

if COMM.rank == 0:
    ns = range(N)
    # Split into however many cores are available.
    ns = split(ns, COMM.size)
else:
    ns = None

# Scatter jobs across cores.
ns = COMM.scatter(ns, root=0)


@jit#(nopython=True)
def inner_loops(i, bispectrum, Tlm, Elm, Blm, hs):
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
                            bispectrum[l1, l2, l3] += (wig3j(l1, l2, l3, m1, m2, m3) *
                                                        Tlm[l1][m1] * Elm[l2][m2] * Blm[l3][m3]) / hs[l1, l2, l3]
#initialize bispectrum to be empty
bispectrum = np.zeros((lmax+1,lmax+1,lmax+1), dtype=complex)

for i in ns:
    print('on rank {}: i={}'.format(COMM.rank, i))
    inner_loops(i, bispectrum, Tlm, Elm, Blm, hs)

# Gather results on rank 0.
bispectrum = COMM.gather(bispectrum, root=0)

if COMM.rank == 0:
    bispectrum = np.array(bispectrum).sum(axis=0)
    np.save('bispectrum_test353_lmax{}.npy'.format(lmax),bispectrum)


_libwigxjpf.wig_temp_free()
_libwigxjpf.wig_table_free()
