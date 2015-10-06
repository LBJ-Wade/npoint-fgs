import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

foo="hi"

DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef np.complex128_t COMPLEX_t

from w3jslatec import drc3jm

cdef extern from 'math.h':
    double fabs(double)

@cython.boundscheck(False)
@cython.wraparound(False)
def w3j_ms(DTYPE_t l1, DTYPE_t l2, DTYPE_t l3,
        DTYPE_t m1):
    """
    For given l1, l2, l3, m1, this returns an array of wigner-3js 
    from m2 = -l2 to m2 = +l2, with m3 = -m2-m1


    For example, w3j_ms(2,2,2,0) will return:
    
        [(2,2,2,0,-2,2), (2,2,2,0,-1,1), (2,2,2,0,0,0), (2,2,2,0,1,-1), (2,2,2,0,2,-2)]

    """
    cdef long n, m2_min, m2_max
    n = int(l2 * 2) + 1
    
    cdef np.ndarray[DTYPE_t] result = np.empty(n, dtype=float)
    
    cdef int ier=0

    m2_min = int(-m1 - l3)
    if m2_min < -l2:
        m2_min = int(-l2)
    m2_max = int(-m1 + l3)
    if m2_max > l2:
        m2_max = int(l2)

    cdef np.ndarray[long] m2s = np.arange(m2_min, m2_max + 1)
    
    drc3jm(l1, l2, l3, m1, -n, n, result, ier)
    return result, m2s


#@cython.boundscheck(False)
#@cython.wraparound(False)
def inner_loops(long l1, long lmax, 
                np.ndarray[COMPLEX_t, ndim=3] bispectrum,
                np.ndarray[COMPLEX_t, ndim=2] alm1,
                np.ndarray[COMPLEX_t, ndim=2] alm2,
                np.ndarray[COMPLEX_t, ndim=2] alm3,
                np.ndarray[DTYPE_t, ndim=3] hs):
    cdef long i

    cdef np.ndarray[long] m1s = np.arange(-l1, l1+1)
    cdef long l2, l3, m1, m2, m3
    cdef long im1, im2, im3
    cdef np.ndarray[DTYPE_t] w3js #is it a problem that there is no size specified?
    cdef np.ndarray[long] m2s
    cdef DTYPE_t wig_factor
    cdef COMPLEX_t B



    for l2 in xrange(lmax+1):
        for l3 in xrange(lmax+1):
            if hs[l1,l2,l3] == 0.:
                continue
            for m1 in m1s:
                # This gives all the w3js you will need...
                w3js, m2s = w3j_ms(l1, l2, l3, m1)
                for i in xrange(len(m2s)):
                    wig_factor = w3js[i]
                    m2 = m2s[i]
                    m3 = -m2-m1
                    
                    # Fix negative indices to mean the right thing for alms
                    if m1 >= 0:
                        im1 = m1
                    else:
                        #because m1 is negative here, this will count from end
                        im1 = (2*lmax + 1) + m1  
                    if m2 >= 0:
                        im2 = m2
                    else:
                        im2 = (2*lmax + 1) + m2
                    if m3 >= 0:
                        im3 = m3
                    else:
                        im3 = (2*lmax + 1) + m3
                    B = alm1[l1,im1] * alm2[l2,im2] * alm3[l3,im3]
                    bispectrum[l1, l2, l3] = bispectrum[l1,l2,l3] + wig_factor * B
                
