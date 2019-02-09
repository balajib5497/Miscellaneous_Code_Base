cdef extern from "math.h":
    double sqrt(double x)
    double ceil(double x)
    double log(double x)

from numpy cimport ndarray
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int partition(int* arr, int low, int high): 
    cdef int i = low - 1
    cdef int j     
    cdef int pivot = arr[high]
  
    for j in range(low , high): 
        if arr[j] <= pivot: 
            i = i+1 
            arr[i], arr[j] = arr[j], arr[i] 
  
    arr[i+1], arr[high] = arr[high], arr[i+1] 
    return i+1 
  
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void quick_sort(int* arr, int low, int high): 
    cdef int pi
    if low < high: 
        pi = partition(arr, low, high) 
        quick_sort(arr, low, pi-1) 
        quick_sort(arr, pi+1, high)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sort(int[:] arr):
    quick_sort(&arr[0], 0, arr.shape[0]-1)
    return np.asarray(arr)

@cython.boundscheck(False)
@cython.wraparound(False)  
cpdef (double, double) mean_and_std_int(int[:] arr):
    cdef int i
    cdef int n = arr.shape[0]
    cdef double m = 0.0
    
    for i in range(n):
        m += arr[i]
    m /= n
    
    cdef double v = 0.0
    for i in range(n):
        v += (arr[i] - m)**2
    
    v = sqrt(v / n)
    return m, v

@cython.boundscheck(False)
@cython.wraparound(False)  
cpdef (double, double) mean_and_std_float(float[:] arr):
    cdef int i
    cdef int n = arr.shape[0]
    cdef double m = 0.0
    
    for i in range(n):
        m += arr[i]
    m /= n
    
    cdef double v = 0.0
    for i in range(n):
        v += (arr[i] - m)**2
    
    v = sqrt(v / n)
    return m, v

@cython.boundscheck(False)
@cython.wraparound(False)  
cpdef double percentile(int[:] arr, int n, double percent):
    cdef double k = (n-1) * percent
    cdef int f = int(k)
    cdef int c = int(ceil(k))
    
    if f == c:
        return arr[f]
    
    cdef double d0 = arr[f] * (c-k)
    cdef double d1 = arr[c] * (k-f)
    return d0 + d1

@cython.boundscheck(False)
@cython.wraparound(False)  
cpdef double median(int[:] arr):
    cdef int n = arr.shape[0]
    cdef double res
    
    if n%2 == 1:
        res = arr[n//2]
        return res

    return (arr[(n-1) // 2] + arr[n//2]) / 2
