#author: Vasundhara Komaragiri

import numpy as np
cimport numpy as cnp
cimport cython 


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef class Variable:
    cdef public int id, d, tval, val
    cdef int is_evid

    def __init__(self, int id_, int d_=0):
        self.id = id_
        self.d = d_
        self.tval = -1
        self.val = -1
        self.is_evid = 0

    cpdef int isEvid(self):
        return self.is_evid 

    cpdef void setValue(self, int v):
        if v >= 0 and v < self.d:
            self.val = v
        self.is_evid = 1
    
    cpdef int getValue(self):
        return self.val

    cpdef void removeValue(self):
        self.val = -1
        self.is_evid = 0
    
