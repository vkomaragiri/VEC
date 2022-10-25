#author: Vasundhara Komaragiri

import numpy as np
cimport numpy as cnp
cimport cython 
import sys
from numpy import logaddexp

from Variable import Variable
from Function import Function
from MN import MN
from Util import getMinDegreeOrder, getMinFillOrder, getMinFillOrderWithTreewidth
from Util import multiplyBucket, elimVarBucket
from Util import printVars, setAddr, getAddr

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef class BTP:
    cdef public int[:] order
    cdef public int[:] var_pos
    cdef double pe
    cdef public int upward_pass, downward_pass, init_done
    cdef public object[:] buckets
    cdef public int[:, :] messages
    cdef public object mn
    cdef public int nvars

    def __init__(self, object mn_ = None, int[:] order=None):
        self.pe = 1.0
        self.upward_pass = 0
        self.downward_pass = 0 
        self.init_done = 0

        self.order = order
        self.var_pos = None
        self.messages = None

        self.mn = mn_
        self.nvars = 0

        if mn_ != None:
            self.nvars = np.asarray(self.mn.variables, dtype=object).shape[0]
            self.initBTP()
        
    
    cpdef void getOrder(self, long order_type):
        cdef cnp.ndarray[int, ndim=1] temp  
        if order_type == 0: #Random order
            temp = np.arange(self.nvars, dtype=np.int32)
            np.random.shuffle(temp)
            self.order = temp
        elif order_type == 1: #Min-degree
            self.order = getMinDegreeOrder(self.mn.functions, self.nvars)
        elif order_type == 2: #Min-fill
            self.order = getMinFillOrder(self.mn.functions, self.nvars)

    cpdef void initBTP(self):
        cdef int nfunctions = np.asarray(self.mn.functions, dtype=object).shape[0]
        cdef int bucket_ind, i, newf_nvars, cur_ind
        cdef object newf
        cdef cnp.ndarray[int, ndim=1] newf_varIDs, var_pos
        cdef cnp.ndarray[object, ndim=1] buckets


        if self.order == None:
            self.getOrder(2)
        var_pos = -1*np.ones(self.nvars, dtype=np.int32)
        buckets = np.zeros(self.nvars, dtype=object)
        for i in range(self.nvars):
            var_pos[self.order[i]] = i
            buckets[i] = np.array([], dtype=object)
        for i in range(nfunctions):
            newf = self.mn.functions[i].instantiateEvid()
            newf_varIDs = np.asarray(newf.getVarIDs(), dtype=np.int32)
            newf_nvars = newf_varIDs.shape[0]
            if newf_nvars == 0:
                self.pe *= newf.getPotential()[0]
                continue
            bucket_ind = np.min(var_pos[newf_varIDs])
            buckets[bucket_ind] = np.hstack([buckets[bucket_ind], newf])
        self.var_pos = var_pos 
        self.buckets = buckets 
        self.init_done = 1

    cpdef void performUpwardPass(self): #leaves to root
        if self.upward_pass == 1:
            return
        if self.init_done == 0:
            self.initBTP()
        cdef cnp.ndarray[int, ndim=2] edges
        cdef int i, j, bucket_ind, nmarg_vars, cur_ind
        cdef cnp.ndarray[int, ndim=1] bucket_varIDs, marg_varIDs, var_pos
        cdef cnp.ndarray[double, ndim=1] bucket_potential, marg_potential
        cdef object func, bucket, marg

        var_pos = np.asarray(self.var_pos)
        edges = np.zeros((1, 4), dtype=np.int32)
        for i in range(self.nvars):
            if np.asarray(self.buckets[i]).shape[0] == 0:
                continue
            bucket = multiplyBucket(self.buckets[i], self.mn.variables)
            bucket_varIDs = np.asarray(bucket.getVarIDs())
            bucket_potential = np.asarray(bucket.getPotential())
            marg = elimVarBucket(bucket.getVarIDs(), bucket_potential, np.array([self.order[i]], dtype=np.int32), self.mn.variables)
            marg_varIDs = np.asarray(marg.getVarIDs())
            marg_potential = np.asarray(marg.getPotential())
            if marg_varIDs.shape[0] == 0:
                self.pe *= marg_potential[0]
                continue 
            bucket_ind = np.min(var_pos[marg_varIDs])
            edges = np.vstack([edges, np.array([i, bucket_ind, self.buckets[bucket_ind].shape[0], -1], dtype=np.int32)])
            self.buckets[bucket_ind] = np.hstack([self.buckets[bucket_ind], marg])
        edges = edges[1:, :]
        self.messages = edges
        self.upward_pass = 1
       
    cpdef void performDownwardPass(self): #root to leaves
        if self.downward_pass == 1:
            return 
        if self.upward_pass == 0:
            self.performUpwardPass()
        cdef int i, child, par, msg_ind, j, k, msg2_ind
        cdef int nmessages = np.asarray(self.messages, dtype=np.int32).shape[0]
        cdef object[:] bucket, parent_functions
        cdef cnp.ndarray[int, ndim=1] sep_vars, parent_vars, marg_vars, elim_vars
        cdef object mult, marg

        for i in range(nmessages-1, -1, -1):
            child, par, msg_ind, msg2_ind = self.messages[i]
            bucket = np.asarray(self.buckets[par], dtype=object)
            parent_vars = np.array([], dtype=np.int32)
            parent_functions = np.array([], dtype=object)
            sep_vars = np.array([], dtype=np.int32)
            for j in range(bucket.shape[0]):
                if j == msg_ind:
                    sep_vars = np.hstack([sep_vars, bucket[j].getVarIDs()])
                    continue 
                parent_vars = np.hstack([parent_vars, bucket[j].getVarIDs()])
                parent_functions = np.hstack([parent_functions, bucket[j]])
            mult = multiplyBucket(parent_functions, self.mn.variables)
            parent_vars = np.unique(parent_vars)
            elim_vars = np.setdiff1d(parent_vars, sep_vars)
            marg = elimVarBucket(parent_vars, mult.getPotential(), elim_vars, self.mn.variables)
            self.messages[i][3] = np.asarray(self.buckets[child], dtype=object).shape[0]
            self.buckets[child] = np.hstack([np.asarray(self.buckets[child], dtype=object), marg])
        self.downward_pass = 1
        
    cpdef double getPR(self):
        if self.upward_pass == 0:
            self.performUpwardPass()
        return self.pe

    
    
