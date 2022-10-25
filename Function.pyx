#author: Vasundhara Komaragiri


import numpy as np 
cimport numpy as cnp
cimport cython 
from Variable import Variable
from Util import getDomainSize, setAddr, getAddr, convertProbCPT


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef class Function:
    cdef object[:] variables
    cdef public double[:] potentials 

    def __init__(self, variables_ = None, potentials_ = None):
        self.variables = variables_
        self.potentials = potentials_

    cpdef void setVars(self, cnp.ndarray[object, ndim=1] vars):
        self.variables = vars

    cpdef object[:] getVars(self):
        return self.variables

    cpdef int[:] getVarIDs(self):
        cdef cnp.ndarray[int, ndim=1] varIDs 
        cdef int i, nvars = np.asarray(self.variables, dtype=object).shape[0]
        varIDs = -1*np.zeros(nvars, dtype=np.int32)
        for i in range(nvars):  
            varIDs[i] = self.variables[i].id 
        return varIDs

    cpdef void setPotential(self, cnp.ndarray[double, ndim=1] potentials_):
        self.potentials = potentials_

    cpdef double[:] getPotential(self):
        return self.potentials

    cpdef void setCPTVar(self, int cvar):
        self.cpt_var_ind = cvar

    cpdef int getCPTVar(self):
        return self.cpt_var_ind

    cpdef object instantiateEvid(self):
        cdef object out = Function()
        cdef int i, j, d_non_evid, d, nvars
        cdef cnp.ndarray[object, ndim=1] non_evid_vars
        cdef cnp.ndarray[double, ndim=1] temp
        
        nvars = np.asarray(self.variables, dtype=object).shape[0]
        non_evid_vars = np.asarray([], dtype=object)
        for i in range(nvars):
            if self.variables[i].isEvid() == 1:
                self.variables[i].tval = self.variables[i].val 
            else:
                non_evid_vars = np.hstack([non_evid_vars, np.asarray([self.variables[i]], dtype=object)])
        if non_evid_vars.shape[0] == nvars:
            return self
        else:
            out.setVars(non_evid_vars)
            d = getDomainSize(non_evid_vars)
            temp = -1*np.ones(d)
            for i in range(d):
                setAddr(non_evid_vars, i)
                temp[i] = self.potentials[getAddr(self.variables)]
            out.setPotential(temp)
        return out 


    
