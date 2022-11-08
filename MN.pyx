#author: Vasundhara Komaragiri

import numpy as np
cimport numpy as cnp
cimport cython 
import sys

from Variable import Variable
from Function import Function
from Util import setAddr, getAddr, getDomainSize, getProb, getPairwiseProb, printVars
from Util import multiplyBucket, elimVarBucket

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef class MN:
    cdef public object[:] variables
    cdef public object[:] functions
    cdef public int nvars

    def __init__(self, vars_=None, funcs_=None):
        self.variables = vars_
        self.functions = funcs_ 
        if vars_ == None:
            self.nvars = 0
        else:
            self.nvars = np.asarray(self.variables).shape[0]

    cpdef double getLogProbability(self, cnp.ndarray[int, ndim=1] example):
        cdef double prob = 1.0, temp
        cdef int i 
        cdef object func 
        for i in range(self.nvars):
            self.variables[i].tval = example[self.variables[i].id]
        for i in range(self.nvars):
            func = self.functions[i]
            temp = func.getPotential()[getAddr(func.getVars())]
            prob *= temp
        return np.log(prob)
    
    cpdef getProbability(self, cnp.ndarray[int, ndim=1] example):
        return np.exp(self.getLogProbability(example))
    
    cpdef double getLLScore(self, cnp.ndarray[int, ndim=2] data):
        cdef double ll_score = 0.0
        cdef int i
        for i in range(data.shape[0]):
            ll_score += self.getLogProbability(data[i, :])
        return ll_score/data.shape[0]
    
    cpdef void write(self, outfilename):
        cdef int i, nfunctions, j
        cdef object func
        cdef object[:] func_vars
        cdef cnp.ndarray[double, ndim=1] func_potentials

        fw = open(outfilename, "w")
        fw.write("MARKOV\n")
        nfunctions = np.asarray(self.functions, dtype=object).shape[0]
        fw.write(str(self.nvars)+"\n")
        for i in range(self.nvars):
            fw.write(str(self.variables[i].d)+" ")
        fw.write("\n\n")
        fw.write(str(nfunctions)+"\n")
        for i in range(nfunctions):
            func = self.functions[i]
            func_vars = func.getVars()
            fw.write(str(func_vars.shape[0])+" ")
            for j in range(func_vars.shape[0]):
                fw.write(str(func_vars[j].id)+" ")
            fw.write("\n")
        fw.write("\n")
        for i in range(nfunctions):
            func = self.functions[i]
            func_potentials = np.asarray(func.getPotential(), dtype=float)
            fw.write(str(func_potentials.shape[0])+"\n")
            for j in range(func_potentials.shape[0]):
                fw.write("{0:.8f} ".format(func_potentials[j]))
            fw.write("\n")

    cpdef void read(self, infilename):
        cdef int nfunctions, i, j, n
        cdef double k
        cdef cnp.ndarray[int, ndim=1] dsize 
        cdef object var
        cdef cnp.ndarray[object, ndim=1] variables, functions, func_vars

        fr = open(infilename, "r")
        if "MARKOV" not in fr.readline():
            print("Invalid input")
            sys.exit(0)
        line = fr.readline()
        self.nvars = int(line[:(len(line)-1)])
        
        variables = np.zeros(self.nvars, dtype=object)
        line = fr.readline()
        dsize = np.array(line.split(" "), dtype=np.int32)
        for i in range(self.nvars):
            variables[i] = Variable(i, dsize[i])
        self.variables = variables 
        fr.readline()
        line = fr.readline()
        line = line[:(len(line)-1)]
        nfunctions = np.int32(line)
        functions = np.zeros(nfunctions, dtype=object)
        for i in range(nfunctions):
            func = Function()
            line = fr.readline()
            vars_ = np.array(line.split(" "), dtype=np.int32)
            func_vars = variables[vars_[1:]]
            func.setVars(func_vars)
            functions[i] = func
        fr.readline()
        for i in range(nfunctions):
            line = fr.readline()
            potentials_ = np.array(line.split(" "), dtype=float)
            functions[i].setPotential(potentials_[1:])
        self.functions = functions

    cpdef void setEvidence(self, int var, int val):
        self.variables[var].setValue(val)

    cpdef void removeEvidence(self, int var):
        self.variables[var].removeValue()
            
