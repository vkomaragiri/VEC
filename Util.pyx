import numpy as np 
cimport numpy as cnp
from igraph import Graph
cimport cython 
from scipy import stats

from Variable import Variable
from Function import Function


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef getPairwiseProb(cnp.ndarray[int, ndim=2] data, int x, int y, int x_d, int y_d, cnp.ndarray[double, ndim=1] weights, double laplace):
    cdef cnp.ndarray[double, ndim=2] cxy 
    cdef int i, j 

    cxy = np.zeros(shape=(x_d, y_d))+laplace
    for i in range(x_d):
        for j in range(y_d):
            cxy[i][j] += np.sum(weights[(data[:, x] == i) & (data[:, y] == j)])
    cxy /= np.sum(cxy)
    cxy[cxy < 0.0001] = 0.0001
    cxy[cxy > 0.9999] = 0.9999
    cxy /= np.sum(cxy)
    return cxy

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef getProb(cnp.ndarray[int, ndim=2] data, int x, int x_d, cnp.ndarray[double, ndim=1] weights, double laplace):
    cdef cnp.ndarray[double, ndim=1] cx 
    cdef int i

    cx = np.zeros(shape=(x_d))+laplace
    for i in range(x_d):
        cx[i] += np.sum(weights[data[:, x] == i])
    cx /= np.sum(cx)
    cx[cx < 0.0001] = 0.0001
    cx[cx > 0.9999] = 0.9999
    cx /= np.sum(cx)
    return cx


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef list computeMI(object[:] variables, cnp.ndarray[int, ndim=2] data, cnp.ndarray[double, ndim=1] weights, double laplace):
    cdef int i, j, nvars, xu, xv 
    cdef list px, pxy 
    cdef object u, v
    nvars = variables.shape[0]
    px = [None]*nvars
    for i in range(nvars):
        u = variables[i]
        px[i] = getProb(data, i, u.d, weights, laplace)
    cdef cnp.ndarray[double, ndim = 2] mi = np.zeros((nvars, nvars), dtype=float)
    pxy = [None]*nvars
    for i in range(nvars):
        u = variables[i]
        pxy[i] = [None]*nvars
        for j in range(nvars):
            v = variables[j]
            pxy[i][j] = getPairwiseProb(data, i, j, u.d, v.d, weights, laplace)
            if i == j:
                continue
            for xu in range(u.d):
                for xv in range(v.d):
                    mi[i][j] += pxy[i][j][xu][xv]*(np.log(pxy[i][j][xu][xv])-np.log(px[i][xu])-np.log(px[j][xv]))
    return [mi, pxy, px]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int getAddr(object[:] variables):
    cdef int ind = 0, multiplier = 1
    cdef int i
    for i in range(variables.shape[0]-1, -1, -1):
        ind += variables[i].tval*multiplier
        multiplier *= variables[i].d 
    return ind

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void setAddr(object[:] variables, int ind):
    cdef int i 
    for i in range(variables.shape[0]-1, -1, -1):
        variables[i].tval = ind % variables[i].d
        ind /= variables[i].d

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int getDomainSize(object[:] variables):
    cdef int d = 1, i
    for i in range(variables.shape[0]):
        d *= variables[i].d
    return d 

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef list getDirectedST(cnp.ndarray[double, ndim=2] adj_mat):
    cdef int i, j
    cdef cnp.ndarray[double, ndim=1] wts
    cdef list children, weights, temp_children, temp_parents, parents
    cdef list component_vertices

    g = Graph.Weighted_Adjacency(adj_mat)
    wts = adj_mat[adj_mat!=0].flatten()
    tree = g.spanning_tree(weights=wts)    
    tree.to_undirected()
    children = []
    parents = []
    temp_children = []
    temp_parents = []
    components = tree.connected_components()
    component_vertices = list(components)
    if(len(component_vertices)) == 1:
        children, parents = tree.dfs(vid=0)
        return [children, parents]
    subgraphs = components.subgraphs()
    for i in range(len(component_vertices)):  
        temp_children, temp_parents = subgraphs[i].dfs(vid = subgraphs[i].vs.indices[0])
        for j in range(len(temp_children)):
            if temp_parents[j] == -1:
                parents.append(-1)
            else:
                parents.append(component_vertices[i][temp_parents[j]])
            children.append(component_vertices[i][temp_children[j]])
        #children.extend([component_vertices[i][temp_children[j]] for j in range(len(temp_children))])
        #parents.extend([component_vertices[i][temp_parents[j]] for j in range(len(temp_parents))])
    return [children, parents]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int[:] getMinDegreeOrder(object[:] functions, int nvars=-1, int[:]id_index=None):
    cdef cnp.ndarray[int, ndim=1] order
    cdef cnp.ndarray[int, ndim=2] adj_mat
    cdef int nfunctions = np.asarray(functions, dtype=object).shape[0], i, j, k, n, min_ind, min_degree, flag = 0, cur_degree
    cdef cnp.ndarray[int, ndim=1] temp_vars, all_vars, indices, triangulate_vertices
    
    if id_index == None:
        if nvars == -1:
            nvars = nfunctions
        order = -1*np.ones(nvars, dtype=np.int32)
        adj_mat = np.zeros((nvars, nvars), dtype=np.int32)
        for i in range(nfunctions):
            temp_vars = np.asarray(functions[i].getVarIDs(), dtype=np.int32)
            for j in range(temp_vars.shape[0]):
                for k in range(j+1, temp_vars.shape[0]):
                    adj_mat[temp_vars[j]][temp_vars[k]] = 1
                    adj_mat[temp_vars[k]][temp_vars[j]] = 1
        all_vars = np.asarray(np.arange(nvars), dtype=np.int32)
        for i in range(nvars):
            indices = np.setdiff1d(all_vars, order)
            order[i] = int(indices[np.argmin(np.sum(adj_mat[indices, :], axis=1))])
            triangulate_vertices = np.setdiff1d(all_vars[adj_mat[order[i], :] > 0], order)
            for j in range(triangulate_vertices.shape[0]):
                for k in range(j+1, triangulate_vertices.shape[0]):
                    adj_mat[triangulate_vertices[j]][triangulate_vertices[k]] = 1
                    adj_mat[triangulate_vertices[k]][triangulate_vertices[j]] = 1
        return order
    else:
        if nvars == -1:
            nvars = nfunctions
        order = -1*np.ones(nvars, dtype=np.int32)
        adj_mat = np.zeros((nvars, nvars), dtype=np.int32)
        for i in range(nfunctions):
            temp_vars = np.asarray(functions[i].getVarIDs(), dtype=np.int32)
            for j in range(temp_vars.shape[0]):
                for k in range(j+1, temp_vars.shape[0]):
                    adj_mat[id_index[temp_vars[j]]][id_index[temp_vars[k]]] = 1
                    adj_mat[id_index[temp_vars[k]]][id_index[temp_vars[j]]] = 1
        all_vars = np.asarray(np.arange(nvars), dtype=np.int32)
        for i in range(nvars):
            indices = np.setdiff1d(all_vars, order)
            order[i] = int(indices[np.argmin(np.sum(adj_mat[indices, :], axis=1))])
            triangulate_vertices = np.setdiff1d(all_vars[adj_mat[order[i], :] > 0], order)
            for j in range(triangulate_vertices.shape[0]):
                for k in range(j+1, triangulate_vertices.shape[0]):
                    adj_mat[triangulate_vertices[j]][triangulate_vertices[k]] = 1
                    adj_mat[triangulate_vertices[k]][triangulate_vertices[j]] = 1
        return order

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int[:] getMinFillOrder(object[:] functions, int nvars=-1, int[:]id_index=None):
    cdef cnp.ndarray[int, ndim=1] order
    cdef cnp.ndarray[int, ndim=2] adj_mat
    cdef int nfunctions = np.asarray(functions, dtype=object).shape[0], i, j, k, n, min_ind, min_fill, flag = 0, cur_fill
    cdef cnp.ndarray[int, ndim=1] temp_vars 
    cdef cnp.ndarray[int, ndim=1] triangulate_vertices, indices, on_indices
    
    if id_index == None:
        if nvars == -1:
            nvars = nfunctions
        adj_mat = np.zeros((nvars, nvars), dtype=np.int32)
        order = -1*np.ones(nvars, dtype=np.int32)
        for i in range(nfunctions):
            temp_vars = np.asarray(functions[i].getVarIDs(), dtype=np.int32)
            for j in range(temp_vars.shape[0]):
                for k in range(j+1, temp_vars.shape[0]):
                    adj_mat[temp_vars[j]][temp_vars[k]] = 1
                    adj_mat[temp_vars[k]][temp_vars[j]] = 1
        all_vars = np.asarray(np.arange(nvars), dtype=np.int32)
        for n in range(nvars):
            indices = np.setdiff1d(all_vars, order)
            min_fill = nvars 
            min_ind = -1
            triangulate_vertices = -1*np.ones(1, dtype=np.int32)
            for i in range(indices.shape[0]):
                on_indices = np.intersect1d(indices, all_vars[adj_mat[indices[i], :]>0])
                cur_fill = 0
                for j in range(on_indices.shape[0]):
                    for k in range(j+1, on_indices.shape[0]):
                        if adj_mat[on_indices[j]][on_indices[k]] == 0:
                            cur_fill += 1
                if cur_fill < min_fill:
                    min_fill = cur_fill 
                    min_ind = indices[i]
                    triangulate_vertices = on_indices 
            order[n] = min_ind 
            for j in range(triangulate_vertices.shape[0]):
                for k in range(j+1, triangulate_vertices.shape[0]):
                    adj_mat[triangulate_vertices[j]][triangulate_vertices[k]] = 1
                    adj_mat[triangulate_vertices[k]][triangulate_vertices[j]] = 1
        return order    

    else:
        if nvars == -1:
            nvars = nfunctions
        adj_mat = np.zeros((nvars, nvars), dtype=np.int32)
        order = -1*np.ones(nvars, dtype=np.int32)
        for i in range(nfunctions):
            temp_vars = np.asarray(functions[i].getVarIDs(), dtype=np.int32)
            for j in range(temp_vars.shape[0]):
                for k in range(j+1, temp_vars.shape[0]):
                    adj_mat[id_index[temp_vars[j]]][id_index[temp_vars[k]]] = 1
                    adj_mat[id_index[temp_vars[k]]][id_index[temp_vars[j]]] = 1
        all_vars = np.asarray(np.arange(nvars), dtype=np.int32)
        for n in range(nvars):
            indices = np.setdiff1d(all_vars, order)
            min_fill = nvars 
            min_ind = -1
            triangulate_vertices = -1*np.ones(1, dtype=np.int32)
            for i in range(indices.shape[0]):
                on_indices = np.intersect1d(indices, all_vars[adj_mat[indices[i], :]>0])
                cur_fill = 0
                for j in range(on_indices.shape[0]):
                    for k in range(j+1, on_indices.shape[0]):
                        if adj_mat[on_indices[j]][on_indices[k]] == 0:
                            cur_fill += 1
                if cur_fill < min_fill:
                    min_fill = cur_fill 
                    min_ind = indices[i]
                    triangulate_vertices = on_indices 
            order[n] = min_ind 
            for j in range(triangulate_vertices.shape[0]):
                for k in range(j+1, triangulate_vertices.shape[0]):
                    adj_mat[triangulate_vertices[j]][triangulate_vertices[k]] = 1
                    adj_mat[triangulate_vertices[k]][triangulate_vertices[j]] = 1
        return order    

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef object multiplyBucket(object[:] bucket, object[:] vars_, int[:]id_ind=None):
    cdef int i, nfunctions, j
    cdef cnp.ndarray[int, ndim=1] bucket_varIDs = np.array([], dtype=np.int32), id_index
    cdef cnp.ndarray[object, ndim=1] bucket_vars = np.array([], dtype=object), variables = np.asarray(vars_)
    cdef cnp.ndarray[double, ndim=1] bucket_potential
    cdef object func, out
    cdef object[:] temp_vars
    cdef double v 

    if id_ind != None:
        id_index = np.asarray(id_ind)
    nfunctions = np.asarray(bucket).shape[0]
    if nfunctions == 1:
        bucket_vars = np.asarray(bucket[0].getVars())
        bucket_potential = np.asarray(bucket[0].getPotential())
    else:
        bucket_vars = np.array([], dtype=object)
        for i in range(nfunctions):
            func = bucket[i]
            bucket_varIDs = np.hstack([bucket_varIDs, func.getVarIDs()])
        bucket_varIDs = np.unique(bucket_varIDs)
        if id_ind == None:
            bucket_vars = variables[bucket_varIDs]
        else:
            bucket_vars = variables[id_index[bucket_varIDs]]

        d = getDomainSize(bucket_vars)
        bucket_potential = np.ones(d)
        for i in range(d):
            setAddr(bucket_vars, i)
            for j in range(nfunctions):
                func = bucket[j]
                temp_vars = func.getVars()
                v = func.getPotential()[getAddr(temp_vars)]
                #if v == 0:
                #    continue
                bucket_potential[i] *= v
        #bucket_potential /= np.sum(bucket_potential)
    out = Function()
    out.setVars(bucket_vars)
    out.setPotential(bucket_potential)
    return out

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef object elimVarBucket(int[:] bucket_varIDs, double[:] bucket_potential, int[:] elim_varIDs, object[:] vars_, int[:]id_ind=None):
    cdef int i, j , marg_d, elim_d
    cdef cnp.ndarray[double, ndim=1] marg_potential
    cdef cnp.ndarray[int, ndim=1] marg_varIDs, id_index
    cdef cnp.ndarray[object, ndim=1] marg_vars, elim_vars, bucket_vars, variables = np.asarray(vars_)
    cdef object out 
    cdef double v 
    
    if id_ind != None:
        id_index = np.asarray(id_ind)
    marg_varIDs = np.setdiff1d(np.asarray(bucket_varIDs, dtype=np.int32), np.asarray(elim_varIDs, dtype=np.int32))
    if id_ind == None:
        marg_vars = variables[np.asarray(marg_varIDs)]
        elim_vars = variables[np.asarray(elim_varIDs)]
        bucket_vars = variables[np.asarray(bucket_varIDs)]
    else:
        marg_vars = variables[id_index[np.asarray(marg_varIDs)]]
        elim_vars = variables[id_index[np.asarray(elim_varIDs)]]
        bucket_vars = variables[id_index[np.asarray(bucket_varIDs)]]

    marg_d = getDomainSize(marg_vars)
    marg_potential = np.zeros(marg_d)
    elim_d = bucket_potential.shape[0]//marg_d
    for j in range(marg_d):
        setAddr(marg_vars, j)
        for i in range(elim_d):
            setAddr(elim_vars, i)
            v = bucket_potential[getAddr(bucket_vars)]
            #if v == 0:
            #    continue
            marg_potential[j] += v
    #marg_potential /= np.sum(marg_potential)
    out = Function()
    out.setVars(marg_vars)
    out.setPotential(marg_potential)
    return out

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void printVars(object[:] vars_):
    cdef int nvars = np.asarray(vars_, dtype=object).shape[0], i 
    for i in range(nvars):
        print vars_[i].id,
    print("")

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double[:] convertProbCPT(double[:] prob, object[:] variables, int cpt_var_ind):
    cdef cnp.ndarray[double, ndim=1] cpt
    cdef int i, d, ind, j
    cdef cnp.ndarray[object, ndim=1] other_vars
    cdef object cpt_var
    cdef double norm_const, temp_prob
    
    cpt_var = variables[cpt_var_ind]
    other_vars = np.array([], dtype=object)
    for i in range(np.asarray(variables).shape[0]):
        if i == cpt_var_ind:
            continue 
        other_vars = np.hstack([other_vars, variables[i]])
    d = np.asarray(prob).shape[0]
    cpt = np.zeros(d)
    for i in range(d//cpt_var.d):
        setAddr(other_vars, i)
        norm_const = 0.0 
        for j in range(cpt_var.d):
            cpt_var.tval = j 
            ind = getAddr(variables)
            cpt[ind] = prob[ind]
            norm_const += prob[ind]
        for j in range(cpt_var.d):
            cpt_var.tval = j 
            ind = getAddr(variables)
            cpt[ind] /= norm_const
    return cpt

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double computeEntropy(list px):
    cdef double entropy = 0.0
    cdef int i, n = len(px)

    for i in range(n):
        entropy += stats.entropy(px[i])
    entropy /= n
    return entropy 


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int[:, :] getMinFillOrderWithTreewidth(object[:] functions, int nvars):
    cdef cnp.ndarray[int, ndim=2] order
    cdef cnp.ndarray[int, ndim=2] adj_mat
    cdef int nfunctions = np.asarray(functions, dtype=object).shape[0], i, j, k, n, min_ind, min_fill, flag = 0, cur_fill, treewidth
    cdef int[:] temp_vars 
    cdef cnp.ndarray[int, ndim=1] triangulate_vertices
    order = -1*np.ones((2, nvars), dtype=np.int32)
    adj_mat = np.zeros((nvars, nvars), dtype=np.int32)
    for i in range(nfunctions):
        temp_vars = functions[i].getVarIDs()
        for j in range(np.asarray(temp_vars, dtype=np.int32).shape[0]):
            for k in range(j+1, np.asarray(temp_vars, dtype=np.int32).shape[0]):
                adj_mat[temp_vars[j]][temp_vars[k]] = 1
                adj_mat[temp_vars[k]][temp_vars[j]] = 1
    for n in range(nvars):
        min_fill = nvars 
        min_ind = -1
        for i in range(nvars):
            if i not in order[0, :]:
                cur_fill = 0
                triangulate_vertices = np.array([], dtype=np.int32)
                for j in range(nvars):
                    if j not in order[0, :]:
                        if adj_mat[i][j] == 1:    
                            triangulate_vertices = np.hstack([triangulate_vertices, np.array([j], dtype=np.int32)])
                for j in range(triangulate_vertices.shape[0]):
                    for k in range(j+1, triangulate_vertices.shape[0]):
                        if adj_mat[triangulate_vertices[j]][triangulate_vertices[k]] != 1:
                            cur_fill += 1
                if min_fill > cur_fill:
                    min_ind = i 
                    min_fill = cur_fill
        order[0][n] = min_ind
        triangulate_vertices = np.array([], dtype=np.int32)
        for j in range(nvars):
            if adj_mat[min_ind][j] == 1:
                if j not in order[0, :]:
                    triangulate_vertices = np.hstack([triangulate_vertices, np.array([j], dtype=np.int32)])
        #print(min_ind, min_fill, triangulate_vertices)
        for j in range(triangulate_vertices.shape[0]):
            adj_mat[min_ind][j] = 0
            adj_mat[j][min_ind] = 0
        for j in range(triangulate_vertices.shape[0]):
            for k in range(j+1, triangulate_vertices.shape[0]):
                adj_mat[triangulate_vertices[j]][triangulate_vertices[k]] = 1
                adj_mat[triangulate_vertices[k]][triangulate_vertices[j]] = 1
        order[1][n] = triangulate_vertices.shape[0]
    return order

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef list localDataUtil(double[:, :] data, int nvars):
    cdef cnp.ndarray[object, ndim=1] variables, functions, vars 
    cdef int i, j, d = 2
    cdef object var, func 
    cdef cnp.ndarray[double, ndim=1] potential 

    variables = np.array([], dtype=object)
    for i in range(nvars):
        var = Variable(i, d)
        variables = np.hstack([variables, var])
    
    functions = np.array([], dtype=object)
    for i in range(data.shape[0]):
        potential = np.asarray(data[i][2:])
        vars = variables[np.asarray(data[i, :2], dtype=np.int32)]
        func = Function()
        func.setVars(vars)
        func.setPotential(potential)
        functions = np.hstack([functions, func])
    return [variables, functions]


