import copy
import sys
from pathlib import Path
from collections import Iterable
from collections import OrderedDict




###################################################################################################
## MLC Data Utils #################################################################################
###################################################################################################

N_LINE_IDX = 0
E_LINE_IDX = 1
Q_LINE_IDX = 2
H_LINE_IDX = 3
T_LINE_IDX = 4
FIRST_DATA_LINE_IDX = 5


class MLCData:
    N = None # total number of variables
    E = None # evidence variables via OrderedDict with key:value in the form E:idx_E where idx_E
    Q = None # query variables via OrderedDict
    H = None # hidden variables via OrderedDict
    T = None # number of mlcData points
    raw_data_lines = None # raw strings, each representing the non-empty lines from the data file
    E_e_data = None # list of evidence assignments in the form (E,e) in the same order as E
    e_data = None # list of evidence assignments in the same order as E
    Q_q_data = None # list of query assignments in the form (Q,q) in the same order as Q
    q_data = None # list of query assignments in the same order as Q
    solutionValues = None # list of costs

    def __init__(self, T=None):
        if T==None:
            pass;
        else:
            T=int(T)
            self.T = T
            self.raw_data_lines = [None] * T
            self.E_e_data =  = [None] * T
            self.e_data = [None] * T
            self.Q_q_data =  = [None] * T
            self.q_data = [None] * T
            self.solutionValues = [None] * T

    def print(self, file=sys.stdout):
        print(self.N, file=file)
        print(len(self.E), *(self.E.keys()), file=file)
        print(len(self.Q), *(self.Q.keys()), file=file)
        print(len(self.H), *(self.H.keys()), file=file)
        print(file=file)
        print(self.T, file=file)
        if self.raw_data_lines:
            for raw_data_line in self.raw_data_lines:
                print(raw_data_line, file=file)
        else:
            for E_e_list, Q_q_list, cost in zip(self.E_e_data, self.Q_q_data, self.solutionValues):
                E_e_Q_q_list = E_e_list + Q_q_list
                E_e_Q_q_tokens = [t for X_x in E_e_Q_q_list for t in X_x]
                print(*E_e_Q_q_tokens, cost, file=file)


def parseMLCDataLines(dataLines):
    # N
    N = int(dataLines[N_LINE_IDX])
    
    # E
    E_line_tokens = dataLines[E_LINE_IDX].split()
    E = OrderedDict()
    for i,X in enumerate(E_line_tokens[1:]):
        assert(X not in E) # safety check
        E[int(X)] = i

    # Q
    Q_line_tokens = dataLines[Q_LINE_IDX].split()
    Q = OrderedDict()
    for i,X in enumerate(Q_line_tokens[1:]):
        assert(X not in Q) # safety check
        assert(X not in E) # safety check
        Q[int(X)] = i

    # H
    H_line_tokens = dataLines[H_LINE_IDX].split()
    H = OrderedDict()
    for i,X in enumerate(H_line_tokens[1:]):
        assert(X not in H) # safety check
        assert(X not in Q) # safety check
        assert(X not in E) # safety check
        H[int(X)] = i

    # safety check
    assert(len(E) + len(Q) + len(H) ==  N)
    
    # T
    T = int(dataLines[T_LINE_IDX])

    # safety check
    assert( (len(dataLines) - FIRST_DATA_LINE_IDX) ==  T)

    # mlcData lines
    raw_data_lines = [None] * T
    E_e_data = [None] * T
    Q_q_data = [None] * T
    solutionValues = [None] * T
    for i,rawDataLine in enumerate(dataLines[FIRST_DATA_LINE_IDX:]):
        E_e_list = [None] * len(E)
        Q_q_list = [None] * len(Q)
        n_E_e_added = 0; # for safety check
        n_Q_q_added = 0; # for safety check
        raw_data_lines[i] = rawDataLine
        dataLine_string_tokens = rawDataLine.split()
        dataLine_X_x_tokens = [int(x) for x in dataLine_string_tokens[:-1]]
        solutionValue = float(dataLine_string_tokens[-1])
        solutionValues[i] = solutionValue
        variables = dataLine_X_x_tokens[0::2]
        assignments = dataLine_X_x_tokens[1::2]
        assert(len(variables) == ( len(E) + len(Q) ))
        assert(len(variables) == len(assignments)) # safety check
        for X_x in zip(variables,assignments):
            X = X_x[0]
            if X in E:
                aligned_idx = E[X]
                E_e_list[aligned_idx] = X_x
                n_E_e_added += 1
            elif X in Q:
                aligned_idx = Q[X]
                Q_q_list[aligned_idx] = X_x
                n_Q_q_added += 1
            else:
                assert(False), "\n|\n" + str(X_x) + "\n|\n" + str(X) + "\n|\n" + str(E) + "\n|\n" + str(Q)
        # safety checks
        assert(n_E_e_added == len(E))
        assert(n_Q_q_added == len(Q))
        E_e_data[i] = E_e_list
        Q_q_data[i] = Q_q_list

    # create and return MLCData object
    mlcData = MLCData()
    mlcData.N = N
    mlcData.E = E
    mlcData.Q = Q
    mlcData.H = H
    mlcData.T = T
    mlcData.raw_data_lines = raw_data_lines
    mlcData.E_e_data = E_e_data
    mlcData.e_data = [ [e for (E,e) in E_e_list] for E_e_list in mlcData.E_e_data]
    mlcData.Q_q_data = Q_q_data
    mlcData.q_data = [ [q for (Q,q) in Q_q_list] for Q_q_list in mlcData.Q_q_data]
    mlcData.solutionValues = solutionValues
    return mlcData

def readMLCDataFileLines(dataFile):
    dataFile = Path(dataFile)    
    dataLines = []
    with dataFile.open('r') as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue;
            dataLines.append(line)
    return dataLines
        
def loadMLCData(dataFile):
    dataLines = readMLCDataFileLines(dataFile)
    mlcData = parseMLCDataLines(dataLines)
    return mlcData

def createNewMLCDataObjWithNewSolutions(mlcData, new_Q_q_data, new_values):
    # safety checks
    Q_list = [ Q for (Q,q) in new_Q_q_data[0]]
    assert(Q_list == list(mlcData.Q.keys()))  # TODO: can instead re-order tuples in entries of new_Q_q_data
    assert(len(new_Q_q_data) == len(new_values))

    new_mlcData = copy.deepcopy(mlcData)
    new_mlcData.T = len(new_Q_q_data)
    new_mlcData.Q_q_data = new_Q_q_data
    new_mlcData.q_data = [ [q for (Q,q) in Q_q_list] for Q_q_list in new_mlcData.Q_q_data]
    new_mlcData.E_e_data = new_mlcData.E_e_data[:new_mlcData.T]
    new_mlcData.e_data = new_mlcData.e_data[:new_mlcData.T]
    new_mlcData.solutionValues = new_values
    new_mlcData.raw_data_lines = None

    
    # safety checks
    assert(new_mlcData.Q_q_data is not mlcData.Q_q_data)
    assert(new_mlcData.Q_q_data[0] is not mlcData.Q_q_data[0])
    assert(new_mlcData.Q_q_data[0][0] is not mlcData.Q_q_data[0][0])

    return new_mlcData



###################################################################################################
## MLC Results Utils ##############################################################################
###################################################################################################

from pathlib import Path
from collections import OrderedDict

class MLCSolutions:
    Q = None # query variables via OrderedDict
    T = None # number of mlcSolutions points
    raw_solution_lines = None # raw strings, each representing the non-empty lines from the solution file
    Q_q_solutions = None # list of query assignments in the form (Q,q) in the same order as Q
    q_solutions = None # list of query assignments in the same order as Q
    solutionValues = None # externally can be added as a list of log10 values

    def __init__(self, T=None):
        if T==None:
            pass;
        else:
            T=int(T)
            self.T = T
            self.raw_solution_lines = [None] * T
            self.Q_q_solutions =  = [None] * T
            self.q_solutions = [None] * T
            self.solutionValues = [None] * T

    def set_solutionValues(self, solutionValues):
        assert(len(solutionValues) == self.T) # safety check
        self.solutionValues = solutionValues



def parseMLCSolutionLines(solutionLines): 
    # T
    T = len(solutionLines)

    # Q
    sampleSolutionLine_vars = None
    numQueryAssignments = None
    if isinstance(solutionLines[0], str): # assume solution lines as strings in UAI result format (will perform safety checks)
        sampleSolutionLine_tokens = [int(x) for x in solutionLines[0].split()]
        numQueryAssignments = sampleSolutionLine_tokens[0]
        sampleSolutionLine_vars = sampleSolutionLine_tokens[1::2]
    else: # assume solution lines are lists/ordered-iterables
        sampleSolutionLine_tokens = [int(x) for x in solutionLines[0]]
        if len(sampleSolutionLine_tokens) % 2 == 0: # needs a first element to signify number of query assignments as per UAI solution format
            numQueryAssignments = len(sampleSolutionLine_tokens)/2
            sampleSolutionLine_vars = sampleSolutionLine_tokens[::2]
        else:
            numQueryAssignments = sampleSolutionLine_tokens[0]
            sampleSolutionLine_vars = sampleSolutionLine_tokens[1::2]
    assert(numQueryAssignments == len(Q)) # safety check that num assignments == len(Q)
    assert(numQueryAssignments == len(sampleSolutionLine_vars))
    Q = OrderedDict()
    for i,X in enumerate(sampleSolutionLine_vars):
        assert(X not in Q) # safety check
        Q[int(X)] = i

    # mlcSolution lines
    Q_q_solutions = [None] * T
    for i,rawSolutionLine in enumerate(solutionLines):
        if isinstance(rawSolutionLine, str): # assume solution lines as strings in UAI result format (will perform safety checks)
            solutionLine_tokens = [int(x) for x in rawSolutionLine.split()]
        else: # assume solution lines are lists/ordered-iterables
            solutionLine_tokens = [int(x) for x in rawSolutionLine]
            if len(solutionLine_tokens) % 2 == 0: # needs a first element to signify number of query assignments as per UAI solution format
                solutionLine_tokens = [len(solutionLine_tokens)/2] + solutionLine_tokens
        assert(solutionLine_tokens[0] == len(Q))
        variables = solutionLine_tokens[1::2]
        assignments = solutionLine_tokens[2::2]
        assert(len(variables) == len(Q)) # safety check
        assert(len(variables) == len(assignments)) # safety check
        Q_q_list = [None] * len(Q)
        n_Q_q_added = 0; # for safety check
        for X_x in zip(variables,assignments):
            X = X_x[0]
            assert(X in Q)
            aligned_idx = Q[X]
            Q_q_list[aligned_idx] = X_x
            n_Q_q_added += 1
        # safety checks
        assert(n_Q_q_added == len(Q))
        Q_q_solutions[i] = Q_q_list

    # create and return MLCSolutions object
    mlcSolutions = MLCSolutions()
    mlcSolutions.Q = Q
    mlcSolutions.T = T
    if isinstance(solutionLines[0], str):
        mlcSolutions.raw_solution_lines = solutionLines
    else:
        if needsNumQ:
            mlcSolutions.raw_solution_lines = " ".join([str(x) for x in solutionLines]) + " ".join([str(x) for x in solutionLines])
        else:
            mlcSolutions.raw_solution_lines = " ".join([str(x) for x in solutionLines])
    mlcSolutions.Q_q_solutions = Q_q_solutions
    mlcSolutions.q_solutions = [ [q for (Q,q) in Q_q_list] for Q_q_list in mlcSolutions.Q_q_solutions]
    mlcSolutions.solutionValues = [None]*T
    return mlcSolutions

def readMLCSolutionFileLines(solutionsFile):
    solutionsFile = Path(solutionsFile)    
    solutionLines = []
    with solutionsFile.open('r') as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue;
            solutionLines.append(line)
    return solutionLines
        
def loadMLCSolutions(solutionsFile):
    solutionLines = readMLCSolutionFileLines(solutionsFile)
    mlcSolutions = parseMLCSolutionLines(solutionLines)
    return mlcSolutions




###################################################################################################
## MLC Solutions Evaluator ########################################################################
###################################################################################################

def evaluateMLCSolution(model,E_e,Q_q):
    # model is a pyGMs graphical model object
    model = model.copy()
    new_E_e = dict(E_e)
    new_E_e.update(Q_q)
    model.condition(new_E_e)
    sumElim = lambda F,Xlist: F.sum(Xlist)
    elimOrder, _ = pyGMs.eliminationOrder(model)
    model.eliminate(elimOrder,sumElim)
    log10Solution = (model.joint().log10())[0]
    return log10Solution;




###################################################################################################
## MLC Learning Util ##############################################################################
###################################################################################################

class MLCUtil:
    model = None # pyGMs graphical model
    data = None # MLCData object, containes information loaded from the MLC data file (includes variable partitions)
    solutions = None # MLCSolutions object, contains information about evaluated solutions

    def __init__(self, uaiFile, dataFile):
        factors = pyGMs.readUai(uaiFile)
        self.model = pyGMs.GraphModel(factors)
        self.data = loadMLCData(dataFile)
        self.mlcSolutions = []
    
    def evaluateFullListOfSolutions(self, solutionsSource, append=True):
        solutions = None
        try: # if filepath is passed in
            p = Path(solutionsSource)
            assert(p.is_file())
            solutions = loadMLCSolutions(p)
        except: # if solution lines are passed in
            solutionLines = None
            if isinstance(solutionsSource, str): # single solution line in raw string form
                solutionLines = [singleSolution,]
            else:
                assert(isinstance(solutionsSource,Iterable))
                try: # check if it is a single solution already in list form
                    singleSolution = [int(x) for x in solutionsSource]
                    solutionLines = [singleSolution,]
                except: # at this point, assume it is a list of solution lines
                    solutionLines = solutionsSource
            solutions = parseMLCSolutionLines(solutionLines)
        # TODO: other safety checks
        solutionValues = []
        for i,Q_q in enumerate(solutions.Q_q_solutions):
            E_e = self.data.E_e_data
            log10SolutionValue = evaluateMLCSolution(model=self.model, E_e=E_e, Q_q=Q_q)
        solutions.set_solutionValues()
        if append == True:
            self.mlcSolutions.append(solutions)
        return solutions




###################################################################################################
## TODO's #########################################################################################
###################################################################################################
#
# print() function for MLCSolutions
# emplace function for MLCSolutions and solution values (based on a list of indices)


