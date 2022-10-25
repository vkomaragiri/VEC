import numpy as np
import os.path
import sys
from pathlib import Path
from time import process_time
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from numba import jit

class Data:
    dataset_x = None
    dataset_y = None
    obsvars = None
    queryvars = None

    def __init__(self,obsvars,queryvars,dataset_x,dataset_y) -> None:
        self.obsvars = obsvars
        self.queryvars = queryvars
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

def read_evidence_file(fname):
    if os.path.isfile(fname):
        return np.loadtxt(fname, delimiter=' ',dtype=int)


def parse_data(data):
    nvars = int(data[0])
    nobs = int(data[1])
    obsvars = np.array(data[2:2 + nobs], dtype=int)
    obsvars_idx = {key:i for (i, key) in enumerate(obsvars)}
    nquery = int(data[2 + nobs])
    queryvars = np.array(data[3 + nobs:3 + nobs + nquery], dtype=int)
    queryvars_idx = {key:i for (i, key) in enumerate(queryvars)}
    nhidden = int(data[3 + nobs + nquery])
    hiddenvars = np.array(data[4 + nobs + nquery:4 + nobs + nquery + nhidden], dtype=int)
    hiddenvars_idx = {key:i for (i, key) in enumerate(hiddenvars)}
    nsamples = int(data[4 + nobs + nquery + nhidden])
    dataset_x = np.zeros([nsamples, nobs], dtype=int)
    dataset_y = np.zeros([nsamples, nquery], dtype=int)
    c = 5 + nobs + nquery + nhidden
    for i in range(nsamples):
        for j in range(nobs + nquery):
            var = int(data[c])
            c = c + 1
            val = int(data[c])
            c = c + 1
            if var in obsvars_idx:
                dataset_x[i, obsvars_idx[var]] = val
            if var in queryvars_idx:
                dataset_y[i, queryvars_idx[var]] = val
        c = c + 1
    processedData = Data(obsvars=obsvars, queryvars=queryvars, dataset_x=dataset_x, dataset_y=dataset_y)
    return processedData

def read_data(fname):
    if os.path.isfile(fname):
        file = open(fname)
        data=file.read().split()
        return parse_data(data)

def printPredictions(jobName, y_vars, y_pred, problemDir):
    problemName = problemDir.name
    predictionFile = problemDir / (problemName + "_" + jobName + ".out")
    n_y = len(y_vars)
    y_pred_rows, y_pred_cols = y_pred.shape
    assert(n_y == y_pred_cols)
    with predictionFile.open('w') as fout:
        for r in range(y_pred_rows):
            print(n_y, end=" ", file=fout)
            for (Y, y) in zip(y_vars, y_pred[r]):
                print(Y, y, end=" ", file=fout)
            print(file=fout)



# Train using Logistic Regression
def train_lr(trainingData):
    clf = MultiOutputClassifier(LogisticRegression()).fit(trainingData.dataset_x, trainingData.dataset_y)
    return clf

# Train using Random Forests
def train_rf(trainingData,d=2):
    clf = MultiOutputClassifier(RandomForestClassifier(max_depth=d, random_state=0)).fit(trainingData.dataset_x, trainingData.dataset_y)
    return clf

if __name__ == '__main__':
    problemDir = Path(sys.argv[1])
    problemName = problemDir.name
    logFile = problemDir / (problemName + "_simpleMLCSolver.log")

    with logFile.open('w') as logfout:

        trainingDataFile = problemDir/(problemName + ".data")
        testDataFile = problemDir/(problemName + ".test")
        print("Reading in Training Data...", file=logfout, flush=True)
        start_t = process_time()
        trainingData = read_data(trainingDataFile)
        end_t = process_time()
        elapsed_t = round(end_t - start_t)
        print("Training Data loaded! (" + str(elapsed_t) + " sec)", file=logfout)
        print("Reading in Test Data...", file=logfout, flush=True)
        start_t = process_time()
        testData = read_data(testDataFile)
        end_t = process_time()
        elapsed_t = round(end_t - start_t)
        print("Test Data loaded! (" + str(elapsed_t) + " sec)", file=logfout)
        print(file=logfout)

        print(" Random Forests (d=2) ", file=logfout)
        print("----------------------", file=logfout)
        print("Beginning training...", file=logfout, flush=True)
        start_t = process_time()
        clf_rf_d2=train_rf(trainingData,d=2)
        end_t = process_time()
        elapsed_t = round(end_t - start_t)
        print("Training complete! (" + str(elapsed_t) + " sec)", file=logfout)
        print("Making predictions...", file=logfout, flush=True)
        start_t = process_time()
        rf_d2_pred_y=clf_rf_d2.predict(testData.dataset_x)
        end_t = process_time()
        elapsed_t = round(end_t - start_t)
        print("Predictions complete! (" + str(elapsed_t) + " sec)", file=logfout)
        print('Hamming distance is '+str(np.sum(np.abs(testData.dataset_y-rf_d2_pred_y))), file=logfout, flush=True)
        print("Printing predictions...", file=logfout)
        printPredictions("RF_d-2", testData.queryvars, rf_d2_pred_y, problemDir)
        print("Predictions printed!", file=logfout)
        print(file=logfout)

        print(" Random Forests (d=4) ", file=logfout)
        print("----------------------", file=logfout)
        print("Beginning training...", file=logfout, flush=True)
        start_t = process_time()
        clf_rf_d4=train_rf(trainingData,d=4)
        end_t = process_time()
        elapsed_t = round(end_t - start_t)
        print("Training complete! (" + str(elapsed_t) + " sec)", file=logfout)
        print("Making predictions...", file=logfout, flush=True)
        start_t = process_time()
        rf_d4_pred_y=clf_rf_d4.predict(testData.dataset_x)
        end_t = process_time()
        elapsed_t = round(end_t - start_t)
        print("Predictions complete! (" + str(elapsed_t) + " sec)", file=logfout)
        print('Hamming distance is '+str(np.sum(np.abs(testData.dataset_y-rf_d4_pred_y))), file=logfout, flush=True)
        print("Printing predictions...", file=logfout)
        printPredictions("RF_d-4", testData.queryvars, rf_d4_pred_y, problemDir)
        print("Predictions printed!", file=logfout)
        print(file=logfout)

        print(" Random Forests (d=8) ", file=logfout)
        print("----------------------", file=logfout)
        print("Beginning training...", file=logfout, flush=True)
        start_t = process_time()
        clf_rf_d8=train_rf(trainingData,d=8)
        end_t = process_time()
        elapsed_t = round(end_t - start_t)
        print("Training complete! (" + str(elapsed_t) + " sec)", file=logfout)
        print("Making predictions...", file=logfout, flush=True)
        start_t = process_time()
        rf_d8_pred_y=clf_rf_d8.predict(testData.dataset_x)
        end_t = process_time()
        elapsed_t = round(end_t - start_t)
        print("Predictions complete! (" + str(elapsed_t) + " sec)", file=logfout)
        print('Hamming distance is '+str(np.sum(np.abs(testData.dataset_y-rf_d8_pred_y))), file=logfout, flush=True)
        print("Printing predictions...", file=logfout)
        printPredictions("RF_d-8", testData.queryvars, rf_d8_pred_y, problemDir)
        print("Predictions printed!", file=logfout)
        print(file=logfout)

        print(" Logistic Regression ", file=logfout)
        print("---------------------", file=logfout)
        print("Beginning training...", file=logfout, flush=True)
        start_t = process_time()
        clf_lr=train_lr(trainingData)
        end_t = process_time()
        elapsed_t = round(end_t - start_t)
        print("Training complete! (" + str(elapsed_t) + " sec)", file=logfout)
        print("Making predictions...", file=logfout, flush=True)
        start_t = process_time()
        lr_pred_y=clf_lr.predict(testData.dataset_x)
        end_t = process_time()
        elapsed_t = round(end_t - start_t)
        print("Predictions complete! (" + str(elapsed_t) + " sec)", file=logfout)
        print('Hamming distance is '+str(np.sum(np.abs(testData.dataset_y-lr_pred_y))), file=logfout, flush=True)
        print("Printing predictions...", file=logfout)
        printPredictions("LR", testData.queryvars, lr_pred_y, problemDir)
        print("Predictions printed!", file=logfout)
        print(file=logfout)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
