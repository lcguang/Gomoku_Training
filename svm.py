import numpy as np
from sklearn import svm
import util

########################################################
# File format:
# row: current values on board \t solution position(num)
#
# Examples:
# row 1: -1 -1 ...  0 1 ... -1 -1 \t 151
# row 2: -1 -1 ...  1 0 ... -1 -1 \t 210
# ...
# row n: -1  0 ... -1 1 ... -1 -1 \t 173
#
# Value types:
# -1: blank
#  1: self
#  0: opponent
#
# Solution position(num):
# x = num / 15
# y = num % 15
########################################################
def readTrainingData(fname):
    for row in open(fname, "r"):
        data = [int(d) for d in row.split()]
        util.training.append(data[:-1])
        util.solution_label.append(data[-1])


########################################################
# Simple linear SVM regression method for training
########################################################
def trainSVM():
    util.classifier = svm.LinearSVR()
    util.classifier.fit(util.training, util.solution_label)


########################################################
# Predict solution from input, return position
########################################################
def predictSolution(input):
    input_array = [int(i) for i in input.split()]
    solution = int(round(util.classifier.predict(input_array)[0]))
    # TODO:
    # Solution position may lies on the cell that has already been filled
    # Need to fix the bug
    return solution / 15, solution % 15


########################################################
# Functionality test for predictions
########################################################
def testPrediction():
    testing = "-1 -1 -1 -1 -1 -1 -1 -1 -1 0 -1 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 -1 0 -1 1 -1 -1 -1 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 -1 -1 1 -1 0 -1 -1 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 0 1 1 1 1 0 -1 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 0 -1 -1 -1 1 -1 0 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 -1 -1 -1 0 1 1 -1 0 -1 -1 -1 " \
              "-1 -1 -1 -1 -1 -1 0 0 1 -1 0 -1 1 -1 -1 " \
              "-1 -1 -1 -1 -1 -1 0 1 0 -1 -1 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 -1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 " \
              "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1"
    print predictSolution(testing)


def main():
    util.init()
    readTrainingData("data/training_example.txt")
    trainSVM()

    testPrediction()


if __name__ == "__main__":
    main()
