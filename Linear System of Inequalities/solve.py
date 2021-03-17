#solves linear system of equatin of the form:
#a11 x1 + ... + a1n xn <= b1
#...
#am1 x1 + ... + amn xn <= bn

#more specifically, it solves whether a solution exists.
#If there is a solution, it outputs one of such solution (for proof that this program is indeed correct)
#If there is no solution, it finds y such that y >= 0, y^TA = 0 but y^Tb < 0 (clearly, there cannot be a solution since it means 0 <= t, for t < 0) when that is the case


#BASED ON FOURIER MOTZSKIN ELIMINATION
#The proof of solution from Farkas's lemma:

#Ax <= b has a solution iff for all y >= 0, y^TA = 0^T, y^Tb < 0 has no solution

import numpy as np


#scale each row such that each row has either 1, 0, -1  as the first coefficients (we can only scale by positive factors)
def scaleAllRows(coefficients, value, index):
    scalarTransformation = np.identity(len(coefficients))
    for ii in range(len(coefficients)):
        scalar = abs(coefficients[ii][index])
        if scalar == 0:
            continue
        else:
            scalarTransformation[ii][ii] = 1/scalar
    coefficients = np.dot(scalarTransformation, coefficients)
    value = np.dot(scalarTransformation, value)
    return (coefficients, value, scalarTransformation)


def find0Row(matrix):
    found = []
    for ii in range(len(matrix)):
        if np.all(matrix[ii] == 0):
            found.append(ii)
    return found




#assumes each row is scaled
def FourierMotzkinEliminate(coefficients, values, index, length, history):
    positiveIdx = np.where(coefficients[:,index] == 1)[0]
    negativeIdx = np.where(coefficients[:, index] == -1)[0]
    zeroIdx = np.where(coefficients[:, index] == 0)[0]
    newCoefficients = []
    newValues = []
    transform = []
    for pIndex in positiveIdx:
        for nIndex in negativeIdx:
            newCoefficients.append((coefficients[pIndex]+ coefficients[nIndex]))
            newValues.append(values[pIndex] + values[nIndex])
            newyTRow = np.zeros(length)
            newyTRow[pIndex] = 1
            newyTRow[nIndex] = 1
            transform.append(newyTRow)
    for zIndex in zeroIdx:
        newCoefficients.append(coefficients[zIndex])
        newValues.append(values[zIndex])
        newyTRow = np.zeros(length)
        newyTRow[zIndex] = 1
        transform.append(newyTRow)
    positiveColumns = values[positiveIdx].copy()
    negativeColumns = values[negativeIdx].copy()
    if positiveColumns.ndim == 1:
        positiveColumns = np.array([positiveColumns])
    if negativeColumns.ndim == 1:
        negativeColumns = np.array([negativeColumns])
    positiveColumns= positiveColumns.T
    negativeColumns = negativeColumns.T
    if len(positiveIdx) > 0:
        lessThanOrEqualToMatrix = np.hstack((positiveColumns, (-1)*coefficients[positiveIdx][:,index+1:].copy()))
    if len(negativeIdx) > 0:
        greaterThanOrEqualToMatrix = np.hstack(((-1)*negativeColumns, coefficients[negativeIdx][:,index+1:].copy()))
    if len(positiveIdx) == 0:
        lessThanOrEqualToMatrix = np.array([])
    if len(negativeIdx) == 0:
        greaterThanOrEqualToMatrix = np.array([])
    history.append((lessThanOrEqualToMatrix, greaterThanOrEqualToMatrix))
    newCoefficients = np.array(newCoefficients)
    newValues = np.array(newValues)
    transform = np.array(transform)
    return (newCoefficients, newValues, transform, history)

def findSolution(history, index, length):
    solution = np.zeros(length - index - 1)
    for ii in range(index, -1, -1):
        lessThanOrEqualToMatrix = history[ii][0]
        greaterThanOrEqualToMatrix = history[ii][1]
        minVal = None
        maxVal = None
        buildingUp = np.append(np.array([1]), solution.copy())
        if lessThanOrEqualToMatrix.size != 0:
            minVal = np.min(np.dot(lessThanOrEqualToMatrix, buildingUp))
            #x needs to be smaller than the smallest possible minVal
        if greaterThanOrEqualToMatrix.size != 0:
            maxVal = np.max(np.dot(greaterThanOrEqualToMatrix, buildingUp))
            #x needs to be greater than the greatest possible maxVal
        if minVal == None:
            nextVal = maxVal
        elif maxVal == None:
            nextVal = minVal
        else:
            nextVal = (maxVal + minVal)/2
        solution = np.append(np.array([nextVal]), solution.copy())
    return solution



def FourierMotzkinSolve(coefficients, values, receipt=True):
    A = coefficients.copy()
    b = values.copy()
    yT = np.identity(len(A))
    numVariables = len(coefficients[0])
    #history consists of [
    # matrix for [1, x_js] (j > i) such that its linear combination is less than or equal to xi),
    # matrix for [1, x_js] (j > i) such that its linear combination is greater than or equal to xi
    # ]
    history = []
    for index in range(len(coefficients)):
        zeroRows = find0Row(A)
        for row in zeroRows:
            if b[row] < 0:
                if receipt:
                    return (False, yT[row])
                else:
                    return False
        A, b, scalarTransformation = scaleAllRows(A, b, index)
        yT = np.dot(scalarTransformation, yT)
        #number of system is always changing so we have to reassign them each time
        numSystem = len(A)
        A, b, transform, history = FourierMotzkinEliminate(A, b, index, numSystem, history)
        if A.size == 0:
            if receipt:
                return ((True, findSolution(history, index, numVariables)))
            else:
                return True
        yT = np.dot(transform, yT)


def verifyTrue(coefficients, value, apparentSol):
    return np.all(np.less_equal(np.dot(coefficients, apparentSol), value))

def verifyFalse(coefficients, value, apparentProof):
    return (np.all(np.dot(apparentProof, coefficients) == 0)) and np.all((np.less(np.dot(apparentProof, value), 0)))

X = np.array(
    [
        [1, 1, 1],
        [-1, -1, 2],
        [2, 3, 5],
        [3, 7, 9],
    ]
).astype(float)

b = np.array(
    [
        -2,
        -4,
        1,
        -2
    ]
).astype(float)

'''
X = np.array(
    [
        [1],
        [-1],
    ]
)

b= np.array(
    [
        1,
        -2,
    ]
)
'''

answer, receipt = FourierMotzkinSolve(X, b, True)
print(answer)
print(receipt)
if answer:
    print(verifyTrue(X, b, receipt))
else:
    print(verifyFalse(X, b, receipt))



