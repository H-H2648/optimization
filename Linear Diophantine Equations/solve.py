#solves linear system of equatin of the form:
#a11 x1 + ... + a1n xn = b1
#...
#am1 x1 + ... + amn xn = bn

#for aij, bi integers, x is a vector of integers

#more specifically, it solves whether a solution exists.
#If there is a solution, it outputs one of such solution (for proof that this program is indeed correct)
#If there is no solution, it finds y^T such that y^TA is an integer but y^Tb is rational (clearly, there cannot be a solution) when that is the case

import numpy as np
import math
#for now the inverse function is used from scipy
#maybe implement my own inverse function(?)
from scipy import linalg


#BASED ON FINDING THE HERMITE NORMAL FORM
#The proof is based on Integer Farkas Lemma:

#Ax = b, has an integer solution iff y^Tb is an integer for each y such that y^TA is an integer

#note no return on the 3 helper elementary operations because it directly mutates the matrix input

#pretty self explanatory
def exchangeTwoColumn(matrix, ii, jj):
    tempCopy = np.copy(matrix[:, ii])
    matrix[:, ii] = matrix[:, jj]
    matrix[:, jj] = tempCopy

#multiplies an entire column by -1
def negativeCol(matrix, ii):
    matrix[:, ii] = -matrix[:, ii]

#we add col[jj]*const to col[ii]
#note const is assumed to be an integer
def addIntMult(matrix, ii, jj, const):
    matrix[:,ii] += matrix[:,jj]*const

X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

def revert(vector, action):
    if action[0] == 'swap':
        ii, jj = action[1], action[2]
        temp = vector[ii]
        vector[ii] = vector[jj]
        vector[jj] = temp
    elif action[0] == 'negative':
        indexNegated = action[1]
        vector[indexNegated] = -vector[indexNegated]
    elif action[0] == 'add':
        indexAdded = action[1]
        indexAdding = action[2]
        const = action[3]
        vector[indexAdding] += const*vector[indexAdded]
    else:
        raise ValueError(
            "SOMETHING HAS GONE TERRIBLY WRONG"
        )

#assumes the number of rows is more than or equal to the number of columns (otherwise, there is only one solution and we can easily check it by applying simple linear system of equations)
#note we assume a general real solution actually exists
def HermiteSolve(coefficients, values, receipt=True):
    coefficientsCopy = coefficients.copy()
    actions = []
    for index in range(len(coefficientsCopy)):
        for ii in range(index, len(coefficientsCopy[0])):
            if coefficientsCopy[index][ii] < 0:
                negativeCol(coefficientsCopy, ii)
                actions.append(('negative', ii))
        valid_idx = np.where(coefficientsCopy[index][index:] > 0)[0] + index
        tempPos = coefficientsCopy[index][valid_idx].argmin()
        minPos = valid_idx[tempPos]
        while len(valid_idx) > 1:
            tempPos = coefficientsCopy[index][valid_idx].argmin()
            minPos = valid_idx[tempPos]
            minVal = coefficientsCopy[index][minPos]
            for ii in range(index, len(coefficientsCopy[0])):
                if ii != minPos and coefficientsCopy[index][ii] > 0:
                    val = coefficientsCopy[index][ii]
                    const = -int(val/minVal)
                    addIntMult(coefficientsCopy, ii, minPos, const)
                    actions.append(("add", ii, minPos, const))
            valid_idx = np.where(coefficientsCopy[index][index:] > 0)[0] + index
        exchangeTwoColumn(coefficientsCopy, minPos, index)
        actions.append(("swap", minPos, index))
    #we want the diagonal to be the unique maximum
    #while each row has nonnegative values
    for index in range(len(coefficientsCopy)):
        for ii in range(index):
            diagVal = coefficientsCopy[index][index]
            val = coefficientsCopy[index][ii]
            if val < 0:
                const = -math.floor(val/diagVal)
                addIntMult(coefficientsCopy, ii, index, const)
                actions.append(("add", ii, index, const))
            if val > diagVal:
                const = -math.floor(val/diagVal)
                addIntMult(coefficientsCopy, ii, index, const)
                actions.append(("add", ii, index, const))
    hermiteB = coefficientsCopy[:len(coefficientsCopy),:len(coefficientsCopy)]
    hermiteBInv = linalg.inv(hermiteB)
    attemptSol = np.dot(hermiteBInv, values)
    #if within 10^{-13} of an int, we consider it to be an int
    nonIntIndex = (np.where(attemptSol.astype(int) - attemptSol >= 10**(-13)))[0]
    if nonIntIndex.size == 0:
        if receipt:
            tempAnswer = np.zeros(len(coefficientsCopy[0]))
            tempAnswer[:len(coefficientsCopy)] = attemptSol
            for ii in range(len(actions)-1, -1, -1):
                action = actions[ii]
                revert(tempAnswer, action)
            answer = tempAnswer
            return (True, answer)
        else:
            return True
    else:
        if receipt:
            return (False, hermiteBInv[nonIntIndex[0]])
        return False


#if within 10^{-13}, good enough
def verifyTrue(coefficients, value, apparentSol):
    return np.allclose(value, np.dot(coefficients, apparentSol))

def verifyFalse(coefficients, value, apparentProof):
    yTA = np.dot(apparentProof, coefficients)
    yTb = np.dot(apparentProof, value)
    return np.all(np.mod(yTA, 1) < 10**(-13)) and (np.any(np.mod(yTb, 1) >= 10**(-13)))


A = np.array(
    [
        [1, 1, 1, 0],
        [1, 1, 2, 0],
        [7, 2, 3, 0],
    ]
)
b = np.array(
    [
        3,
        4,
        12,
    ]
)

print(A)
print(b)


answer, receipt = HermiteSolve(A, b, receipt=True)
print(answer)
print(receipt)
if answer:
    print(verifyTrue(A, b, receipt))
else:
    print(verifyFalse(A, b, receipt))

