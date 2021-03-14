import numpy as np


#Assumes M *N matrix, with M rows and N columns
#the vector is of size M

# helper
# check if a given column is already zeros
#O(M)
def check0Column(matrix, index):
    #we add + index at the end to acount for us "shifting" to the right by index (we don't care about the zero values before the index)
    return np.nonzero(matrix[index:,index])[0] + index


# helper
# find any possible zero rows after the indexed row.
# O(MN)
def find0Row(matrix, index):
    found = []
    for ii in range(index, len(matrix)):
        if np.all(matrix[index] == 0):
            found.append(index)
    return found


# For determining the vector to "standardize" the column (matrix[index][index]) is 1 or 0 (0 only when the entire column is 0)
# This is so upper triangulizing the column is much easier
def standardizeColumn(matrix, index):
    print("standardizing... {0}".format(index))
    transform = np.identity(len(matrix))
    nonZeroPositions = check0Column(matrix, index)
    if (nonZeroPositions).size == 0:
        return transform
    else:
        if nonZeroPositions[0] == index:
            transform[index] = transform[index]/matrix[index][index]
        else:
            transform[index] += transform[nonZeroPositions[0]]/matrix[nonZeroPositions[0]][index]
    return transform


# For determining the vector to "upper-triangulizing" the column with Gaussian elimination algorithm
# assumes column for position 0, 1, ..., index - 1 is already upper triangulized
# assumes the column in question is already standardized from above
def upperTriangulizeColumn(matrix, index):
    transform = np.identity(len(matrix))
    if matrix[index][index] == 0:
        return transform
    else:
        for ii in range(index+1, len(matrix)):
            transform[ii] = transform[ii] - matrix[ii][index]*transform[index]
        return transform

def solveUpperTriangular(matrix, values):
    solution = np.zeros(len(matrix[0]))
    for ii in range(len(matrix)-1, -1, -1):
        solution[ii] = values[ii]
        for jj in range(ii + 1, len(matrix)):
            solution[ii] -= matrix[ii][jj]*solution[jj]
    return solution




# main function
def GaussianEliminationSolve(coefficients, values, receipt=True):
    if len(values) != len(coefficients):
        raise ValueError(
            "The number of rows of the matrix and the vector must match!\nNumber of rows of matrix: {0}\nNumber of values of the vector: {1}\n".format(
                len(values), len(coefficients))
        )
    #YT refers to coefficients of linear combination for the rows of the matrix and the vectors
    YT = np.identity(len(coefficients))
    for index in range(len(coefficients)):
        zeroRows = find0Row(coefficients, index)
        for row in zeroRows:
            if values[row] != 0:
                if receipt:
                    return (False, YT[index])
                else:
                    return False
        transformToStandardizeColumn = standardizeColumn(coefficients, index)
        YT = np.dot(transformToStandardizeColumn, YT)
        coefficients = np.dot(transformToStandardizeColumn, coefficients)
        print("index: {0}".format(index))
        print(coefficients)
        values = np.dot(transformToStandardizeColumn, values)
        transformToUpperTriangulizeColumn = upperTriangulizeColumn(coefficients, index)
        YT = np.dot(transformToUpperTriangulizeColumn, YT)
        coefficients = np.dot(transformToUpperTriangulizeColumn, coefficients)
        values = np.dot(transformToUpperTriangulizeColumn, values)
    if receipt:
        solution = solveUpperTriangular(coefficients, values)
        return (True, solution)
    else:
        return True

#if within 10^{-13}, good enough
def verifyTrue(coefficients, value, apparentSol):
    return np.allclose(value, np.dot(coefficients, apparentSol))

def verifyFalse(coefficients, value, apparentProof):
    return (np.all(np.dot(apparentProof, coefficients) == 0)) and (np.dot(apparentProof, value) != 0)

x = np.array([
    [1, -1, 1, 0],
    [2, -1, 2, -2],
    [-1, 1/2, -1, 1],
])

y = np.array([
    1,
    3,
    2
])

answer, receipt = GaussianEliminationSolve(x, y, receipt=True)
print(answer)
print(receipt)
if answer:
    print(verifyTrue(x, y, receipt))
else:
    print(verifyFalse(x, y, receipt))









