import numpy as np
import math
from math import e

def find_vertex_complexity_upper_bound(A, b):
    return 4*len(A[0])*max(A.max(), b.max())


def KhachiyanSolve(coefficients, values, receipt =True):
    n = len(coefficients[0])
    v = find_vertex_complexity_upper_bound(np.abs(coefficients), np.abs(values))
    radius = math.pow(2, v)
    z = np.zeros(n)
    D = np.identity(n)*radius
    lower_bound_ellipse_volume = math.pow(2, (-2*n*v))
    upper_bound_ellipse_volume = math.pow(2*radius, n)
    index = 0
    while upper_bound_ellipse_volume >= lower_bound_ellipse_volume:
        index +=1
        check_inequality = np.less_equal(np.dot(coefficients, z), values)
        if np.all(check_inequality):
            if receipt:
                return ((True, z))
            else:
                return True
        else:
            row_to_fix = coefficients[np.argmax(np.logical_not(check_inequality))]
            z -= 1/(n+1) * (np.dot(D, row_to_fix))/(math.sqrt(np.dot(np.dot(row_to_fix, D), row_to_fix)))
            D = n**2/(n**2 - 1) * (D - 2/(n + 1)* np.outer(np.dot(D, row_to_fix), np.dot(row_to_fix, D))/np.dot(np.dot(row_to_fix, D), row_to_fix))
            upper_bound_ellipse_volume = e**(-index/(2*n + 2))*(2*radius)**n
    if receipt:
        return ((False, (D, z, upper_bound_ellipse_volume, lower_bound_ellipse_volume , index)))
    else:
        return False

def verifyTrue(coefficients, value, apparentSol):
    return np.all(np.less_equal(np.dot(coefficients, apparentSol), value))

def verifyFalse(receipt):
    D = receipt[0]
    z = receipt[1]
    upper_bound_ellipse_volume = receipt[2]
    lower_bound_ellipse_volume = receipt[3]
    index = receipt[4]
    return verifyFalseHelper(D, z, upper_bound_ellipse_volume, lower_bound_ellipse_volume, index)

def verifyFalseHelper(D, z, upper_bound_ellipse_volume, lower_bound_ellipse_volume, index):
    print("Consider the Ellipse centered at z and the matrix D:")
    print("Where z = {0}".format(z))
    print("and D = {0}".format(D))
    print("The volume of the ellipse has upper bound: {0} since we were at index {1}".format(upper_bound_ellipse_volume, index))
    print("This must contain our polyhedron")
    print("The volume of our polyhedron has lower bound: {0}".format(lower_bound_ellipse_volume))
    print("We have a contradiction")
    return (upper_bound_ellipse_volume < lower_bound_ellipse_volume)

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
        [1, 2],
        [-1, -2],
    ]
)

b= np.array(
    [
        1,
        -2,
    ]
)
'''
answer, receipt = KhachiyanSolve(X, b)
print(answer)
print(receipt)
print(verifyTrue(X, b, receipt))