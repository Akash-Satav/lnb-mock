from numpy import array
from numpy.linalg import eig
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
values, vectors = eig(A)
print(values)
print(vectors)