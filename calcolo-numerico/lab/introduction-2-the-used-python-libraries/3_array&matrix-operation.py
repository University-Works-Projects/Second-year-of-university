## 3 - ARRAY & MATRIX - Operation with matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
from skimage import data, metrics

C, R = 5, 5

A = np.random.randint(5, size=(C,R))
B = np.random.randint(5, size=(C,R))

# np.dop(A, B) or A @ B - Moltiplication: row-colums
print("Date la seguenti matrici CxC (generate casualmente):\n", A, "\n\n", B, '\n')
print("Il loro prod8 riga per colonna è:\n", np.dot(A, B), '\n')
  #print("Il loro prod8 riga per colonna è:\n", A @ B, '\n')

# np.linalg.norm(matrix, n) - Norma n
print("Le varie norme e numeri di condizionamento K(matrix) della matrice:\n", A, "\nSono le seguenti:")
print("Norma 1:\t\t", np.linalg.norm(A, 1), "\n  K(A):\t\t\t", np.linalg.cond(A, 1))                        # Norma 1
print("Norma 2:\t\t", np.linalg.norm(A, 2), "\n  K(A):\t\t\t", np.linalg.cond(A, 2))                        # Norma 2
print("Norma infinito:\t\t", np.linalg.norm(A, 'fro'), "\n  K(A):\t\t\t", np.linalg.cond(A, np.inf))        # Norma infinito
print("Norma di Frobenius:\t", np.linalg.norm(A, np.inf), "\n  K(A):\t\t\t", np.linalg.cond(A, 'fro'))      # Norma di Frobenius

# np.outer(vectorA, vectorB) - See documentation: https://numpy.org/doc/stable/reference/generated/numpy.outer.html

# Relative errror
my_x = 5                                                                                                    # my_x rappresenta il valore (teorico) corretto della varibile x
err = np.linalg.norm(my_x - x, 2) / np.linalg.norm(x, 2)
