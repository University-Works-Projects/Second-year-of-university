## 4 - IMMAGINI 
 
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
from skimage import data, metrics

# Fattorizzazione LR
x = LUdec.lu_solve(LUdec.lu_factor(A), b)

# Fattorizzazione LR con pivoting (Metodo di Gauss)
lu, piv = LUdec.lu_factor(A)                                                                        # Fattorizza A in LR e pivot
my_x = LUdec.lu_solve((lu, piv), b)                                                                 # Risolve er sistema

# Fattorizzazione LR con pivoting (Alternativa)
P, L, U = LUdec.lu(A)                                                                               # !!! A = P*L*U (non P*A = L*U) !!!
invP = np.linalg.inv(P)
y = scipy.linalg.solve_triangular(L, np.matmul(invP, b), lower = True, unit_diagonal = True)
my_x = scipy.linalg.solve_triangular(U, y, lower = False)

# Fattorizzazione di Cholesky
L = np.linalg.cholesky(A)
y = scipy.linalg.basic.solve_triangular(L, b, lower = True)
x = scipy.linalg.basic.solve_triangular(L.T, y, lower = False)
