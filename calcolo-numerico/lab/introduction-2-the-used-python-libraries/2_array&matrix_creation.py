## 2 - ARRAY & MATRIX - Creation of array and matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
from skimage import data, metrics

C, R = 5, 3



# np.array([]) - 1 Dimension
Arr = np.array([3,3,3,3])                                               # {Array | 1 row matrix} initialization
print ("Arr: ", Arr, '\n')

# np.arange()
Arr = np.arange(4)                                                      # Arr = [0,1,2,3]
print ("Arr created with arange(n): form 0 to n-1:\n", Arr, '\n')

# np.linespace() - Per capire linespace guardare l'output che si capisce al volo
Arr = np.linspace(0, 10, num=5)                                         # Arr = [0., 2.5, 5., 7.5, 10.]
print ("Arr created with linspace(0, 10, num=5):\n", Arr, '\n')

# np.array([],[],[],...) - Multiple Dimension
A = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]])                           # 3x4 matrix initialization
print ("A:\n", A, '\n')
#print(A[n])                                                            # To print the n-row of: A

# np.zeros()
onlyZeroMatrix = np.zeros((C, R))                                       # Create a CxR matrix made with only 0
print("Matrix A made with only 0:\n", onlyZeroMatrix, '\n')

# np.ones()
onlyOneMatrix = np.ones((C, R))                                         # Create a CxR matrix made with only 1
print("Matrix A made with only 1:\n", onlyOneMatrix, '\n')

# np.random.randint() - Generazione casuale
A = np.random.randint(5, size=(C,R))                                    # I valori della matrice sono compresi tra 0 e 5



# np.eye() - Matrice Identità
print("Matrice identità:\n", np.eye(5, k=1), '\n')                      # Matrice identità con la diagonale {sopra (+) | sotto (-)} di k posizioni

# Trasposta di una matrice
A = np.random.randint(5, size=(C,R))
print("Data la seguente matrice CxR (generata casualmente):\n", A, '\n')
print("La sua trasposta è:\n", A.T, '\n')

# Inversa di una matrice
A = np.random.randint(5, size=(3,3))
print("Data la seguente matrice CxC (generata casualmente):\n", A, '\n')
print("La sua inversa è:\n", np.linalg.inv(A), '\n')



# scipy.linalg.hilbert(n)
print("Ecco due esempi di matrici di Hilbert:\n")
print(scipy.linalg.hilbert(3), '\n', scipy.linalg.hilbert(6), '\n')    # Matrice nxn, compresa tra 10 e 20
