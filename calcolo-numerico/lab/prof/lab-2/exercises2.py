import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Mareici e norme
def es_0():
    #For more, see: help(np.linalg)
    A = np.array([[1, 2], [0.499, 1.001]])      # Riga 1 di A: [1, 2] ; Riga 2 di A: [0.499, 1.001]

    # 0.1
    #For more, see: help(np.linalg.norm)
    print("ES 1")
    print("norma p=1:", np.linalg.norm(A,1))
    print("norma p=2:", np.linalg.norm(A,2))
    print("norma Frebenious:", np.linalg.norm(A,'fro'))
    print("norma infinito:", np.linalg.norm(A,np.inf), '\n')
    
    # 0.2
    #For more, see: help(np.linalg.cond)
    print("K(A)_1 =", np.linalg.cond(A, 1))
    print("K(A)_2 =", np.linalg.cond(A, 2))
    print("K(A)_F =", np.linalg.cond(A, 'fro'))
    print("K(A)_inf =", np.linalg.cond(A, np.inf), '\n')

    # 0.3
    #x = np.ones((2, 1))
    #x = np.array([[1], [1]])
    x = np.array([[1, 1]]).T

    b = A.dot(x)           # Calcola le soluzioni b
    print("b =", b)

    # 0.4
    tilde_b = np.array([[3], [1.4985]])
    tilde_x = np.array([[2, 0.5]]).T
    # Verificare che xtilde è soluzione di A xtilde = btilde
        # A * xtilde = btilde
    print("A * tilde_x = ", A.dot(tilde_x))

    print("Delta_x ", np.linalg.norm(x - tilde_x, 2))
    print("Delta_b ", np.linalg.norm(b - tilde_b, 2))

#es_0()

import scipy.linalg.decomp_lu as LUfunctions    # LUfunctions è un nome personalizzabile
"""
LUfunction contiene - di a noi uilte - al suo interno le seguenti 3 funzioni:
    1. lu
    2. lu_factor
    3, lu_solve
Tocca guardare le info delle funzioni per capirci qualcosa.
"""
def es_1():
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    A = np.array([[3, -1, 1, -2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]], dtype=np.float)
    #n = A.shape[1]
    x = np.array([[1, 1, 1, 1]]).T
    #x = np.ones((n,1))
    b = A.dot(x)
    #b = np.matmul(A,x)
    #b = A@x

    print ("K(A) = ", np.linalg.cond(A, 1))
    print("Matrix A: \n", A)
    print("Matrix b: \n", b)
    print("\n")


    # Fattorizzazione LU con pivoting

    # A = PLU anziché PA = LU
    LU, PIV = LUfunctions.lu_factor(A) # LUfunctions returns lu e piv (vedi info)
    print("LU:\n",LU)
    print("PIV:\n",PIV)     # se il vettore in posizione 2. vale 4, allora significa che la riga 2 è stata scambiata con la riga 4"
                            # Se non vi sono stati scambi piv = [0, 1, 2, 3]
    
    # Risoluzione di: Ax=b => PLUx=b => {PLy=b & Ux=y} => {Ly=Pb & Ux=y}
    # Si risolvono due sistemi triangolari
    x_fattoriz = LUfunctions.lu_solve((LU, PIV), b)     # lu_solve returns the vector x, the solution
    print("Solution x:\n", x_fattoriz)


    """
    l = np.tril(lu, -1) + np.diag(np.ones(lu.shape[0]))
    u = np.triu(lu)
    print("l @ u =\n", l, "\n@\n", u, "\n=\n", l @ u, "\n=\n", A, "\n = A");
    my_x = LUfunctions.lu_solve((lu, piv), b)
    print("Soluzione calcolata: ", my_x)
    print("Errore relativo: ", np.linalg.norm(abs(x - my_x)) / np.linalg.norm(x))
    """

es_1()