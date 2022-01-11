import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#import numpy.linalg.norm as norma

# Matrici e norme
def es_0():
    """
        Considera la matrice A
        A=(1     2     )
            (0.499 1.001 )
        Calcola la norma 1, la norma 2, la norma Frobenius e la norma infinito di A
        con  numpy.linalg.norm() (guarda l'help della funzione).
        Calcola il numero di condizionamento di A con numpy.linalg.cond() (guarda
        l'help della funzione).
        Considera il vettore colonna x=(1,1)T e calcola il corrispondente termine
        noto b per il sistema lineare Ax=b.
        Considera ora il vettore b~=(3,1.4985)T e verifica che x~=(2,0.5)T è
        soluzione del sistema Ax~=b~.
        Calcola la norma 2 della perturbazione sui termini noti Δb=∥b−b~∥2 e la
        norma 2 della perturbazione sulle soluzioni Δx=∥x−x~∥2.
        Confronta Δb con Δx.
    """
    #For more, see: help(np.linalg)
    A = np.array([[1, 2], [0.499, 1.001]])      # Riga 1 di A: [1, 2] ; Riga 2 di A: [0.499, 1.001]

    # 0.1
    #For more, see: help(np.linalg.norm)
    print("ES 0")
    print("norma p=1:", norma(A,1))
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

    print("Delta_x =", np.linalg.norm(x - tilde_x, 2))
    print("Delta_b =", np.linalg.norm(b - tilde_b, 2))

#es_0()

#################################################################### Es 1 ####################################################################
#################################################################### Es 1 ####################################################################
#################################################################### Es 1 ####################################################################

import scipy.linalg.decomp_lu as LUfunctions    # LUfunctions è un nome personalizzabile
import scipy.linalg
"""
    LUfunction contiene - di a noi uilte - al suo interno le seguenti 3 funzioni:
        1. lu
        2. lu_factor
        3, lu_solve
    Tocca guardare le info delle funzioni per capirci qualcosa.
"""
def es_1():
    """
        Considera la matrice
        A = ⎛ 3 -1  1 -2⎞
            | 0  2  5 -1|
            | 1  0 -7  1|
            ⎝ 0  2  1  1⎠
        
        Crea il problema test in cui il vettore della soluzione esatta è
        x=(1,1,1,1)T e il vettore termine noto è b=Ax. Guarda l'help del modulo
        scipy.linalg.decomp_lu e usa una delle sue funzioni per calcolare la
        fattorizzazione LU di A con pivolting. Verifica la correttezza dell'output.
        Risolvi il sistema lineare con la funzione lu_solve del modulo decomp_lu
        oppure con scipy.linalg.solve_triangular. Visualizza la soluzione calcolata
        e valutane la correttezza.
    """
    print("ES 1")
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
    print("Matrix A =\n", A)
    print("Matrix b =\n", b)
    print("\n")


    # Risoluzione del sistema
    #es_1_fattorizzazione_LU_con_pivoting_v1 (A, x, b)
    es_1_fattorizzazione_LU_con_pivoting_v2 (A, b)

def es_1_fattorizzazione_LU_con_pivoting_v1 (A, x, b):
    # Fattorizzazione LU con pivoting

    # A = PLU anziché PA = LU
    LU, PIV = LUfunctions.lu_factor(A) # LUfunctions returns lu e piv (vedi info)
    print("LU =\n",LU)
    print("PIV =\n",PIV)     # se il vettore in posizione 2. vale 4, allora significa che la riga 2 è stata scambiata con la riga 4"
                            # Se non vi sono stati scambi piv = [0, 1, 2, 3]
    
    # Risoluzione di: Ax=b => PLUx=b => {PLy=b & Ux=y} => {Ly=Pb & Ux=y}
    # Si risolvono due sistemi triangolari
    x_fattoriz = LUfunctions.lu_solve((LU, PIV), b)     # lu_solve returns the vector x, the solution
    print("Solutione x calcolata:\n", x_fattoriz)

    print("Errore relativo: ", np.linalg.norm(x_fattoriz-x, 2) / np.linalg.norm(x))

def es_1_fattorizzazione_LU_con_pivoting_v2 (A, b):
    P, L, U = LUfunctions.lu(A)     # See info of .lu. Remember that: A = P*L*U,  lu returns -> P, L and U
    print ('A =\n', A)
    print ('P =\n', P)
    print ('L =\n', L)
    print ('U =\n', U)
    print ('P * L * U =\n', P @ (L @ U))

    print ('diff = ',   np.linalg.norm(A - P @ (L @ U)), 'fro'  )        # RIGA NON CAPITA

    P_inversa = np.linalg.inv(P)
    y = scipy.linalg.solve_triangular(L, P_inversa @ b, lower=True, unit_diagonal=True)
    x_fattoriz = scipy.linalg.solve_triangular(U, y, lower=False)
    print("Solutione x calcolata:\n", x_fattoriz)

#es_1()

#################################################################### Es 2 ####################################################################
#################################################################### Es 2 ####################################################################
#################################################################### Es 2 ####################################################################

def es_2():
    """
        Ripeti l'esercizio 1 sulla matrice di Hilbert, creata con
        A=scipy.linalg.hilbert(n) per n=5,…,10. In particolare:
        - calcola il numero di condizionamento di A
        - considera il vettore colonna  x=(1,…,1)T
        - calcola il corrispondente termine noto b per il sistema lineare Ax=b e la
            relativa soluzione x~ usando la fattorizzazione LU come nel caso
            precedente.
    """
    print("ES 2")
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    n = 15                          # Si ha dell'instabilità per valori di n > 12 circa
    A = scipy.linalg.hilbert(n)
    x = np.ones((n,1))
    print("x =\n", x)
    print ("K(A) = ", np.linalg.cond(A, 2))

    b = A @ x

    # Fattorizzazione LU di A
    LU, PIV = LUfunctions.lu_factor(A)

    # Risoluzione del sistmea lineare
    x_fattoriz = LUfunctions.lu_solve((LU, PIV), b)

    # Calcolo dell'errore
    print("Errore relativo: ", np.linalg.norm(x_fattoriz-x, 2) / np.linalg.norm(x))

#es_2()

#################################################################### Es 2 - BONUS ####################################################################
#################################################################### Es 2 - BONUS ####################################################################
#################################################################### Es 2 - BONUS ####################################################################

import matplotlib.pyplot as plt

def es_2_bonus_graph_K_A():
    """
        Grafico del valore K(A) (condizionamento di A) al variare di n.
    """

    K_A = np.zeros((20,1))
    Err = np.zeros((20,1))
    for n in np.arange(10,30):
        A = scipy.linalg.hilbert(n)
        x = np.ones((A.shape[1], 1))
        b = A @ x
        K_A[n-10] = np.linalg.cond(A)
        LU, PIV = LUfunctions.lu_factor(A)
        x_fattoriz = LUfunctions.lu_solve((LU, PIV), b)
        Err[n-10] = np.linalg.norm(x_fattoriz, 2) / np.linalg.norm(x)

    x = np.arange(10, 30)

    plt.plot(x, K_A, color='blue')
    plt.title("CONDIZIONAMENTO DI A: K(A)")
    plt.xlabel("Dimendione matrice: n")
    plt.ylabel("K_A")
    plt.show()

def es_2_bonus_graph_relative_Err():
    """
        Grafico dell'errore di K(A).
    """

    K_A = np.zeros((20,1))
    Err = np.zeros((20,1))
    for n in np.arange(10,30):
        A = scipy.linalg.hilbert(n)
        x = np.ones((A.shape[1], 1))
        b = A @ x
        K_A[n-10] = np.linalg.cond(A)
        LU, PIV = LUfunctions.lu_factor(A)
        x_fattoriz = LUfunctions.lu_solve((LU, PIV), b)
        Err[n-10] = np.linalg.norm(x_fattoriz, 2) / np.linalg.norm(x)

    x = np.arange(10, 30)

    plt.plot(x, Err, color='red')
    plt.title('ERRORE RELATIVO')
    plt.xlabel('Dimensione matrice: n')
    plt.ylabel('Err = ||my_x-x||/||x||')

    plt.show()

#es_2_bonus_graph_K_A()
#es_2_bonus_graph_relative_Err()

#################################################################### Es 3 ####################################################################
#################################################################### Es 3 ####################################################################
#################################################################### Es 3 ####################################################################

def es_3():
    """
        Scrivi le due funzioni LTrisol() e UTrisol() per implementare i metodi di
        sostituzione all'avanti e all'indietro, poi:
            - usa la fattorizzazione P,L,U=LUdec.lu(A) sulla matrice degli esercizi
                precedenti;
            - risolvi i sistemi triangolari usando la tue funzioni.
    """
    print("ES 3")
    """
    def LTrisol(L,b):
    n=b.size
    x=np.zeros(n)
    x[0]=  ... 
    for i in ...:
        x[i]= ...
    return x

    def UTrisol(U,b):
    n=b.size
    x=np.zeros(n)
    x[n-1]=  ...
    for i in ...:
        x[i]=...
    return x

    A = ...
    n = ...
    x = ...
    b = ...

    ... = LUdec.lu(A)


    # Ax = b   <--->  PLUx = b  <--->  LUx = invPb  <--->  Ly=invPb & Ux=y
    ...
    my_x = ...

    print('\nSoluzione calcolata:' )
    for i in range(n):
        print('%0.2f' %my_x[i])
    """

#################################################################### Es 4 ####################################################################
#################################################################### Es 4 ####################################################################
#################################################################### Es 4 ####################################################################

def es_4():
    """
        Comprendere i seguenti codici che implementano la fattorizzazione LU senza
        pivoting.
    """
    print("ES 4")

    """
    # LU senza pivoting

    def LU_fact_NOpiv(A):
    a = np.copy(A)
    n=a.shape[1]
    
    for k in range(n-1):
        if a[k, k] != 0:
            a[k+1:, k] = a[k+1:, k]/a[k,k]
            
            a1 = np.expand_dims(a[k+1:, k], 1)
            a2 = np.expand_dims(a[k, k+1:], 0)
            a[k+1:, k+1:] = a[k+1:, k+1:] - (a1 * a2)
    return a
    """

#################################################################### Es 5 ####################################################################
#################################################################### Es 5 ####################################################################
#################################################################### Es 5 ####################################################################

def es_5():
    """
        Calcola la fattorizzazione di Choleski sulla matrice A generata come
        A=np.array([[3,-1,1,-2],[0,2,5,-1],[1,0,-7,1],[0,2,1,1]],dtype=np.float)
        A=np.matmul(A,np.transpose(A)) 
        usando la funzione np.linalg.cholesky.
        Verifica la correttezza della fattorizzazione.
        Risolvi il sistema lineare Ax = b dove  x = (1,1,1,1)T.
    """
    print("ES 5")

#################################################################### Es 6 ####################################################################
#################################################################### Es 6 ####################################################################
#################################################################### Es 6 ####################################################################

def es_6():
    """
        Scrivi le funzioni Jacobi(A, b, x0, maxit, tol, xTrue) e
        GaussSeidel(A, b, x0, maxit, tol, xTrue) per implementare i metodi di Jacobi e
        di Gauss Seidel per la risoluzione di sistemi lineari con matrice a diagonale
        dominante. In particolare:
        - x0  sia l'iterato iniziale;
        - la condizione d'arresto sia dettata dal numero massimo di iterazioni
            consentite maxit e dalla tolleranza tol sulla differenza relativa fra due
            iterati successivi.
        Si preveda in input la soluzione esatta xTrue per calcolare l'errore relativo
        ad ogni iterazione. Entrambe le funzioni restituiscano in output:
        - la soluzione x;
        - il numero  k  di iterazioni effettuate;
        - il vettore  relErr  di tutti gli errori relativi.
    """
    print("ES 6")

#################################################################### Es 7 ####################################################################
#################################################################### Es 7 ####################################################################
#################################################################### Es 7 ####################################################################

def es_7():
    """
        Testa le due funzioni dell'esercizio precedente per risolvere il sistema
        lineare Ax = b dove A è la matrice 10x10
            A =  
            ⎛ 5  1  0  0 ... 0 ⎞
            | 1  5  1  0 ... 0 |
            | 0  1  ⋱  ⋱  ⋮  ⋮ |
            | 0  0  ⋱  5  1  0 |
            | 0  0 ... 1  5  1 |
            ⎝ 0  0 ... 0  1  5 ⎠
        e x=(1,1,...,1)T la soluzione esatta.
        Confronta i due metodi e grafica in un unico plot i due vettori relErr.
    """
    print("ES 7")








