import numpy as np
import scipy.linalg.decomp_lu as LUdec
import scipy
import matplotlib.pyplot as plt



###################################### LAB 1 #####################################################################
# EXERCISE on machine precision.
# Credo comunque che la cosa dell'approx dei float sia sbagliata
# 1-Execute the following code
import sys
help(sys.float_info)
print(sys.float_info)
# and understand the meaning of max, max_exp and max_10_exp.
  # max: maximum representable finite float
  # max_exp: maximum int e such that radix**(e-1) is representable
  # max_10_exp: maximum int e such that 10**e is representable


"""
Tra i risultati che riceviamo dopo aver eseguito questi comandi per avere le informazioni su come è implementato il floating point
in Python, vediamo che è una struttura con massimo numero rappresentabile 1.7976931348623157e+308, il cui massimo esponente possibile è 308 in base 10.
Inoltre abbiamo anche informazioni sul massimo esponente in base 2, ovvero 1024, infatti se proviamo ad eseguire:
"""
print(float(2**1023))
"""
Non abbiamo errori, invece se proviamo ad eseguire 2**1024 python ci darà un errore

2-Write a code to compute the machine precision ϵ in (float) default
precision with a WHILE construct. Compute also the mantissa digits number.
https://stackoverflow.com/questions/3478743/trying-to-write-a-code-for-finding-the-machine-epsilon
"""
def epsilon(var_type):
  pow = int(0)
  eps = var_type(1)
  while eps + 1.0 != 1.0:
    eps /= 2
    pow -= 1
  return pow + 1

print(f"La precisione del float è {epsilon(float)}")#numpy.float128
"""
Quindi sappiamo che la precisione del float in python secondo questa approssimazione, che conta il numero di iterazioni
dopo il quale eps + 1 non è più distinguibile da 1, che la mantissa del float è di 52 bit, uguale alla mantissa del double in c++.


3-Import NumPy (import numpy as np) and exploit the functions float16 and float32 in the while statement
and see the differences. Check the result of print(np.finfo(float).eps)
"""
def es3():
    """
    Import NumPy (import numpy as np) and exploit the functions float16 and
    float32 in the while statement and see the differences. Check the result of
    print(np.finfo(float).eps)
    """
    print("float16:")
    epsilon = np.float16(1.0)
    mant_dig = 1
    while np.float16(1.0) + epsilon / np.float16(2.0) > np.float16(1.0):
        epsilon /= np.float16(2.0)
        mant_dig += 1
    print("  epsilon = " + str(epsilon))
    print("  mant_dig = " + str(mant_dig))

    print("float32:")
    epsilon = np.float32(1.0)
    mant_dig = 1
    while np.float32(1.0) + epsilon / np.float32(2.0) > np.float32(1.0):
        epsilon /= np.float32(2.0)
        mant_dig += 1
    print("  epsilon = " + str(epsilon))
    print("  mant_dig = " + str(mant_dig))
    print("np.finfo(float).eps = " + str(np.finfo(float).eps))

#OUTPUT
'''
float16:
  epsilon = 0.000977
  mant_dig = 11
float32:
  epsilon = 1.1920929e-07
  mant_dig = 24
np.finfo(float).eps = 2.220446049250313e-16
'''


def es4():
    """
    Matplotlib is a plotting library for the Python programming language and its
    numerical mathematics extension NumPy, from https://matplotlib.org/
    Create a figure combining together the cosine and sine curves, from 0 to 10:
    - add a legend
    - add a title
    - change the default colors
    """
    linspace = np.linspace(0, 10);
    plt.subplots(constrained_layout = True)[1].secondary_xaxis(0.5);#a metà
    plt.plot(linspace, np.sin(linspace), color='blue')
    plt.plot(linspace, np.cos(linspace), color='red')
    plt.legend(['sin', 'cos'])
    plt.title('Sine and cosine from 0 to 10')
    plt.show()



def fibs(n):
    """
    Write a script that, given an input number n, computes the numbers of the
    Fibonacci sequence that are less than n.
    """
    if n <= 0: return 0
    if n <= 1: return 1                                                                                        
    fibs = [0, 1, 1]
    cont = 2                                                                                        
    while fibs[-1] + fibs[-2] < n:                                                                                       
        fibs.append(fibs[-1] + fibs[-2])
        cont+=1                                                                       
    return (fibs[-1],cont)#max numero sotto n raggiungibile dopo m iterazioni nella serie di fibonacci



def es6():
    """
    Write a code computing, for a natural number k, the ratio  r(k) = F(k+1) /
    / F(k), where F(k) are the Fibonacci numbers. Verify that, for a large k,
    {{rk}}k converges to the value φ=1+5√2 create a plot of the error (with
    respect to φ)
    """
    arange = np.arange(50)#1 2 3 4 5 6 7 8 9..
    plt.plot(arange, [relative_error(i) for i in arange]) #LAMBDA
    plt.legend(['relative error'])
    plt.show() #già quando x = 10 la funzione tende a 0

def r(k): # assuming k > 0
    if k <= 1: return 0                                                                                                
    fibs = [0, 1]                                                                                           
    for f in range(1, k):                                                                                      
        fibs.append(fibs[-1] + fibs[-2])                                                                         
    print(fibs[-1] / fibs[-2])
    return fibs[-1] / fibs[-2]

def relative_error(k):
    phi = (1.0 + 5 ** 0.5) / 2.0
    return abs(r(k) - phi) / phi


#########################################LAB 2####################################################################
#########################################LAB 2####################################################################
#########################################LAB 2####################################################################
#########################################LAB 2####################################################################
#########################################LAB 2####################################################################
A = np.array([[1, 2], [0.499, 1.001]])
#Calcola la norma 1, la norma 2, la norma Frobenius e la norma infinito di A
norm1 = np.linalg.norm(A, 1)
norm2 = np.linalg.norm(A, 2)
normfro = np.linalg.norm(A, 'fro')
norminf = np.linalg.norm(A, np.inf)
#Calcola il numero di condizionamento di A
#Il numero di condizionamento di A (=K(A)) è norm(A)*norm(A^-1) => errore inerente
cond1 = np.linalg.cond(A, 1)
cond2 = np.linalg.cond(A, 2)
condfro = np.linalg.cond(A, 'fro')
condinf = np.linalg.cond(A, np.inf)
#Considera il vettore colonna  x=(1,1)T  e calcola il corrispondente termine noto  b  per il sistema lineare  Ax=b
#Ax=b conosco la soluzione esatta => calcolo gli errori (problema test)
x = np.ones((2,1))
b = A.dot(x) #dot product = prodotto scalare => prodotto tra matrici = 3    1.5  2*1 + 1*1 = 3 componente per componente

#Considera ora il vettore  btilde=(3,1.4985)T  e verifica che  xtilde=(2,0.5)T  è soluzione del sistema  Axtilde=btilde
btilde = np.array([[3], [1.4985]])
xtilde = np.array([[2, 0.5]]).T #trasposto
# Verificare che xtilde è soluzione di A xtilde = btilde
# A * xtilde = btilde
btilde = A.dot(xtilde) #ho perturbato b e la soluzione è cambiata di qualche unità => mal condizionato

#Calcola la norma 2 della perturbazione sui termini noti  Δb=∥b−btilde∥2
#e la norma 2 della perturbazione sulle soluzioni  Δx=∥x−xtilde∥2 . Confronta  Δb  con  Δx .
deltax = np.linalg.norm(x-xtilde, ord=2)
deltab = np.linalg.norm(b-btilde, ord=2)

#METODI DIRETTI - risoluzione sistemi lineari
#fattorizzazione di matrici
#La soluzione viene calcolata in un numero finito di passi, modificando la matrice del problema
#in modo da rendere piú agevole il calcolo della soluzione.
#Con matrici triangolari: metodi di sostituzione;
#Con qualsiasi matrice: metodo di eliminazione di Gauss;
#Con matrici simmetriche: metodo di Cholesky.

#1 Crea il problema test in cui il vettore della soluzione esatta è  x=(1,1,1,1)T  e il vettore termine noto è  b=Ax .
# 1. creazione del problema test

A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ], dtype=float)
print ('K(A) = ', np.linalg.cond(A, 1))
n = A.shape[1] #con n indichiamo la dimensione del problema
x = np.ones((n,1))
b =  np.matmul(A,x)#andiamo a costruirci il termine noto noi conoscendo la soluzione esatta - Matrix product of two arrays.
#b= A@x

#fattorizzazione LU(LR o metodo di eliminazione gaussiana) con pivoting (=invertire o scambiare le righe)(pivot in caso a11 è uguale a 0)
#A=LR L tr inf e R tr sup (la funzione ritornerà un'unica matrice la quale sarà R diagonale compresa
#sopra e L sotto(diagonale di L sono tutti 1 a priori))
#PA=LR con P matrice di permutazione
#riguardo piv ritornato dalla funzione: "se il vettore nella posizione 2 vale 4 vuole dire che la riga 2 è
#stata scambiata con la riga 4"
#se non scambi => 0 1 2 3 => la matrice di permutazione è come se fosse la matrice identità
lu, piv = LUdec.lu_factor(A)

#RISOLUZIONE DI Ax=b => PLUx=b => {PLy=b & Ux=y} => {Ly=Pb & Ux=y}
#Si risolvono due sistemi triangolari 
#soluzione compatta
my_x=LUdec.lu_solve((lu,piv),b) #O(n^3/3)
#verifica
print("Soluzione calcolata: \n")
for i in range(n):
	print("%0.2f" %my_x[i])
er = (np.linalg.norm(x-my_x, 2))/(np.linalg.norm(x, 2)) #siccome x è un vettore dobbiamo usare delle norme per misurare la distanza
										#(altrimenti faremmo divisione elemento per elemento) (noi vogliamo una scalare)
print("%e" %er) #doppia precisione => max cifra significativa è la 16esima
#metodo alternativo
P, L, U = LUdec.lu(A) #A = P*L*U (non P*A = L*U)
print("diff = ", np.linalg.norm(A - np.matmul(P, np.matmul(L,U)), "fro"))

#if P != np.eye(n): (=matrice identità quadrata n x n)
# risoluzione
# Ax = b   <--->  PLUx = b  <--->  LUx = inv(P)b  <--->  Ly=inv(P)b & Ux=y : matrici triangolari //inv(P)*P = I
# Ax = b   <--->  PLUx = b  <--->  PLy=b & Ux=y : PL matrice non triangolare
invP = np.linalg.inv(P) #O(n^3)
y = scipy.linalg.solve_triangular(L, np.matmul(invP,b), lower=True, unit_diagonal=True)
my_x = scipy.linalg.solve_triangular(U, y, lower=False) #O(n^2/2)


#Esercizio 2
#Ripeti l'esercizio 1 sulla matrice di Hilbert (simmetrica definita positiva)
#(aij = 1/(i +j))(nota per essere mal condizionata), creata con  A=scipy.linalg.hilbert(n)  per  n=5,…,10 . In particolare:
#1) calcola il numero di condizionamento di A
#2) Considera il vettore colonna  x=(1,…,1)T , calcola il corrispondente termine noto  b  per il sistema lineare
#Ax=b  e la relativa soluzione  xtilde  usando la fattorizzazione LU come nel caso precedente.
n= 14
A=scipy.linalg.hilbert(n)
cond = np.linalg.cond(A, 2) #1) calcola il numero di condizionamento di A
x = np.ones(n).T #2) Considera il vettore colonna  x=(1,…,1)T
b = np.matmul(A,x) #calcola il corrispondente termine noto  b per il sistema lineare Ax=b

#fattorizzazione LU di A
lu, piv = LUdec.lu_factor(A)
#risolvo il sistema/i sistemi lineari
my_x=LUdec.lu_solve((lu,piv),b) #O(n^3/3)
#calcolo l'errore
er = np.linalg.norm(my_x-x, 2)/(np.linalg.norm(x, 2))
print(er)
#grafico con ascisse n e ordinata K(A)
#grafico con ascisse n e ordinata er (i due grafici hanno un andamento molto simile)
K_A = np.zeros((20, 1)) #array
Err = np.zeros((20, 1))
for n in np.arange(10, 30):
	A=scipy.linalg.hilbert(n)
	#K_A[n] = np.linalg.cond(A, 2)
	x = np.ones(n).T
	b = np.matmul(A,x)
	K_A[n-10] = np.linalg.cond(A, 2)
	lu, piv = LUdec.lu_factor(A)
	my_x=LUdec.lu_solve((lu,piv),b)
	Err[n-10] = np.linalg.norm(my_x-x, 2)/(np.linalg.norm(x, 2))
x_axis = np.linspace(10,30,num=len(Err))

plt.title("Errori relativi della matrice di Hilbert")
plt.plot(x_axis, Err)
plt.show()
plt.title("Condizionamento della matrice di Hilbert")
plt.plot(x_axis, K_A)
plt.show()

#Esercizio 3
#Scrivi le due funzioni  LTrisol()  e  UTrisol()  per implementare i metodi di sostituzione all'avanti e all'indietro, poi:
#1) usa la fattorizzazione  P,L,U=LUdec.lu(A)  sulla matrice degli esercizi precedenti;
#2) risolvi i sistemi triangolari usando la tue funzioni.
#https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html

#Soluzione di G.Genovese (human readable)
'''def LTrisol(L,b):
  n=b.size
  x=np.zeros(n)
  for i in range(0,n-1):
    x[i] = b[i]/L[i,i]
    for j in range(0,n):
        b[j] -= x[i]*L[j,i]
  x[n-1]= b[n-1]/L[n-1,n-1]
  return x

def UTrisol(U,b):
  n=b.size
  x=np.zeros(n)
  for i in range(n-1,0,-1):
    x[i] = b[i]/U[i,i]
    for j in range(0,n):
        b[j] -= x[i]*U[j,i]
  x[0]= b[0]/U[0,0]
  return x'''



def LTrisol(L,b):
  	n=b.size
  	x=np.zeros(n)
  	x[0]= b[0] / L[0, 0]
  	for i in range(1, n):
  		x[i]= (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i] #x[i]= (b[i] - np.dot(L[i, 0:1], x[0:i])) / L[i, i]
  	return x

def UTrisol(A,b):
  	n= len(b)
  	x= np.empty(n)
  	x[n-1]= b[n-1] / A[(n-1, n-1)]
  	for i in range (n-1, -1, -1): #start stop(not included) step
  		x[i] = (b[i] - np.dot(A[i,i:], x[i:])) / A[i,i] #x[i]= (b[i] - np.dot(A[i, i+1:n], x[i+1: n])) / A[i, i]
  	return x
#!!!L'esercizio consiste nel crearsi il problema test A e x per calcolare b, poi usando utrisol e ltrisol dai A e b e trova la my_x
A = np.array([[1,2,1],[1,2,2],[2,1,4]])
n = np.shape(A)
x = np.ones(n)
b = A@x

lu,piv = LUdec.lu_factor(A)
P, L, U = LUdec.lu(A)
# Ax = b   <--->  PLUx = b  <--->  LUx = inv(P)b  <--->  Ly=inv(P)b & Ux=y

invP = scipy.linalg.inv(P)
y = LTrisol(L,invP.dot(b)) #Ly=inv(P)b
my_x = UTrisol(U,y) #Ux=y #vs x che è la soluzione esatta
#my_x = LUdec.lu_solve((lu,piv),b)
#np.linalg.norm(x-my_x)/np.linalg.norm(x) per errore relativo

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

'''Esercizio 5
Calcola la fattorizzazione di Choleski sulla matrice A generata come

A=np.array([[3,−1,1,−2],[0,2,5,−1],[1,0,−7,1],[0,2,1,1]],dtype=np.float)
A=np.matmul(A,np.transpose(A)) 

usando la funzione  np.linalg.cholesky .

Verifica la correttezza della fattorizzazione.
Risolvi il sistema lineare Ax = b dove  x=(1,1,1,1)T  .'''
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ], dtype=float)
A = np.matmul(A, np.transpose(A)) #pedanteria
L = np.linalg.cholesky(A)
x = np.ones((4,1))
b = L.dot(x)

my_x = LTrisol(L,b)




#METODI ITERATIVI
#Calcolo di una soluzione come limite di una successione di approssimazioni  xk , senza modificare la struttura della matrice A.
#Sono metodi adatti per sistemi di grandi dimensioni con matrici sparse (pochi elementi non nulli).
#https://youtu.be/VH0TZlkZPRo
def Jacobi(A,b,x0,maxit,tol,xTrue): #solo con matrici con diagonale dominante
  n = np.size(x0)
  ite = 0
  x = np.copy(x0)
  
  relErr = np.zeros((maxit,1))
  errIter = np.zeros((maxit,1))
  errIter[0] = tol+1
  relErr[0] = np.linalg.norm(xTrue-x0)/np.linalg.norm(xTrue)

  while (ite <= maxit and errIter[ite] > tol):#come se avesse raggiunto di fatto l'asintoto per x che tende ad infinito
    x_old = np.copy(x)
    for i in range(0,n):#     qui sommatoria
      x[i] = ( b[i] - np.dot(A[i,0:i],x_old[0:i]) - np.dot(A[i,i+1:n],x_old[i+1:n]) ) / A[i,i]
    ite += 1
    relErr[ite] = np.linalg.norm(xTrue-x)/np.linalg.norm(xTrue)
    errIter[ite] = np.linalg.norm(x-x_old)/np.linalg.norm(x)

  relErr=relErr[:ite]
  errIter=errIter[:ite]

  return [x, errIter, relErr]

def GaussSeidel(A,b,x0,maxit,tol,xTrue): #non c'è il limite delle matrici con diagonali dominanti
  n = np.size(x0)
  ite = 0
  x = np.copy(x0)
  
  relErr = np.zeros((maxit,1))
  errIter = np.zeros((maxit,1))
  errIter[0] = tol+1
  relErr[0] = np.linalg.norm(xTrue-x0)/np.linalg.norm(xTrue)

  while (ite <= maxit and errIter[ite] > tol):
    x_old = np.copy(x)
    for i in range(0,n):#             qui no x_old differenza con Jacobi
      x[i] = ( b[i] - np.dot(A[i,0:i],x[0:i]) - np.dot(A[i,i+1:n],x_old[i+1:n]) ) / A[i,i] #calcolo nuova iterata
    ite += 1
    relErr[ite] = np.linalg.norm(xTrue-x)/np.linalg.norm(xTrue)
    errIter[ite] = np.linalg.norm(x-x_old)/np.linalg.norm(x)

  relErr=relErr[:ite]
  errIter=errIter[:ite]

  return [x, errIter, relErr]

n = 10

A = 5*np.eye(n)#        qui numero di elementi sopra o sotto la diagonale
A = A + np.diag(np.ones(n-1),1)
A = A + np.diag(np.ones(n-1),-1)
print(A)

xTrue = np.ones((n,1))#1 1 1 1 1 
b = np.matmul(A,xTrue)

x0 = np.zeros((n,1))# 0 0 0 0 0
maxit = 200
tol = 1.e-6

(xJacobi, kJacobi, relErrJacobi) = Jacobi(A,b,x0,maxit,tol,xTrue)
(xGS, kGS, relErrGS) = GaussSeidel(A,b,x0,maxit,tol,xTrue)

print('\nSoluzione calcolata da Jacobi:' )
for i in range(n):
    print('%0.2f' %xJacobi[i])

print('\nSoluzione calcolata da Gauss Seidel:' )
for i in range(n):
    print('%0.2f' %xGS[i])


# CONFRONTI

print("Errore Jacobi:", np.linalg.norm(xJacobi-xTrue)/np.linalg.norm(xTrue))
print("Errore GS:", np.linalg.norm(xGS-xTrue)/np.linalg.norm(xTrue))

# Confronto grafico degli errori di Errore Relativo

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

rangeJabobi = range(0, len(kJacobi))
rangeGS = range(0, len(kGS))

ax.plot(rangeJabobi, relErrJacobi, label='Jacobi', color='blue', linewidth=1, marker='o'  )
ax.plot(rangeGS, relErrGS, label='Gauss Seidel', color = 'red', linewidth=2, marker='.' )
legend = ax.legend(loc='upper right')
plt.xlabel('iterations')
plt.ylabel('Relative Error')
plt.title('Comparison of the different algorithms')
plt.show()