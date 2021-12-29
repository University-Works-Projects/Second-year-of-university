#import sys
#print(sys.float_info)
#max: maximum representable finite float
#max_exp: maximum int e such that radix**(e-1) is representable print(float(2**1023))
#max_10_exp: maximum int e such that 10**e is representable
def machine_precision_float(): #conta il numero di iterazioni dopo il quale eps + 1 non è più distinguibile da 1
    mant_dig = 0
    eps = float(1)
    while 1.0 + eps/2 > 1.0: #DOMANDA
        eps /= 2
        mant_dig += 1
    print(f"float precision is {mant_dig} digits")

#FUNZIONI NUMPY
import numpy as np
A = np.array([[1, 2], [0.499, 1.001]], dtype=float)
A = np.random.rand(7,7) #genera una matrice di numeri random A, di dimensione 7x7.
x = np.ones((1,2)).T #.ones() e .zeros() .T = trasposta. n = A.shape[1]; x = np.ones((n,1))
b = A@x #@ = np.matmul() = matrix multiplications. Ax=b conosco sol esatta => calcolo errori (problema test)
arange = np.arange(50)#0 1 2 3 4 5 6 7 8 9 ...49
linspace = np.linspace(0, 10, num=5, endpoint=False); #cinque elementi equispaziati da 0 a 10 con 10 ESCLUSO
main_diag = 5*np.eye(n=3, k=0) #k=+-1 =>upper/lower diagonal
A = A + np.diag(np.ones(n-1),1) #+1 => aggiunti n-1 elementi sopra la diagonale #-1 => aggiunti n-1 elementi SOTTO la diagonale
#.dot() = riga per colonna. (raddoppiare elementi matrice => A.dot(2))
#.outer() = colonna per riga = prodotto esterno/vettoriale. Outer product of two vectors is a matrix (no prodotto scalare)
X, Y = np.meshgrid(x, y) #grigliatura dominio R2 vista dall'alto in R3. Punti piano definiti come (x,y,f(x,y)) => Z=f(X,Y)
#in seguito valutiamo la nostra funzione su tutti i punti presenti nella grigliatura per disegnare superfici e curve di livello

#FUNZIONI FOR PLOTTING definendo subplot dopo subplot
import matplotlib.pyplot as plt
plt.suptitle("TITOLO DELLA FIGURA CHE CONTIENE TUTTI I SUBPLOTS")
fig, ax = plt.subplot(M, N, numerosxdxcontinuatogiu, projection='3d').secondary_xaxis(0.5) #aggiunto asse metà per f periodiche
plt.subplot(1, 2, 1, projection='3d').plot_surface(X, Y, Z, cmap='viridis').view_init(elev=45) #colormap e inclinazione
plt.contour(X, Y, Z, levels=10) #10 linee livello per mostrare cammino gradiente lungo curve plt.plot(x[0,0:k],x[1,0:k],'*')
f(x)_array = [f(i) for i in x_array]
plt.plot(x_array, f(x)_array, color='blue', marker = "o", label="Nome funzione") #marker matplotlib
plt.legend(['first function', 'nth function'], loc='upper right')
plt.xlabel("ASCISSE") #plt.ylabel("ORDINATE")
plt.grid() #squadra il grafico
plt.figure(figsize=(30, 10))
plt.imshow(X, cmap="gray") #per mostrare immagini anzichè grafici. X è una matrice
plt.title('SUBPLOT TITLE')
ax = plt.axes(projection='3d'); ax.plot_surface(v_x0, v_x1, z,cmap='viridis')#Add ax to current fig and make it the current ax
plt.show()
print("%1.2f\n%d" %(arg1, arg2)) #1 e 2 pad sse mantissa strettamente minore di 2 o parte intera strett. min di 1

#NORME <-> vettori. ERRORE RELATIVO=np.linalg.norm(x - xTrue)/np.linalg.norm(xTrue).
norm1 = np.linalg.norm(A, 1)
norm2 = np.linalg.norm(A, 2)
normfro = np.linalg.norm(A, 'fro')
norminf = np.linalg.norm(A, np.inf)

#CONDIZIONAMENTO K(A)=||A||*||A^-1||. If the condition number is not too much larger than one, the matrix is well-conditioned
cond1 = np.linalg.cond(A, 1)
cond2 = np.linalg.cond(A, 2)
condfro = np.linalg.cond(A, 'fro')
condinf = np.linalg.cond(A, np.inf)

#LU FACTORING PIVOT risoluzione sistemi lineari tramite fatt. matrice per rendere piú agevole calcolo soluzione. #DOMANDA
import scipy.linalg.decomp_lu as LUdec
import scipy
lu, piv = LUdec.lu_factor(A)
#l = np.tril(lu, -1) + np.diag(np.ones(lu.shape[0])) #-1 = diagonal below
#u = np.triu(lu)
my_x = LUdec.lu_solve((lu,piv),b) #vettore n-dimensionale

#LU FACTORING PIVOT - metodo alternativo Ax = b   <--->  PLUx = b  <--->  LUx = inv(P)b  <--->  Ly=inv(P)b & Ux=y
P, L, U = LUdec.lu(A) #print("diff = ", np.linalg.norm(A - np.matmul(P, np.matmul(L,U)), "fro"))
invP = np.linalg.inv(P) #O(n^3) #DOMANDA
y = scipy.linalg.solve_triangular(L, np.matmul(invP,b), lower=True, unit_diagonal=True) #LTrisol(L,invP.dot(b))
my_x = scipy.linalg.solve_triangular(U, y, lower=False) #my_x = UTrisol(U,y)

#METODI DI SOSTITUZIONE per matrici triangolari
def LTrisol(L,b): #all'avanti
	n=b.size
	x=np.zeros(n)
	x[0]= b[0] / L[0, 0]
	for i in range(1, n):
  		x[i]= (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i] #:i => i escluso
	return x
def UTrisol(A,b): #all'indietro
  	n= len(b)
  	x= np.empty(n)
  	x[n-1]= b[n-1] / A[(n-1, n-1)]
  	for i in range (n-1, -1, -1): #start stop(not included) step size
  		x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i,i] #i: => i INCLUSO
  	return x

#LU FACTORING WITHOUT PIVOT https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html
def LU_fact_NOpiv(A):
	n = A.shape[0] #Get the number of rows
	U = A.copy().astype('float32') #float32 altrimenti problemi di casting
	L = np.eye(n).astype('float32')
	for i in range(n): #Loop over rows
		if U[i, i] != 0: #sse il perno è diverso da 0
			factor = U[i+1:, i] / U[i, i]
			L[i+1:, i] = factor #aggiunto array factor sotto il perno dell'iesima colonna (L era matrice identità)
			U[i+1:] -= factor[:, np.newaxis] * U[i] #per tutte le righe sotto l'iesima, t.c. sotto il perno in U solo zeri.
	return L, U

#CHOLESKY per matrici simmetriche definite positive Ly=b & Ux=y; U=L.T
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ], dtype=float)
A = np.matmul(A, np.transpose(A)) #DOMANDA
L = np.linalg.cholesky(A) #U=L.T
b = A @ xTrue #x fornita
y = LTrisol(L, b)
my_x = UTrisol(L.T, y)


#JACOBI & GAUSS-SIDEL-METODI ITERATIVI: sol. come limite di una successione di approssimazioni  xk , senza modificare matrice A
def Jacobi(A, b, x0): #Jacobi solo matrici diagonale dominante, Gauss-Sidel no questo vincolo https://youtu.be/VH0TZlkZPRo
	maxit = 200
	tol = 1.e-6 #differenza relativa fra due iterati successivi
	n = np.size(x0) #numero variabili
	ite = 0
	x = np.copy(x0) #estimated result so far
	errIter = np.zeros((maxit,1))
	errIter[0] = tol+1
	while (ite < maxit and errIter[ite] > tol): #tolleranza tol sulla differenza relativa fra due iterati successivi
		x_old = np.copy(x) #x_old e x usati anche per determinare se passi sono abbastanza lunghi - x nuova calcolata su x_old
		for i in range(0,n):#l'unica differenza con Gauss-Sidel è x anzichè x_old nel primo np.dot()
			x[i] = ( b[i] - np.dot(A[i,0:i],x_old[0:i]) - np.dot(A[i,i+1:n],x_old[i+1:n]) ) / A[i,i]
		ite += 1
		errIter[ite] = np.linalg.norm(x-x_old)/np.linalg.norm(x)
	
	errIter=errIter[:ite]
	return [x, errIter]

#MINIMI QUADRATI applicato alla approssimazione dei dati con un polinomio => polinomio che passa vicino a tutti i punti.
#FONTI: https://www.youtube.com/watch?v=jEEJNz0RK4Q https://www.youtube.com/watch?v=YwZYSTQs-Hk
#Le incognite sono le alpha del polinomio poichè fissato un grado dobbiamo scegliere fra le infinite rette del piano una retta
#coefficienti sono le ordinate degli actual values => p(x[i])=alpha0 + alpha1*x[i] + ... alphan*x[i]^n per tutti i punti iesimi
#I valori della matrice A sono gli expected values
n = 5 # Grado del polinomio approssimante
x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]) #expected values
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5]) #actual values
N = x.size # Numero dei dati
A = np.zeros((N, n+1)) #Aij=l'iesima x^j partendo da 0 incluso a n
for j in range(n+1):
	A[:, j] = x**j #=> scorrendo ogni colonna, tutti elementi vettore x elevati alla j
#sse la matrice 𝐴𝛼 (𝛼 = incognite x) ha rango massimo (si dimostra che se i punti sono distinti la matrice ha rango massimo)
#=> Il problema min𝛼||𝐴𝛼−𝑦||^2 può essere risolto col metodo delle equazioni normali, ossia osservando che
#il problema può essere riscritto come: 𝐴𝑇𝐴𝛼=𝐴𝑇𝑦 Risolvendo questo sistema lineare (ad esempio con fattorizzazione di Cholesky
#o con metodi iterativi) si ottiene il vettore degli 𝛼 che corrisponde ai coefficenti del polinomio approssimante.
ATA = np.dot(A.T, A) #ATA è simmetrica definita positiva => Cholesky
lu, piv = LUdec.lu_factor(ATA)
ATy = np.dot(A.T, y)
alpha_normali = LUdec.lu_solve((lu, piv), ATy) #risolto 𝐴𝑇𝐴𝛼=𝐴𝑇𝑦
#altrimenti applicare la fattorizzazione SVD la quale si può applicare per qualsiasi rango della matrice, in qualsiasi caso.
#operazioni diverse e processori diversi=> errori algoritmici diversi
#La decomposizione SVD è unica, ove:
#A MxN
#U : Unitary matrix having left singular vectors as columns. shape (M, r)
#s(sigma) : diagonal matrix with singular values, sorted in non-increasing
#			order. shape (r, r). s.shape = r = rango della matrice A
#Vh(V.T) : Unitary matrix having right singular vectors as rows. shape(N, r)
U, s, Vh = scipy.linalg.svd(A) #U∑V, h = trasposta coniugata
alpha_svd = np.zeros(s.shape) #inizializzazione vettore rxr
for i in range(n+1): 
	ui = U[:,i] #colonne della matrice U
	vi = Vh[i,:] #righe della matrice Vh
	alpha_svd = alpha_svd + (ui @ y) * vi/ s[i] #alpha=∑i=1 up to N=(uiTy)vi/si
	#s[i] scalare. il comando np.dot() prima traspone di default il primo vettore poi lo moltiplica per il secondo vettore
#np.linalg.norm(alpha_normali - alpha_svd) / np.linalg.norm(alpha_svd)

def p(alpha, x): #calcolare immagine del polinomio di grado 5 che interpola/approssima/passa vicino tutti i punti
	N = len(x) #Numero dei dati passati come argomento
	n = len(alpha) #5 + 1 (poichè ^0 incluso)
	A = np.zeros((N,n))
	for i in range(n):
		A[:, i] = x ** i
	return A @ alpha #x_plot = np.linspace(1, 3, 100), y_normali = p(alpha_normali, x_plot)

#COMPRESSIONE IMMAGINE tramite fattorizzazione in valori singolari SVD https://www.youtube.com/watch?v=DG7YTlGnCEo
#Approximate higher rank matrices as a sum of rank 1 matrices!
#A=U∑Vh=∑i=1 upto k sigmai*ui*viT somma di prodotti esterni. + rango k grande più vicini all'imm. di partenza ma meno compressa 
#sigmai (singular values) ordinati secondo la loro preminenza nel ricostruire l'immagine originale
from skimage import data #set di immagini precaricate salvate come matrici
A = data.camera().astype(np.float64) #oppure caricare immagine da file e salvata come matrice => A = plt.imread('foto.jpg')
A = A[3000:4000, 5000:6000] #crop su matrice immagine => nuova dimensione è 1000x1000
U, s, Vh = scipy.linalg.svd(A)
A_k = np.zeros(A.shape) #init immagine compressa che tende all'immagine originale aumentando il suo rango k fino a r originale
k_max = 20
for i in range(k_max): #fattore di compressione ck = min(m,n)/(k+1) -1 (k+1 perchè 0-based)
	ui = U[:, i]
	vi = Vh[i, :] #Vh è gia V trasposto (di default)
	A_k = A_k + np.outer(ui, vi) * s[i] #somma di matrici rango 1 pesate. aggiungendo matr. di r=1 l'immagine. diventa + chiara

#ZERI DI FUNZIONE - METODO APPR0SSIMAZIONI SUCCESSIVE (=> metodo bisezione, etc..) https://youtu.be/ucz233Izov0
#calcolare radici funzione non lineare => anzichè cercare direttamente 0 di f si calcola iterat. punto fisso di una funzione g.
#All'interno del ciclo devo calcolare l'iterato xk come valore di una funzione g in xk-1.
#k è numero iterazioni che ha impiegato il metodo per arrivare alla soluzione tollerata/approssimata/troncata di x*(sol esatta)
#La f non è la g: nel metodo delle approssimazioni successive si passa dalla risoluzione di
#un' eq non lineare al calcolo del punto fisso di un'eq g.
#Es. Metodo di Newton (ottenuto prendendo g(x) = x - f(x)/f'(x)), ha velocità di convergenza quadratica (=>p = 2) =>converge +
#velocemente alla soluzione rispetto altre g => per la g relativa al metodo di Newton il numero di iterazioni per arrivare
#alla stessa precisione è + basso https://youtu.be/rmcG1ef6Nkw Per avere la convergenza ci vogliono certe condizioni(Wolfe)
def succ_app(f, g, x0=0): #f=lambda x: np.exp(x)-x**2, gi=lambda x: x-f(x)*np.exp(x/2) np.exp(j) = e**j
	tolx= 10**(-10)
	tolf = 10**(-6) #se f:Rn->R=>no norma poichè no vettore!
	i = 0
	maxit=100
	err=np.zeros(maxit+1, dtype=np.float64)#diff tra due iterate xk successive per tolx; +1 perchè 0-based e init condizione
	err[0]=tolx+1 #+1 in modo che appena iniziamo il ciclo siamo sicuri che la condizione relativa all'errore sia verificata
	x=x0
	#la distanza tra iterati successivi dev'essere sopra certa tolleranza e valore assoluto di f(x) dev'essere sopra certa 
	#soglia. Se scendiamo sotto certa soglia => la nostra f valutata in x è molto vicina a 0.
	#Usciamo dal ciclo quando entrambe le condizioni di convergenza sotto sono false (criteri assoluti).
	#tolleranze poichè aritmetica finita e non esatta/continua/reale. Condizioni di convergenza <--> criteri d'arresto
	while (i<maxit and (err[i]>tolx or abs(f(x))>tolf) ): #finchè entrambe condizioni verificate
		x_new=g(x) #x_new = xk
		err[i+1]=abs(x_new-x) #scarto assoluto tra iterati
		i+=1
		x=x_new #best estimated result so far
	err=err[0:i]
	return (x, i, err) #i= numero di iterazioni compiute per arrivare al risultato

#METODO DEL GRADIENTE - OTTIMIZZAZIONE NON VINCOLATA
#Il metodo del gradiente risolve iterativamente il problema di calcolare l'argmin =>ascissa punto minimo funzione f(x):Rn->R)
#L'algoritmo converge ad un punto stazionario (in generale un minimo locale). Se f convessa => unico minimo che è globale 
#L'iterato xk si calcola xk+1 = xk -alphak*∇f(xk) ove alphak calcolato ad ogni iter con la procedura di backtracking la quale 
#parte da un guess iniziale alpha=1, controllando le PROPRIE condizioni di terminazione (non c'entrano niente
#con le condizioni di terminazione dell'algoritmo del gradiente).
#Se queste condizioni sono verificate l'alpha va bene altrimenti viene dimezzato finchè non si verificano le condizioni - entro
#un numero massimo di iterazioni.
#Se la procedura di backtracking termina per il numero massimo di iterazioni significa che non abbiamo trovato un alpha buono 
#=> l'algoritmo del gradiente si arresta e non possiamo calcolare il nostro minimo.
#
#while condizioni di arresto negate dell'algoritmo del gradiente:
#	while condizioni di arresto negate della procedura di backtracking:
def next_step(x,grad): # backtracking procedure for the choice of the steplength
	alpha=1.1
	rho = 0.5
	c1 = 0.25
	p=-grad
	j=0
	jmax=10
	#condizioni che servono per soddisfare dei criteri di convergenza - condizioni di Wolfe
	while ((f(x[0]+alpha*p[0],x[1]+alpha*p[1]) > f(x[0],x[1])+c1*alpha*grad.T@p) and j<jmax):
		alpha= rho*alpha #dimezzata
		j+=1
	if (j>jmax): return -1
	else: return alpha #se termina correttamente assicura convergenza a un punto stazionario.
def minimize(x0): #punti xi := np.array((ordinata,ascissa))
	k=0
	MAXITERATIONS=1000
	ABSOLUTE_STOP=1.e-5 #soglia minima per gradiente che deve tendere a 0
	x_last = np.array([x0[0],x0[1]]) #initialize first values; x e y
	while (np.linalg.norm(grad_f(x_last))>ABSOLUTE_STOP and k < MAXITERATION ):#se fissassimo una len di passo sarebbe + lento 
		k+=1
		grad = grad_f(x_last)#calcolare il gradiente funzione valutata sull'ultima iterata
		step = next_step(x_last,grad) #caso in cui next_step ritorni -1 omesso altrimenti exit no convergenza
		x_last=x_last-step*grad
	return (x_last, k)
def f(x1,x2): return 10*(x1-1)**2 + (x2-2)**2 #f(x1,x2):Rn->R => return numero#x0 = np.array((3,-5))
def grad_f(x): return np.array([20*(x[0]-1),2*(x[1]-2)]) #Df(x,y) => return np.array([derivata parziale per ogni incognita xi])