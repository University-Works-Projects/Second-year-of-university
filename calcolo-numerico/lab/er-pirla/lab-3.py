'''
1 esercizio)soluzione del problema dei minimi quadrati applicato alla approssimazione dei
dati con un polinomio.
Dati alcuni punti nel piano chiede di calcolare un polinomio (il grado varierà)
che approssima questi punti, quindi un polinomio che passa vicino ai punti.
Quello che passa più vicino ai punti, nel senso di una distanza in norma due
(quindi euclidea) dei residui, porta a risolvere il problema di minimi quadrati.
Le incognite sono le alpha del polinomio perchè fissato  un grado dobbiamo
scegliere fra le infinite rette del piano una retta
termine noto sono le ordinate degli actual value (https://www.youtube.com/watch?v=YwZYSTQs-Hk)
(https://www.youtube.com/watch?v=jEEJNz0RK4Q)
I valori della matrice A sono gli expected value della nostra funzione
dal disegno del polinomio otterremo una rappresentazione grafica dei punti da
approssimare e della funzione approssimante che è stata calcolata

2)l'approssimazione (compressione) di una matrice che rappresenta un'immagine
utilizzando la fattorizzazione in valori singolari SVD
'''
import numpy as np
import matplotlib.pyplot as plt
#Fornito {(xi,yi)} set di N punti che devono essere approssimati da un polinomio di grado n fissato
#min𝛼||𝐴𝛼−𝑦||^2
n = 5 # Grado del polinomio approssimante

x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]) #expected value
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5]) #actual value
N = x.size # Numero dei dati

A = np.zeros((N, n+1)) #A ∈ R^N x n+1 t.c. Aij=l'iesima x^j partendo da 0 a n

for j in range(n+1): #scorrendo le colonne
	A[:, j] = x**j
'''
Abbiamo gli ingredienti per risolvere il problema dei minimi quadrati 𝐴𝛼−𝑦 per
calcolare alpha
sse la matrice 𝐴𝛼 ha rango massimo (si dimostra che se i punti sono distinti
la matrice ha rango massimo) => metodo delle equazioni normali (sistema lineare)
altrimenti applicare la fattorizzazione SVD che si può applicare per qualsiasi
rango della matrice, cioè in qualsiasi caso.
operazioni diverse => errori algoritmici diversi
   

Il problema ai minimi quadrati min𝛼||𝐴𝛼−𝑦||^2può essere risolto col metodo 
delle equazioni normali, ossia osservando che il problema di minimo può essere
riscritto come: 𝐴𝑇𝐴𝛼=𝐴𝑇𝑦
Risolvendo questo sistema lineare (ad esempio con fattorizzazione di Cholesky
o con metodi iterativi) si ottiene il vettore degli 𝛼 che corrisponde ai
coefficenti del polinomio approssimante.
'''

import scipy.linalg
import scipy.linalg.decomp_lu as LUdec

# Per chiarezza, calcoliamo prima la matrice del sistema
#poi subito dopo il termine noto a parte
ATA = np.dot(A.T, A) 
ATy = np.dot(A.T, y)
lu, piv = LUdec.lu_factor(ATA)

alpha_normali = LUdec.lu_solve((lu, piv), ATy) 

#Risoluzione tramite SVD - la decomposizione è unica - A MxN
#U : Unitary matrix having left singular vectors as columns. shape (M, r)
#s(sigma) : diagonal matrix with singular values, sorted in non-increasing
			#order. shape (r, r)
#Vh(V.T) : Unitary matrix having right singular vectors as rows. shape(N, r)
U, s, Vh = scipy.linalg.svd(A) #U∑V, h = trasposta coniugata
alpha_svd = np.zeros(s.shape) #s.shape = r = rango della matrice A
for i in range(n+1):
	#per estrarre delle righe o delle colonne da una matrice possiamo usare la
	#notazione ":" che ci serve per individuare gli indici da estrarre
	#se mettiamo i due punti su una dimensione estrae tutta la dimensione altrimenti
	#si usa la tupla (i:j) per estrarre solo una porzione
	#estrarre una colonna => ":" = su tutte le righe poi indice della colonna
	#da estrarre (è come se fosse (1,i) + (2,i) + ... + (n,i))
	ui = U[:,i] #colonne della matrice U #i = indice rispetto alle righe
	vi = Vh[i,:] #righe della matrice Vh
	#alpha=∑i=1 up to N=(uiTy)vi/si
	alpha_svd = alpha_svd + (np.dot(ui, y) * vi)/ s[i] #s[i] scalare. il comando
	#np.dot() prima traspone di default il primo vettore poi lo moltiplica per
	#il secondo vettore
np.linalg.norm(alpha_normali - alpha_svd) / np.linalg.norm(alpha_svd) #calcoliamo
#la differenza per vedere quanto sono distanti in norma due
#stessa operazione ma processori diversi => avremo errori algoritmici diversi

#adesso che abbiamo i coefficienti alpha disegniamo il polinomio assieme ai punti
#il polinomio dovrebbe passare vicino ai punti

#dobbiamo calcolare partire dai coefficienti l'immagine del polinomio
def p(alpha, x): #valutazione dell'approssimazione del polinomio
	N = len(x)
	n = len(alpha)
	A = np.zeros((N,n))
	for i in range(n): A[:, i] = x ** i #oppure ciclo sul numero di ascisse che ho scelto ed
	#oppure ciclo sul numero di ascisse che ho scelto ed
	#in ogni ascissa mi calcolo il polinomio
	#p(x[i]) = alpha0 + alpha1*x[i] + ... . alphan*x[i]^n per tutti i punti
	#noi abbiamo calcolato il vettore in modo compatto:
	#A[1,i] = x**1, A[2,i] = x ** 2, fino all' n riga
	return A.dot(alpha) #btw: @ = np.matmul() = matrix multiplications
    #quando si fa un prodotto matrice per vettore => il numero di colonne
    #della matrice al numero di righe del vettore
#the figure has 1 row, 2 columns, and this plot is the first plot. 
x_plot = np.linspace(1, 3, 100) #100 punti equidistanti tra loro nell'intervallo [1, 3] 
y_normali = p(alpha_normali, x_plot)
y_svd = p(alpha_svd, x_plot)
plt.figure(figsize=(20, 10))
#the final figure will have 1 row, 2 columns, and this plot will be the first plot. 
plt.subplot(1, 2, 1) #subplot() => you can draw multiple plots in one figure
plt.plot(x, y, 'o') #'o' è un marker per identificare l'elemento nel grafico
#https://matplotlib.org/stable/api/markers_api.html
plt.plot(x_plot, y_normali, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Approssimazione tramite Eq. Normali')
plt.grid()
plt.subplot(1, 2, 2)#subplot sottostante al primo (due disegni, uno sopra e uno sotto)
plt.plot(x, y, 'o')
plt.plot(x_plot, y_svd, 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Approssimazione tramite SVD')
plt.grid()
plt.show()



###2 esercizio###
'''
https://www.youtube.com/watch?v=DG7YTlGnCEo
approssimazione (compressione) di una matrice che rappresenta un'immagine
la matrice come dimensione ha il numero dei pixel dell'immagine
utilizzando la fattorizzazione in valori singolari SVD
!!Approximate higher rank matrices as a sum of rank 1 matrices!!
1) A = U∑Vh, U e Vh rotation matrices, ∑ scaling/stretching matrix
sigma11= horizontal stretching of the plane by a factor of sigma11
sigma22= vertical stretching of the plane by a factor of sigma11
2) A = U∑Vh = ∑i=1 up to k sigmai*ui*viT, più k è grande,
più ci avviciniamo all'immagine di partenza (=> più approssimata ma meno compressa)
compressa significa da rango originale r a rango k, sigmai*ui*viT è una diade
ui = colonne di U, sigmai = elementi diagonali di ∑, viT = righe trasposte di V
#colonna di V = riga di V.T
sigmai (singular values) ordinati secondo la loro preminenza nel ricostruire l'immagine originale
'''
'''from skimage import data #set di immagini precaricate salvate come matrici
A = data.camera() #camera-man
plt.imshow(A, cmap='gray')
plt.show()'''
#oppure caricare immagine da file e salvata come matrice
A = plt.imread('Stereo-1_channel_image_of_Phobos_ESA214117.jpg')
A = A[3000:4000, 5000:6000] #crop su matrice immagine => nuova dimensione è 1000x1000
U, s, Vh = scipy.linalg.svd(A)
A_k = np.zeros(A.shape) #immagine compressa che tende all'immagine originale aumentando il suo rango k fino a r #A_p
k_max = 20 #p_max
err_rel = np.zeros((k_max))
c = np.zeros((k_max)) #fattore di compressione ck = (1/k) * min(m,n) -1
for i in range(k_max):
	ui = U[:, i]
	vi = Vh[i, :] #Vh è gia V trasposto (di default)
	A_k = A_k + np.outer(ui, vi) * s[i] #somma di matrici rango 1 pesate. s[i] scalare
	#outer product of two vectors is a matrix (no prodotto scalare)
	#.dot() = riga per colonna
	#.outer() = colonna per riga = prodotto esterno/vettoriale
	plt.imshow(A_k, cmap='gray')
	plt.title("k=" + str(i) + " , k->r")
	plt.show() #aggiungendo matrici di rango 1 l'immagine diventa più chiara
	err_rel[i] = np.linalg.norm(A_k - A) / np.linalg.norm(A)
	c[i] = min(A.shape) / (i + 1) - 1 #i + 1 perchè siamo 0-based
#errore relativo della ricostruzione di A è err_rel[-1]
#Il fattore di compressione è c = c[-1]
plt.figure(figsize=(10, 5))
fig1 = plt.subplot(1, 2, 1)
fig1.plot(err_rel, 'o-')
plt.title('Errore relativo')
fig2 = plt.subplot(1, 2, 2)
fig2.plot(c, 'o-')
plt.title('Fattore di compressione')
plt.show()
plt.figure(figsize=(20, 10))
fig1 = plt.subplot(1, 2, 1)
fig1.imshow(A, cmap='gray')
plt.title('True image')
fig2 = plt.subplot(1, 2, 2)
fig2.imshow(A_k, cmap='gray')
plt.title('Reconstructed image with k =' + str(k_max))
plt.show()