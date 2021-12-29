'''
"[..]Al contrario degli animali, ognuno dei quali appartiene nella sua perfezione a un tipo
fisso, l'uomo è "l'animale non ancora determinato" e perciò, a causa dell'incertezza
delle sue possibilità e decisioni, egli è già nella sua semplice esistenza una 
malattia della terra.
Ma proprio in tale suo difetto consiste la chance dell'uomo. Egli non è ancora quello che
può essere, non è perfetto, ma ha ancora aperte tutte le possibilità.
Per Nietzsche non è auspicabile che l'uomo diventi un "animale determinato" (un DFA), cioè
un tipo e quindi necessariamente un esemplare del branco.
Al contrario, l'essenza autentica dell'uomo consiste nella sua indeterminatezza (NFA), come
capacità di trascendere se stesso.[..]" ~~~ Nietzsche e il Cristianesimo, pg 100, Jaspers
'''
#LAB 4

#le lambda function prendono in input quello che mettiamo prima dei due punti
#e poi restituiscono quello che mettiamo a destra dei due punti. Così facendo possiamo
#assegnare ad una variabile la nostra funzione
def upper_text(text):  
    return text.upper()

upper_text2 = lambda text: text.upper()
print(upper_text2("Grandemaxi\n"))
x = lambda a, b, c : a + b + c 
print(x(5, 6, 2))  #output -> 13 
#funzioni anonime le quali possono essere passate ad 
#una funzione perchè considerate variabili
def hello(func, text = "Ciao!\n"): #test = "Ciao!\n" sse non viene passato il secondo argomento!
    # storing the function return value in a variable  
    greeting = func(text)  
    print(greeting) 

hello(upper_text2,"hello")
'''
così possiamo creare metodi numerici che hanno come dato di input una funzione
i quali minimizzeranno la funzione o calcoleranno i suoi zeri.
'''
#EX1 - calcolare le radici di una funzione non lineare (=> metodo bisezione, etc..)
'''
metodo delle approssimazioni successive che anzichè andare a calcolarsi lo 0 di f, si
calcola il punto fisso di una funzione g, e il cui schema algoritmico è:

1. Dati: x0
2.k=1
3. Ripeti finchè convergenza
	3.1 xk = g(xk-1)
	3.2 k++
all interno del ciclo devo calcolare l'iterato xk come valore della funzione g in xk-1
il risultato di questo algoritmo iterativo è un'approssimazione della mia soluzione esatta x*
poichè la mia sequenza non la posso calcolare fino all'infinito, la tronco ad un certo punto.
è un metodo iterativo => è interessante com output avere il numero di iterazioni contenuto
nella variabile k che ha impiegato il metodo per arrivare alla soluzione tollerata/approssimata.
=> in output avremo la soluzione x tilde calcolata, il numero di iterazioni. poi altri due vettori
che ci servono per vedere come si avvicina il metodo alla convergenza, quindi come questa successione
xk si avvicina al punto x*

quindi devo costruire una successione di iterati a partire da un iterato x0 segnato uguale a 0
x1, x2, ..., xk che convergono alla soluzione esatta, ovvero alla mia radice dell'equazione
non lineare (al punto in cui f(x)=0).
La f non è la g: nel metodo delle approssimazioni successive si passa dalla risoluzione di
un' eq non lineare al calcolo del punto fisso di un'eq g.

Scrivere una function che implementi il metodo delle approssimazioni successive per il
calcolo dello zero di una funzione f(x) per x∈Rn,
prendendo come input una funzione per l'aggiornamento:
    g(x) = x - f(x)*e^x/2
	g(x) = x - f(x)*e^-x/2
	g(x) = x - f(x)/f'(x)

provare lo schema sulle tre funzioni g
la funzione g è assegnata, viene scelta. Ogni funzione g ti dà luogo ad una successione
di iterati e di velocità di convergenza diversa. Per avere la convergenza ci vogliono certe
condizioni (non approfondito da prof).
Per esempio il metodo di Newton (che è un particolare metodo delle approssimazioni successive
ottenuto prendendo la funzione g(x) = x - f(x)/f'(x)) ha in generale una velocità di convergenza
quadratica (=> p = 2) quindi converge più velocemente alla soluzione rispetto alle altre g => ci
aspettiamo che per la g relativa al metodo di Newton il numero di iterazioni per arrivare alla stessa
precisione sia più basso

Testare il risolutore per risolvere f(x)=e*x - x^2; f(x) = 0
la cui soluzione è x* = −0.7034674
prendendo in input una function e la funzione di punto fisso g, il nostro metodo ci va a 
calcolare iterativamente il punto per il quale la funzione è uguale a 0

I)Disegnare il grafico della funzione f nell’intervallo I = [−1,1] e verificare che x*
sia lo zero di f in [-1, 1]. (mostrare dove si trova x*)

II)Calcolare lo zero della funzione utilizzando entrambe le funzioni precedentemente scritte.
(confrontare i vari metodi utilizzando le differenti g
Confrontare l'accuratezza delle soluzioni trovate e il numero di iterazioni effettuate dai solutori.
Modificare le due funzioni in modo da calcolare l'errore ||xk-x*|| 
ad ogni iterazione k-esima e graficare 
'''

import numpy as np
import matplotlib.pyplot as plt

def succ_app(f, g, tolf, tolx, maxit, xTrue, x0=0):
  i=0
  '''
  err = vettore dove salveremo la diff tra due iterate successive per vedere se
  il nostro metodo inizia ad andare vicino ad un punto e se i vari aggiornamenti
  muovono o meno il punto sul quale stiamo iterando
  '''
  err=np.zeros(maxit+1, dtype=np.float64) 
  err[0]=tolx+1 #tolx = tolleranza (+1 in modo che appena iniziamo il ciclo siamo sicuri che la
  				#condizione relativa all'errore sia verificata
  '''
  vecErrore = vettore dove salveremo la distanza tra la singola iterata e la
  soluzione reale che noi conosciamo a priori
  '''
  vecErrore=np.zeros(maxit+1, dtype=np.float64)
  vecErrore[0] = np.abs(x0-xTrue) #xTrue = soluzione reale, x0 = prima iterata
  x=x0
  #maxit = max numero di iterate; la distanza tra iterati successivi dev'essere sopra
  #una certa tolleranza o il valore assoluto di f(x) dev'essere sopra una certa soglia
  #se scendiamo sotto una certa soglia => la nostra f valutata in x è molto vicina a 0
  #usciamo dal ciclo quando entrambe le condizioni di convergenza sotto sono false (criteri assoluti).
  '''
  Condizioni di convergenza <--> criteri d'arresto
  noi dovremmo trovare un punto per cui f è esattamente uguale a zero, ma richiedere che la
  funzione si annulli è troppo a causa degli errori => richiediamo che sia piccola.
  la distanza di due iterati successivi xk e xk +1 dev'essere significativa (no mille passi piccolissimi).
  '''                #finchè entrambe verificate
  while (i<maxit and (err[i]>tolx or abs(f(x))>tolf) ): # scarto assoluto tra iterati
    x_new=g(x) #x_new = xk
    err[i+1]=abs(x_new-x) #valori assoluti perchè si calcolano distanze
    vecErrore[i+1]=abs(x_new-xTrue) #vettore per plot per vedere come scende velocemente una g rispetto ad un'altra g - problema test
    i=i+1
    x=x_new
  err=err[0:i]      
  vecErrore = vecErrore[0:i]
  return (x, i, err, vecErrore)


f = lambda x: np.exp(x)-x**2 #e^x - x^2 => df/dx = e^x -2*x
df = lambda x: np.exp(x)-2*x #derivata di f

g1 = lambda x: x-f(x)*np.exp(x/2)
g2 = lambda x: x-f(x)*np.exp(-x/2)
g3 = lambda x: x-f(x)/df(x) #metodo di newton - sarà il più veloce

xTrue = -0.7034674
fTrue = f(xTrue)
print('fTrue = ', fTrue)

tolx= 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 0
#variabili definite ed usate sul momento
[sol_n, iter_n, err_n, vecErrore_g1]=succ_app(f, g1, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g1 \n x =',sol_n,'\n iter_new=', iter_n)
[sol_n, iter_n, err_n, vecErrore_g2]=succ_app(f, g2, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g2 \n x =',sol_n,'\n iter_new=', iter_n)
[sol_n, iter_n, err_n, vecErrore_g3]=succ_app(f, g3, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g3 \n x =',sol_n,'\n iter_new=', iter_n)
# GRAFICO Errore vs Iterazioni
# g1
plt.plot(vecErrore_g1, '.-', color='blue')
# g2
plt.plot(vecErrore_g2[:20], '.-', color='green') #scelgo di mostrare fino alla 20esima ascissa (0-based)
# g3
plt.plot(vecErrore_g3, '.-', color='red')

plt.legend( ("g1", "g2", "g3"))
plt.xlabel('iter')
plt.ylabel('errore')
plt.title('Errore vs Iterazioni')
plt.grid()
plt.show() #g3 e g1 convergono (g3 + veloce), g2 oscilla

#########################EX 2#############################
'''
metodo del gradiente per l'ottimizzazione in ℝ2 per trovare l'argmin

il metodo del gradiente risolve il problema di calcolare l'argmin
(l'ascissa del punto di minimo di una funzione f(x) : Rn -> R)
il metodo del gradiente è un metodo iterativo dove mi calcolo una
sequenza di iterati x1, x2, ..., xk partendo da x0 dove xk per k -> oo = x*
= ascissa del minimo di f (xk € Rn => vettori)

In verità questo algoritmo converge ad un punto stazionario (in generale un
minimo locale)

L'iterato xk si calcola xk+1 = xk -alphak*∇f(xk)
alphak calcolato con la procedura di backtracking la quale parte da
un guess (tentativo) di alpha=1, controlla le PROPRIE condizioni di terminazione (che
non c'entrano niente con le condizioni di terminazione dell'algoritmo del gradiente);
se queste condizioni sono verificate l'alpha va bene altrimenti viene dimezzato finchè non si
verificano queste condizioni - entro un numero massimo di iterazioni.
Se la procedura di backtracking termina per il numero massimo di iterazioni significa che non
abbiamo trovato un alpha buono e quindi  l'algoritmo del gradiente si arresta e non possiamo
calcolare il nostro minimo.

while condizioni di arresto negate dell'algoritmo del gradiente:
    while condizioni di arresto negate della procedura di backtracking: #già fornite

riguardo le condizioni di arresto dell'algoritmo del gradiente: il punto di stazionarietà come condizione
necessaria (il fatto che il gradiente sia nullo) => richiederemo un gradiente piccolo. Il gradiente
è un vettore e sui vettori si lavora con la norma

Visualizzazione di funzioni come superfici o linee di livello

Scrivere una funzione che implementi il metodo del gradiente rispettivamente con step size ak
variabile, calcolato secondo la procedura di backtracking ad ogni iterazione k-esima.
Testare la function per minimizzare 𝑓(𝑥) definita come: 𝑓(𝑥)=10(𝑥−1)^2+(𝑦−2)2
In particolare:
    plotta la superficie 𝑓(𝑥) con 𝚙𝚕𝚝.𝚙𝚕𝚘𝚝⎯𝚜𝚞𝚛𝚏𝚊𝚌𝚎() e le curve di livello con 𝚙𝚕𝚝.𝚌𝚘𝚗𝚝𝚘𝚞𝚛()
    plotta, al variare delle iterazioni, la funzione obiettivo, l'errore e la norma del gradiente.
'''
def f(x,y):
    return 10*(x-1)**2 + (y-2)**2


x = np.linspace(-1.5,3.5,100)
y = np.linspace(-1,5,100)
X, Y = np.meshgrid(x, y) #grigliatura del dominio R2 (come se vista dall'alto in R3)
#punti del piano definiti come (x,y,f(x,y))
#mi crea i punti nella griglia per disegnare superfici e curve di livello
'''
5  |
   |
   |
   |
   |----questa intersezione è un punto creato nella griglia 
-1 |    |
    ------------
    -1.5

se avessimo fatto 
x = np.linspace(1,3,2)
y = np.linspace(1,3,2)
X, Y = np.meshgrid(x, y)
allora print(X, Y) mi avrebbe restituito (1,3), (1,1), (3, 1), (3, 3)
(ogni elemento in Y è associato ad un solo elemento in X, e viceversa. Questa associazione forma un punto)
'''
Z=f(X,Y) #valutiamo la nostra funzione su tutti i punti presenti nella grigliatura

plt.figure(figsize=(15, 8))
ax1 = plt.subplot(1, 2, 1, projection='3d') #primo grafico in uno spazio 2x1 //m=1 n=2
ax1.plot_surface(X, Y, Z, cmap='viridis') #colormap
ax1.set_title('Surface plot')
ax1.view_init(elev=45) #il nostro sguardo forma un angolo di 45 gradi con la base del modello

ax2 = plt.subplot(1, 2, 2, projection='3d') #secondo grafico
ax2.plot_surface(X, Y, Z, cmap='viridis')
ax2.set_title('Surface plot from a different view')
ax2.view_init(elev=5)
plt.show()
contours = plt.contour(X, Y, Z, levels=10) #vogliamo mostrare 10 linee di livello 
#10 colori differenti che rappresentano quanto vale la funzione lungo ogni curva di livello
plt.title('Contour plot')
plt.show() #poichè le linee di livello sono degli elissi => funzione convessa con un solo punto di minimo =>globale
# => quando applichiamo il metodo del gradiente il quale converge ad un minimo locale, in questo caso essendo la funzione
#convessa siamo sicuri che converga all'unico minimo che c'è = minimo globale
#queste cose ci servono per rappresentare i passi del gradiente
#fornito
def next_step(x,grad): # backtracking procedure for the choice of the steplength
  alpha=1.1
  rho = 0.5
  c1 = 0.25
  p=-grad
  j=0
  jmax=10
  #condizioni che servono per soddisfare dei criteri di convergenza - condizioni di Wolfe
  while ((f(x[0]+alpha*p[0],x[1]+alpha*p[1]) > f(x[0],x[1])+c1*alpha*grad.T@p) and j<jmax ):
    alpha= rho*alpha #dimezzata
    j+=1
  if (j>jmax):
    return -1
  else:
    #print('alpha=',alpha)
    return alpha #se termina correttamente ci assicura la convergenza del metodo ad un punto stazionario 

#b = vero punto di minimo, step è il passo di backtracking, ABSOLUTE_STOP rappresenta la soglia per cui vogliamo
#il nostro gradiente sopra sotto quella soglia
def minimize(x0,b,mode,step,MAXITERATION,ABSOLUTE_STOP): # funzione che implementa il metodo del gradiente
  #declare x_k and gradient_k vectors
  if mode=='plot_history': #per far salvare tutta la storia delle iterazioni nel processo iterativo
    x=np.zeros((2,MAXITERATION))#salvate per rappresentarle sul grafico delle curve a livello per vedere come scende

  #inizializzazione vettori
  norm_grad_list=np.zeros((1,MAXITERATION))#norma del gradiente la quale deve tendere a 0 perchè andiamo verso
  #un punto stazionario nel quale il gradiente è nullo
  function_eval_list=np.zeros((1,MAXITERATION)) #contiene tutti i valori della funzione. f:Rn->R => numero
  #la funzione obiettivo nel metodo di discesa deve decrescere => facendo il plot del vettore soprastante
  #dovremmo vedere un grafico che decresce decrescono in modo monotono
  error_list=np.zeros((1,MAXITERATION))#norma dell'errore ove b è nella nostra convenzione xTrue => diff tra
  #l'ultimo iterato e la soluzione esatta(=> siamo in un problema test) sempre in norma poichè vettori => numeri che
  #in questo algoritmo decrescono in modo monotono
  
  #initialize first values
  x_last = np.array([x0[0],x0[1]])

  if mode=='plot_history':
    x[:,0] = x_last
  
  k=0

  #prima di iniziare il ciclo while, come nel caso delle approssimazioni successive vengono prima inizializzati
  #dei vettori che servono a tenere delle informazioni per poi fare dei grafici per analizzare l'andamento
  #del metodo 

  #vettori che li inizializziamo fuori dal ciclo per quanto riguarda l'elemento di indice 0
  #salvare ad ogni iterata la valutazione della funzione sull'ultimo punto che abbiamo calcolato
  function_eval_list[:,k]=f(x_last[0], x_last[1]) #0 = x, 1 = y; ":" perchè rimane sempre 1
  error_list[:,k]=np.linalg.norm(x_last-b)#distanza tra l'ultima x calcolata e il minimo reale
  norm_grad_list[:,k]=np.linalg.norm(grad_f(x_last)) #quanto vale il gradiente sull'ultima x calcolata

  while (np.linalg.norm(grad_f(x_last))>ABSOLUTE_STOP and k < MAXITERATION ):
    k=k+1

    #funzione grad_f() la trovi sotto - passi il vettore delle variabili
    grad = grad_f(x_last)#calcolare il gradiente della nostra funzione valutata sull'ultima iterata

    # calcolare quanto vale lo step utilizzando la procedura di backtracking
    step = next_step(x_last,grad) #è il passo di backtracking
    # Fixed step - se fissassimo una lunghezza di passo sarebbe più lento 
    #step = 0.1
    
    if(step==-1):
      print('non convergente')
      return (iteration) #no convergence

    #IMP!!!
    #cuore dell'algoritmo che va a calcolarsi l'iterato successivo
    x_last=x_last-step*grad #aggiorniamo l'ultima x tramite formula


    if mode=='plot_history':
      x[:,k] = x_last #aggiungere x_last all'indice/iterata k-esima

    function_eval_list[:,k]=f(x_last[0], x_last[1])
    error_list[:,k]=np.linalg.norm(x_last-b)
    norm_grad_list[:,k]=np.linalg.norm(grad_f(x_last))

  function_eval_list = function_eval_list[:,:k+1]
  error_list = error_list[:,:k+1]
  norm_grad_list = norm_grad_list[:,:k+1]
  
  print('iterations=',k)
  print('last guess: x=(%f,%f)'%(x[0,k],x[1,k]))
 
  #plots
  if mode=='plot_history':
    v_x0 = np.linspace(-5,5,500)
    v_x1 = np.linspace(-5,5,500)
    x0v,x1v = np.meshgrid(v_x0,v_x1)
    z = f(x0v,x1v)
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(v_x0, v_x1, z,cmap='viridis')
    ax.set_title('Surface plot')
    plt.show()

    # plt.figure(figsize=(8, 5))
    contours = plt.contour(x0v, x1v, z, levels=100)
    plt.plot(x[0,0:k],x[1,0:k],'*')
    #plt.axis([-5,5,-5,5])
    plt.axis ('equal')
    plt.show()
  return (x_last,norm_grad_list, function_eval_list, error_list, k)
b=np.array([1,2]) #argmin esatto

def f(x1,x2):
  res = 10*(x1-1)**2 + (x2-2)**2 
  return res

def grad_f(x):
  return np.array([20*(x[0]-1),2*(x[1]-2)]) #ritorna gradiente

step=0.1 #si potrebbe omettere per come strutturato l'algoritmo
MAXITERATIONS=1000
ABSOLUTE_STOP=1.e-5
mode='plot_history'
x0 = np.array((3,-5))
(x_last,norm_grad_list, function_eval_list, error_list, k)= minimize(x0, b,mode,step,MAXITERATIONS, ABSOLUTE_STOP)
plt.plot(norm_grad_list.T, 'o-')
plt.xlabel('iter')
plt.ylabel('Norma Gradiente')
plt.title('norma del gradiente la quale deve tendere a 0 perchè\nandiamo verso un punto stazionario nel quale il gradiente è nullo')
plt.grid()
plt.show()
plt.plot(error_list.T, 'o-')
plt.xlabel('iter')
plt.ylabel('Errore')
plt.title("lontananza tra l'iterato xkesimo e la soluzione esatta x*, decrescono in modo monotono")
plt.grid()
plt.show()
plt.plot(function_eval_list.T, 'o-')
plt.xlabel('iter')
plt.ylabel('Funzione Obiettivo')
plt.title('contiene tutti i valori della funzione. f:Rn->R => numero\nla funzione (che vogliamo minimizzare =>\npuò essere una funzione obiettivo) nel metodo\ndi discesa deve decrescere')
plt.grid()
plt.show()