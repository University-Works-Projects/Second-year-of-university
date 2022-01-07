## I numeri finiti e il calcolo numerico

### Accuratezza di un risultato numerico

Gli errori sono molto importanti nella decisione se accettare o meno la soluzione di un problema numerico.

Se $A$ è una quantità che vogliamo calcolare e $\tilde{A}$ è un'approssimazione di $A$, allora l'errore commesso è la differenza fra i due valori:
$$
E = A - \tilde{A}
$$
L'**errore assoluto** è il valore assoluto dell'errore:
$$
E_a = | A - \tilde{A} |
$$
L'**errore relativo** si ottiene normalizzando l'errore assoluto con il valore esatto, se $A \neq 0$:
$$
E_r = \frac{|A - \tilde{A}|}{|A|} = \frac{E_a}{|A|}
$$

### Rappresentazione dei numeri

I numeri in memoria vengono rappresentati in forma binaria, cioè nella base 2. L'unità minima del linguaggio digitale è il **Bit** (Binary Digit, cifra binaria): spento (**0**), acceso (**1**).

Ogni numero reale $x \in \R$ può essere rappresentato come $x = \pm (0.d_1d_2d_3..)\beta^p$. Questa viene chiamata **rappresentazione scientifica normalizzata**. La parte frazionaria (mantissa) è sempre minore di 1 e la cifra significativa è sempre diversa da 0. E.g. $7824 = 0.7824 \cdot 10^4$

- La parte frazionaria $0.d_1d_2d_3d_4$ viene chiamata **mantissa** di x (len=4 nell'esempio).
- L'esponente della potenza $p$ viene chiamata **esponente** o caratteristica di $x$ (4 nell'esempio).
- $\beta$ viene chiamata **base** (10 nell'esempio).

### Sistema floating point

Un calcolatore, a causa della sua capacità finita, non è in grado di rappresentare tutto l'insieme $\R$. Abbiamo quindi bisogno di una rappresentazione approssimata di un numero binario. Si tratta di un fatto tecnico, ma con importanti implicazioni nel calcolo numerico.

Tutti i numeri possono essere rappresentati nella notazione scientifica normalizzata, ma non tutti i numeri possono essere memorizzati sul calcolatore. Come faccio a scegliere questo insieme di numeri? Tramite appunto il **sistema floating point** o virgola mobile, perché si tratta di spostare la virgola giocando sull'esponente.

L'insieme dei numeri macchina dipende da **4 parametri**:

- $\beta$ **base di rappresentazione**, sul calcolatore è 2, fissata.
- $t$ **numero di cifre della mantissa**. La maggior parte dei calcolatori ha la possibilità di operare con lunghezze diverse di $t$, a cui corrispondono, ad esempio, la semplice e la doppia precisione (vedi IEEE). Maggiore è il numero $t$ di cifre della mantissa, minore sarà l'ampiezza dei sotto-intervalli e quindi migliore l'approssimazione, ossia la precisione (dal momento che hai più bit per rappresentare un numero).
- $L,U$ gli **estremi di un intervallo a cui appartiene l'esponente** $L \leq P \leq U$. L'esponente è $P$ (numero intero positivo o negativo), anche lui andrà rappresentato. Usualmente $U$ è positivo e $L$ negativo.

$$
F(\beta,t,L,U) = \{0\} \cup \{0.d_1d_2...d_t\ \cdot \beta^P \}
$$

Esempio:
$$
\beta = 2, t = 3, L = -1, U = 2\\
F = \{0\} \cup \{0.100 \cdot 2^p, 0.101 \cdot 2^p, 0.110 \cdot 2^p, 0.111 \cdot 2^p, p = -1,0,1,2 \}
$$
dove $0.100, 0.101, 0.110, 0.111$ sono tutte le possibili mantisse e $p$ i possibili valori dell'esponente. È importante notare come viene utilizzata la rappresentazione normalizzata, sarà l'esponente poi a traslare.

L'insieme dei numeri finiti ha un minimo e un massimo. Si parla di **overflow aritmetico** quando cerco di superare il massimo o il minimo (esponente fuori dal mio range $L,U$). Si parla di **underflow aritmetico** quando cerco di rappresentare un numero tra lo 0 e il numero subito dopo o subito prima: un risultato più piccolo della sensibilità del calcolatore (epsilon di macchina). Dopodiché si può notare che esiste un precedente e un successivo per ogni numero. La distanza fra i numeri non è la stessa, ma quelli vicino allo 0 sono più vicini, e man mano che ci si allontana l'intervallo è sempre più grande. Gli intervalli sono corrispondenti a degli intervalli della base.

I bit disponibili per la memorizzazione di un numero finito vengono suddivisi tra le $t$ cifre della mantissa e l'esponente $p$ che può assumere $U-L+1$ configurazioni diverse.

#### Epsilon di macchina

L'epsilon di macchina $\epsilon$, o precisione di macchina, è il più piccolo numero appartenente a un dato insieme di numeri in virgola mobile $F$, diverso in valore assoluto da zero, che sommato all'unità, da un risultato diverso da 1. Se $a - b < \epsilon$, allora in $F$ abbiamo che $a-b = 0$. Questo viene chiamato fenomeno di cancellazione dei dati. È a sua volta il più grande intervallo che puoi trovare tra due numeri, quindi da un upper bound sull'errore massimo di arrotondamento.

#### Standard IEEE 754

Lo standard IEEE per il calcolo in virgola mobile è lo standard più fisso nel campo del calcolo automatico. Definisce il formato per la rappresentazione dei numeri in virgola mobile ed un set di operazioni effettuabili su questi.

Alcune tipiche rappresentazioni, formati, sono:

- **Precisione singola** $\mathcal{F}(2,24,-128,127)$, **32 bit**. 23 bit per la mantissa, 8 per l'esponente, 1 per il segno ($2^8 = 256 = U-L+1$).
- **Precisione doppia** $\mathcal{F}(2,53,-1024,1023)$, **64 bit**. 52 bit per la mantissa, 11 per l'esponente, 1 per il segno ($2^{11} = 2048 = U-L+1$).

Esistono anche sequenze di bit per rappresentare i valori $+0, -0, +\infty, -\infty$ e quantità simboliche chiamate $NaN$ (not a number). Queste quantità sono usate per gestire "speciali" operazioni", tipo la radice quadrata di un numero negativo, altrimenti non rappresentabili.

#### Aritmetica floating point

Le operazioni eseguite sul calcolatore (calcolo numerico) possono essere eseguite solo con numeri rappresentabili sul calcolatore stesso. Poiché $\mathcal{F}$ (insieme dei numeri macchina) è un sottoinsieme di $\R$, le usuali operazioni aritmetiche sono definite anche per operandi in $\mathcal{F}$, ma il loro risultato, in generale, non sta in $\mathcal{F}$. Il risultato esatto è arrotondato ad un numero che sta in $\mathcal{F}$ (quindi si verifica un errore, detto **errore di arrotondamento**, molto piccolo). E.g. 0.1 e 0.01 non sono rappresentabili in binario, quindi vanno arrotondati.

Il singolo errore è piccolo, ma il problema è la **propagazione degli errori**, dovuto al fatto che il risultato di un'operazione utilizzato come dato di quella successiva può portare a risultati finali molto errati.

Come viene realizzata un'operazione floating point in hardware (con registro con precisione estesa):

1. Eseguo l'operazione esatta $z = x + y$. Di solito questa operazione viene fatta avvalendosi di un registro, non accessibile all'utente, che ha più bit di quelli utilizzati per memorizzare il risultato. Non valgono le proprietà commutativa e associativa.
2. Effettuo l'arrotondamento del calcolo fatto in precisione estesa con il fine di rappresentarlo in memoria. Può essere ottenuto tramite arrotondamento o troncamento della mantissa. In generale se ho che le cifre della mantissa sono $>$ di $t$, commetto un **errore di rappresentazione** che posso maggiorare con l'epsilon di macchina.

Tutte le operazioni hanno un errore più piccolo della precisione di macchina ma nella propagazione degli errori ci sono delle operazioni che possono fare più danni di altri.

- Somma di due numeri che hanno ordine di grandezza molto diversi. Perdita di cifre significative.
- Sottrazione tra due numeri quasi uguali (stesso esponente, mantissa quasi uguale). **Fenomeno cancellazione delle cifre significative**. Può avere effetti catastrofici (catastrophic cancellation) sul risultato finale dell'algoritmo.

### Errori nel calcolo numerico

L'errore che si commette nel calcolo numerico di un risultato ha generalmente diversi sorgenti:

- **Errore di troncamento**: quando un procedimento infinito viene realizzato come procedimento finito (vedi metodi iterativi).
- **Errore algoritmico**: dovuto al propagarsi degli errori di arrotondamento sulle singole operazioni in un procedimento complesso. Legato al concetto di **stabilità** (quando limitato da una costante) e **instabilità** di un algoritmo. Quando l'errore algoritmico è limitato si dice che l'algoritmo è stabile, quando è non limitato si dice che l'algoritmo è instabile. L'errore algoritmico dipende dall'algoritmo, quindi modificando l'algoritmo posso cercare di ridurlo.
- **Errore inerente**: dovuto al fatto che i dati di un problema non sempre appartengono all'insieme $\mathcal{F}$ dei numeri rappresentabili e quindi vengono approssimati. Legato ai dati e al problema che sto risolvendo (non c'entra l'algoritmo), può provenire da dei calcoli precedenti.

## Metodi numerici

Requisiti di un metodo numerico:

- **Esistenza ed eventuale unicità della soluzione**: le condizioni che mi permettono di sapere se la soluzione del mio problema ha soluzione in ambito reale. E.g. le equazioni di secondo grado con il $\Delta < 0$, è inutile che provo a cercare la soluzione con un algoritmo. Se la soluzione esiste sotto certe ipotesi, mi devo mettere sempre sotto queste ipotesi.
- **Metodi numerici**: il metodo deve essere implementabile sul calcolatore e realizzabile come algoritmo. I metodi sono più di uno, ognuno conveniente sotto certe condizioni, ognuno mi da risultati differenti, con metodi di precisioni differenti. Quello più preciso non per forza mi da il risultato migliore (bisogna fare l'**analisi dell'errore**). Bisogna trovare un giusto equilibrio tra precisione e tempo di esecuzione (a trovare l'errore nel caso di metodi iterativi).
- **Condizionamento**: caratteristica legata all'errore inerente.

###  Sistemi Lineari

Un sistema lineare, con matrice associata $A$, ammette una e una sola soluzione nei seguenti casi:

- La matrice $A$ è non singolare, quindi invertibile ($\det A \neq 0$).
- La matrice $A$ ha rango massimo $rg(A) = n$.
- $Ax = 0 \iff x = 0$

### Norme

Una norma è una funzione $|| \cdot || : R^n \implies R$, quindi possiamo dire che assegna una lunghezza a ciascun vettore, ci permette di misurarlo. Le proprietà che deve soddisfare, essendo misure, sono:

- Deve essere $\geq 0$. È uguale a $0$ solo se stiamo trattando il vettore nullo.
- Deve valere la disuguaglianza triangolare.
- Deve valere la linearità rispetto al prodotto per uno scalare.

In generale si misurano degli errori, serviranno anche nel condizionamento. Misuro oggetti (vettori, matrici) o la distanza tra due oggetti e mi fa capire quanto una cosa è vicino all'altra (calcolo l'errore). 

La classe più importante di norme vettoriali è costituita dalle norme $p$.

Le proprietà delle norme matriciali, solo quelle indotte (norma-inf, norma 1, norma 2), oltre alle proprietà generali hanno:

- La norma di un prodotto $\leq$ del prodotto delle norme.
- La norma dell'identità 1.

#### Norme $p$

Avendo $x = (-1,2,3) \in R^n$.

La norma con $p=2$ è detta norma euclidea.
$$
||x||_1 = \sum_{i=1}^n |x_i | = 1+2+3 = 6\\
||x||_\infty = \max_{i} |x_i| = \max (1,2,3) = 3\\
||x||_p = \sqrt{\sum_{i=1}^n |x_i|^p}\\
||x||_2 = \sqrt{1^2 + 2^2 + 3^2} = 14
$$

#### Norma matriciale

La norma 1 è il massimo tra la somma dei coefficienti di ogni colonna. La norma infinito è il massimo tra la somma dei coefficienti di ogni riga.

#### Norma di Frobeneus

### Condizionamento

Il condizionamento riguarda il rapporto tra errore commesso sul risultato di un calcolo e incertezza sui dati in ingresso.

Dato $w = P(v)$, $v$ è l'input e $w$ l'output, io ho che $w$ è calcolato in modo esatto. Se faccio la stessa cosa con $(w + \Delta w) = P(v + \Delta v)$, posso andare ad analizzare come sia la relazione fra la misura $|\Delta w|$ e $|\Delta v|$. Un comportamento buono si ha quando $\Delta w$ è dell'ordine di $\Delta v$.

Un problema è **ben condizionato** se a piccole perturbazioni dei dati corrispondono piccole perturbazioni dei risultati. Al contrario, un problema è **mal condizionato** quando le soluzioni sono molto sensibili a piccole perturbazioni dei dati iniziali.

- Ben condizionato: $\Delta w \sim \Delta v$
- Mal condizionato: $\Delta w \ggg \Delta v$

Il condizionamento è legato all'errore inerente (quindi ai dati), e l'errore inerente è un errore che si ha sul risultato causato dalla rappresentazione di numeri finiti (errore rappresentazione), il numero reale viene rappresentato come numero finito e quindi viene commesso un errore, o comunque da un errore sui dati.

L'errore di rappresentazione è un errore molto piccolo, perché noi sappiamo che se usiamo per esempio i dati in doppia precisione è circa del $10^{-16}$ (epsilon macchina). Partendo da un errore di rappresentazione piccolo, l'errore inerente è l'errore che abbiamo sul risultato.

Devo confrontare l'errore inerente ($E_I$) con l'errore di rappresentazione ($E_R$). $E_I \geq E_R$. Guardo l'ordine di grandezza delle cose.

- Se l'errore inerente è molto più grande dell'errore di rappresentazione $E_I \ggg E_R$, che di solito è dell'ordine di $10^{-16}$, il problema (non l'algoritmo) è mal condizionato.
- Se l'errore inerente è a grandi linee piccolo, tipo $10^{-16}$ (non è definito, è un esempio), il problema è ben condizionato.

##### Numero di condizionamento

La reazione del sistema rispetto alla perturbazione dei dati **dipende solo dalla matrice** (non dal termine noto). Per esempio la matrice di Hilbert è mal condizionata. Al crescere del numero di condizionamento aumenta la sensibilità della soluzione del sistema lineare $Ax = b$ alle perturbazione dei dati. Ovviamente dal momento che mi serve l'inversa, il numero di condizionamento lo posso avere solo per *matrici non singolari*.
$$
K(A) = \Vert A^{-1} \Vert \Vert A \Vert
$$
La relazione fra errori è legato al numero di condizione, poi dipende dall'applicazione se è grande o piccolo. Ma in generale:

- $K(A)$ piccolo quando $n^p$ (dove $n$ è la dimensione del sistema e $p$ la norma) 
- $K(A)$ grande quando $10^n$

Siccome le norme $p$ sono equivalenti, l'ordine di grandezza del numero di condizionamento è lo stesso, e siccome il ruolo del numero di condizionamento è quello di amplificare l'errore sui dati, a noi interessa solo l'ordine di grandezza. Inoltre si dimostra che per tutte le norme p, il numero di condizione è $\geq 1$. Un numero di condizionamento pari a 1 è il condizionamento migliore che possiamo ottenere per un sistema lineare.
$$
K(A) \geq 1
$$

### Metodi diretti

Nei metodi diretti la soluzione viene calcolata in un numero finito di step (passi), **modificando la matrice** del problema in modo da rendere più agevole il calcolo della soluzione. Esempi sono l'ottenimento di **matrici triangolari** attraverso il metodo di sostituzione o il metodo di eliminazione di gauss oppure il metodo di Cholesky (se la matrice è simmetrica e definita positiva).

Per risolvere una matrice triangolare è semplice, abbiamo due metodi:

- **Backward Substitution** (metodo di sostituzione all'indietro). Permette di risolvere un sistema a *matrice triangolare superiore*.
- **Forward Substitution** (metodo di sostituzione all'avanti). Permette di risolvere un sistema a *matrice triangolare inferiore*. Stesse caratteristiche del Backward Substitution.

Complessità computazionale pari a $O(\frac{n^2}{2})$, dal momento che si fanno sempre più moltiplicazioni/divisioni: $1+2+3+...+n = \frac{n\cdot(n+1)}{2}$

Quindi la risoluzione di sistemi triangolari è particolarmente semplice, abbiamo visto che ci conviene arrivare alla risoluzione di un sistema triangolare perché ha una complessità computazionale di $O(\frac{n^2}{2})$. In caso contrario il calcolo di $A^{-1}$ per risolvere $x = A^{-1}b$ ha un costo computazione dell'ordine $O(n^3)$.

Per ricondursi a sistemi equivalenti di forma triangolare, conosciamo bene il **metodo di eliminazione di Gauss**, che ci permette appunto di trasformare il sistema in una matrice triangolare superiore per poi risolverlo eventualmente con una *backward substition*. Però, vedremo che come algoritmo da dare al calcolatore non è l'ideale per una serie di motivi: le sottrazioni dell'eliminazione di Gauss moltiplicato per un opportuno coefficiente potrebbe far verificare il fenomeno della cancellazione delle cifre significative in un calcolatore, quindi algoritmicamente non è stabile e non è neanche facile da estendere a matrici di grandi dimensioni.

#### Fattorizzazione LU

Abbiamo quindi una versione migliore simile all'algoritmo di Gauss per il calcolatore, chiamata **fattorizzazione LU** o decomposizione LU. Risolvendo un sistema lineare con la fattorizzazione LU la maggior parte del costo risiede nella fattorizzazione e non nella soluzione dei sistemi triangolari ottenuti $Ax = LUx = b$, e non c'è bisogno di ripetere tutto il procedimento come farei con l'eliminazione gaussiana. Il costo totale è infatti di:
$$
O (\frac{n^3}{3}) + O(n^2) \simeq O(\frac{n^3}{3})
$$

I metodi di fattorizzazione sono quindi particolarmente convenienti quando si devono risolvere un certo numero di sistemi lineari in cui la matrice dei coefficienti è sempre la stessa e cambia solo il vettore dei termini noti.

**Come funziona?** Consiste nel scambiare le righe in modo da avere nell'elemento del pivot un elemento diverso da 0. Da un punto di vista algebrico significa andare a moltiplicare la matrice per una matrice di permutazione $P$, che **ha come effetto di scambiare le righe**. Da notare che facendo solamente così ci potrebbero essere matrici non singolari per cui non si riesce a calcolare la fattorizzazione, oltre a risultare instabile, per questo motivo si introduce la **strategia del pivoting**: scelgo l'elemento più grande in valore assoluto nella colonna del pivot, in modo che $|l_ik| < 1$. Si applica quindi quando la matrice non è fattorizzabile o se l'errore algoritmico non è limitato. Con la strategia del pivoting, ogni matrice singolare è fattorizzabile.

Abbiamo che $PA = LU$, dove $P$ è una matrice di permutazione, $L$ è una matrice inferiore a diagonale unitaria ($\forall l_{ii} = 1, \forall i$) e $U$ è una matrice triangolare superiore. L'equazione può essere scritta come $Ax = LUx = b$. Nel caso generale, usando la fattorizzazione $LU$ con pivoting ($PA = LU$), il sistema si può risolvere risolvendo:
$$
\begin{cases}
Ly = Pb\\
Ux = y
\end{cases}
$$

#### Fattorizzazione di Cholesky

Ogni matrice simmetrica ($\lambda \in \R$) e definita positiva (tutti gli autovalori $\lambda > 0$) può essere fattorizza nel prodotto di matrici triangolari $A = L \cdot L^T$. Se $\lambda \leq 0$, allora la matrice non è definita positiva. Costo: $O(n^3/6)$.

### Metodi iterativi

I metodi iterativi non sono addetti solo alla risoluzione di sistemi lineari, ma anche alla minimizzazione di funzioni che è legata alla risoluzione di equazioni singole non lineari.

I metodi iterativi sono basati su questa idea: io ho il mio sistema $Ax = b$ e mi costruisco una successione di approssimazioni alla soluzione $\{x_0,x_1,x_2,...,x_k\}$ con la proprietà che questa successione $x_k$ converge per k che tende all'infinito, con $x^*$ che rappresenta la soluzione esatta del mio sistema lineare. Converge elemento per elemento cercando di arrivare alla soluzione esatta.

Come faccio a calcolare questa successione? A partire da $x_0$ mi calcolo i valori successivi $x_k$ come valori di una certa funzione. Più vado avanti più la soluzione si avvicina a $x^*$, ma se vado molto avanti il metodo può costare tanto. Serve un compromesso tra precisione e tempo d'esecuzione. $G$ è la mia funzione che caratterizza i diversi metodi.
$$
x_k = G(x_{k-1})
$$
Il ciclo (iterativo) che mi calcola $x_k = G(x_{k-1})$ deve fermarsi ad un certo punto, non posso di certo "ripetere finché convergenza", farei qualcosa di infinito. E qui entra in gioco l'**errore di troncamento**, dovuto ad un troncamento di un procedimento infinito in un procedimento finito.

```pseudocode
Dati: X_0
k=1
while (convergenza)
	x_k = G(x_{k-1})
	k = k+1
```

La complessità computazione è pari a quella di un prodotto matrice vettore per iterazione.

Se cambio il vettore iniziale $x_0$ ottengo delle convergenze diverse e può succedere che alcune successioni siano convergente, altre non lo siano. Un metodo iterativo per la risoluzione di sistemi lineari è **convergente** se qualunque sia il vettore iniziale $x_0$, la successione $x_k$ è convergente. Solo se il raggio spettrale della matrice di iterazione è minore di 1 $\rho (T) < 1$.

Affinché abbia senso bisogna dimostrare che la successione $x_k$ è convergente e che la successione per $k$ che tende all'infinito converge a $x*$ (tale che $x* = A^{-1}b$).

#### Ordine di convergenza

L'ordine di convergenza o velocità di convergenza mi permette di discriminare un metodo rispetto all'altro. Mi dice **quanto velocemente decresce l'errore di troncamento** dal passo $k$ al passo $k+1$.

Convergere più velocemente non significa che ci metto meno tempo per arrivare all'errore, perché dipende dal costo computazionale del mio algoritmo. Ma considerando che i metodi iterativi per la risoluzione di sistemi lineari hanno un costo computazionale simile, dato dal prodotto matrice vettore $O(n^2)$, l'ordine di convergenza diventa un buon punto di riferimento per la scelta del mio metodo.

Il tempo totale è dato dal tempo per iterazione per il numero di iterazioni:
$$
T_{tot} = T_{it} \cdot K_{it} = O(n^2) \cdot K_{it}
$$

#### Criteri di convergenza

Sono le condizioni che il ciclo di un metodo iterativo deve avere. Oltre alla condizione di convergenza, con un'eventuale tolleranza, è necessario aggiungere un controllo di upper bound $k \leq \text{maxit}$ per il numero di iterazioni per evitare cicli infiniti.

#### Metodi iterativi stazionari

Che si riassume in: prodotto di una matrice $H \in R^{n \times n}$ (detta **matrice di iterazione del metodo**) per l'iterato $x_{k-1}$ + un vettore $d \in \R^n$:
$$
x_k = Hx_{k-1} + d\\
x_{k+1} = Hx_k + d
$$
Abbiamo:

- Metodo di Jacobi (metodo delle sostituzioni simultanee)
- Metodo di Gauss Seidel (metodo delle sostituzioni successive)
- Metodi di Rilassamento (SOR, SSOR)

##### Metodo di Jacobi

Detto anche metodo delle sostituzioni simultanee. Consideriamo il seguente splitting di una matrice $A$:
$$
A = \begin{pmatrix}
& & F\\
& D &\\
E & &
\end{pmatrix}
$$
dove $D$ è la diagonale di $A$, $E$ e $F$ sono la parte sotto/sopra, rispettivamente, sopra la diagonale.

Per matrici con diagonale dominante in senso stretto, oppure irriducibile con diagonale dominante, il metodo di Jacobi converge.

##### Metodo di Gauss-Seidel

Per matrici con diagonale dominante in senso stretto, oppure irriducibile con diagonale dominante, il metodo di Jacobi converge. Per matrici simmetriche hermitiane non singolare definite positive, il metodo di Gauss-Seidel converge.

#### Metodi iterativi non stazionari

Non ci sono matrici, ma solo vettori.

- Metodo Gradienti Coniugati
- Metodo GMRES
- Metodi di Krylov

### Differenza: Metodi Diretti e Iterativi

|                  | Metodi Diretti | Metodi Iterativi                        |
| ---------------- | -------------- | --------------------------------------- |
| Precisione       | Yes            | No (per via dell'errore di troncamento) |
| Gestione Tempo   | No             | Yes                                     |
| Gestione Memoria | No             | Yes                                     |

I metodi diretti sono più idonei per matrici di piccole dimensioni o dense. I metodi iterativi per matrici di grandi dimensioni: **lasciano inalterata la matrice** (al contrario dei metodi diretti) e permettono di maneggiare i coefficienti del sistema in maniera più diretta, operazioni fra array che sono prodotto scalari e prodotto matrice vettore.

Per esempio una matrice sparsa (pochi elementi diversi da 0, tipo 5 per 1000), si adatta meglio nei metodi iterativi:

- Memorizzo della matrice solo gli elementi diversi da 0 e i loro indici per risparmiare sulla gestione della memoria.
- Anziché fare il prodotto per ogni elemento della matrice, lo faccio solo per gli elementi diversi da 0.
- Il tempo per iterazioni diventa $T_{it} = O(m) \ll O(n^2)$

## Interpolazione

Un metodo per individuare **nuovi punti** del piano cartesiano a partire da un **insieme finito di punti dati** $(x_k, y_k)$, nell'ipotesi che tutti i punti si possano **riferire ad una funzione** $f(x)$ di una data famiglia di funzioni di una variabile reale. In poche parole voglio costruire la funzione $f$, voglio che $f$ nei nodi $x_k$ sia uguale a $y_k$.
$$
f(x_k) = y_k \text{ per } k = 1 \ldots n
$$
Questa tecnica viene utilizzata in due casi:

- Quando non si conosce l'espressione di $f(x)$, ma si conosce il valore in determinati nodi.
- Conosciamo $f(x)$ ma è troppo complicata, conviene quindi crearsi i nodi e applicare l'interpolazione.

Disponiamo di due modi per calcolare la funzione:

- **Funzione interpolante**: dove impongo di passare esattamente per i punti. Si usa di solito quando ho dei dati che sono esatti (non si intende l'errore, quello c'è sempre), ma che non derivano da strumenti di misura o calcoli precedenti.
- **Funzione approssimante**: dove non passo esattamente per i punti ma mi ci avvicino. Si usa di solito quando si sta lavorando con dei dati affetti da diversi tipi di rumore.

### Interpolazione polinomiale

L'interpolazione di una **serie di valori** (ad esempio dei dati sperimentali) con una **funzione polinomiale** che passa per i punti dati. In particolare, un qualsiasi insieme di $n+1$ punti distinti può essere sempre interpolato da un polinomio di grado $n$ che assume esattamente il valore dato in corrispondenza dei punti iniziali.

Dati $n+1$ punti $(x_i, y_i)$ distinti, dove $y = f(x)$, dobbiamo determinare il polinomio di grado minimo che passa per i punti assegnati:
$$
(x_i, y_i), i = 0,1,...,n \quad y_i = f(x_i)
$$

Si può intuitivamente pensare che più punti si hanno, più l'approssimazione migliori. Ma non è così, l'errore addirittura in alcune funzioni tende all'infinito. È controintuitiva come cosa, ma è così, più aumenti il grado del polinomio, più la funzione tende ad oscillare.

Si utilizza quella che si chiama **interpolazione polinomiale a tratti** per ovviare a questo problema.

#### Teorema di unicità del polinomio interpolatore

Esiste uno ed un sol polinomio di grado $n$ che assume valori $y_i, i = 0,1,...,n$ in corrispondenza di $n+1$ punti distinti $x_i, i = 0,1,...,n$.
$$
P_n(x) = \alpha_0 + \alpha_1 x + ... \alpha_n x^n
$$

#### Interpolazione nella forma di Lagrange

Un particolare tipo di interpolazione polinomiale.

#### Interpolazione rispetto ai nodi di Chebyshev

Utilizzando questi punti si ha un diverso comportamento del polinomio di interpolazione. Aumentando il numero dei nodi, in questo caso l'errore di interpolazione tenderà a 0.

#### Fenomeno di Runge

Il fenomeno di Runge è un problema relativo all'interpolazione polinomiale su nodi equispaziati con polinomi di grado elevato. Esso consiste nell'aumento di ampiezza dell'errore in prossimità degli estremi dell'intervallo. Utilizzando i nodi di Chebyshev si può risolvere il problema. Funzione di Runge:
$$
f(x) = \frac{1}{1+25x^2}
$$

### Approssimante

Nel caso in cui ho dei punti e la funzione che vado a calcolare non passa esattamente per i punti, non impongo una condizione di passaggio.

Se i punti sono affetti da errore di misura, non ha senso chiedere il passaggio per i punti, perché altrimenti è come se imponessi la funzione che vado a calcolare un errore. Se richiedo passaggio per i punti, se voglio calcolare una funzione, e.g. polinomio, il polinomio di grado n è molto oscillante. Se invece non richiedo passaggio per i punti, con un polinomio più semplice con grado più basso $p(x) = \alpha_0 + \alpha_1 x + \ldots + a_nx^n$ l'oscillazione è più bassa.

Il passare vicino i punti viene espresso calcolando una misura. Vado a misurare la distanza fra il **valore della misura** e il **valore della funzione approssimante**. Quindi richiedere che il polinomio passi vicino ai punti, significa richiedere che la distanza fra il polinomio e i punti sia piccolo. Come misuro la distanza? Con la norma 2 di questo vettore, chiamato vettore dei residui, facendo la differenza tra il valore del polinomio e il dato che possiedo. Il vettore dei residui è la mia distanza e misuro la distanza a norma 2, e scelgo il più piccolo (min) dei residui. Problema di minimizzazione.

Sia $A$ una matrice $m \times n$ con $m > n$ e $rg(A) = k \leq n$, il problema ai minimi quadrati ammette sempre almeno una soluzione. Se $k=n$ il problema ha una ed una sola soluzione (e si risolve con il metodo delle equazioni normali), se $k < n$  il problema ha infinite soluzioni (e si risolve con la decomposizione SVD).

#### Metodo dei minimi quadrati (OLS)

È un problema frequentissimo nelle applicazioni. È il problema base di moltissimi algoritmi di machine learning.

Può essere risolto con il **metodo delle equazioni normali**, osservando che il problema di minimo $\min_{\alpha} = || A \alpha - y ||_2^2$ può essere riscritto come questo sistema lineare:
$$
A^T A \alpha = A^T y
$$
Risolvendo questo sistema di lineare con Cholesky essendo simmetrica definita positiva oppure con metodi iterativi (e.g. Gauss-Seidel), si ottiene il vettore degli $\alpha$ che corrisponde ai coefficienti del polinomio approssimante.

##### Decomposizione ai valori singolari (SVD)

Un altro modo per risolvere un problema ai minimi quadrati. Dato il problema ai minimi quadrati
$$
\min_{\alpha} = || A \alpha - y ||_2^2
$$



## Risoluzione di equazioni non lineari

Risoluzione di equazioni non lineari o anche conosciuto come calcolo di uno zero di una funzione. Risolvere una equazione non lineare significa calcolare i punti in cui questa funzione si annulla.

Calcolare uno zero (o radice) di una funzione $f(x)$ a dominio reale e valori reali, significa trovare l'elemento $x$ nel dominio di $f$ tale che $f(x) = 0$.

I metodi per questo tipo di problema sono metodi iterativi. Quando parliamo di metodi iterativi per risolvere equazioni non lineari, parliamo di convergenza a partire da un iterato iniziale, distinguiamo:

- **convergenza globale**: si ha quando il metodo converge per ogni scelta di $x_0$, nel dominio che stiamo considerando.
- **convergenza locale**: si ha quando il metodo converge solo se $x_0$ è vicino alla soluzione esatta. Quindi non possiamo prendere il punto iniziale a piacere, perché il metodo può non convergere. Bisogna scegliere iterato iniziale vicino alla soluzione esatta.

### Metodo della bisezione

Il metodo di bisezione è il metodo numerico più semplice per trovare le **radici di una funzione**. La sua efficienza è scarsa e presenta lo svantaggio di richiedere ipotesi particolarmente restrittive. Ha però il notevole pregio di essere **stabile** in ogni occasione e quindi di garantire sempre la buona riuscita dell'operazione. Questo metodo può essere in parte paragonato all'algoritmo di ricerca binaria dell'informatica, per la ricerca di un determinato valore all'interno di un insieme ordinato di dati. La ricerca binaria ha tuttavia andamento logaritmico, cioè molto più rapido, perché opera su interi (le posizioni degli elementi nell'insieme).

Data l'equazione $f(x) = 0$ definita e continua in un intervallo $[a,b]$, tale che $f(a) \cdot f(b) < 0$, è allora possibile calcolarne una approssimazione in $[a,b]$.

Si procede dividendo l'intervallo in due parti uguali e calcolando il valore della funzione nel punto medio di ascissa $\frac{a+b}{2}$. Se risulta $f(\frac{a+b}{2}) = 0$ allora abbiamo trovato la radice. Altrimenti tra i due intervalli $[a, \frac{a+b}{2}]$ e $[\frac{a+b}{2}, b]$ si sceglie quello ai cui estremi la funzione assume valori di segno opposto. Così continuando si ottiene una successione di intervalli $[a_1, b_1], [a_2, b_2], \ldots, [a_n, b_n]$ incapsulati, cioè ognuno incluso nel precedente. L'algoritmo può non avere fine, quindi c'è il solito ciclo delle massime iterazioni e della tolleranza ammessa (epsilon).

```python

```



### Metodo delle approssimazioni successive



### Metodo di Newton

Si applica dopo avere determinato un intervallo $[a,b]$ che contiene una sola radice.

## Ottimizzazione non vincolata

Ottimizzare significa trovare la migliore possibilità secondo un determinato criterio. Questo migliore criterio viene tradotto matematicamente nel trovare il minimo o il massimo di una funzione. Esempi sono, ricerca del percorso minimo, ricerca del tempo minimo. Il machine learning è basato sulla minimizzazione di funzioni, chiamate funzioni di costo, che rappresentano il costo di una determinata funzione.

Ottimizzazione di una funzione $f$ di $n$ variabili su un insieme $\R^n$. La $f$ prende il nome di **funzione obiettivo**, l'insieme $\R^n$ insieme o **regione ammissibile**. Ciascun punto $x = (x_1,x_2,...,x_n)^T \in X$ costituisce una **soluzione ammissibile**. Determinare un punto $x* \in \R^n$ tale da rendere minima la funzione $f$. Il problema può indicarsi in generale così:
$$
\min_{x \in \R^n} f(x)
$$
Un vettore $x^*$ è un **punto di minimo globale** di $f(x)$ se
$$
f(x^*) \leq f(x) \quad \forall x \in \R^n
$$
Un vettore $x^*$ è un **punto di minimo locale** di $f(x)$ se esiste un intorno circolare $I(x*, \epsilon)$, avente raggio $\epsilon > 0$ tale che
$$
f(x*) \leq f(x) \quad \forall x \in \R^n \cap I(x^*, \epsilon)
$$
Un vettore $x^*$ di minimo (globale o locale) si dice **stretto** se $f(x^*) < f(x)$, quindi $\forall x \in \R^n, x \neq x^*$.

Una funzione $f(x)$ può avere un punto di minimo locale e tuttavia non avere un punto di minimo globale. Inoltre, può non avere né minimi locali né globali, può avere sia minimi locali che globali..

I problema di determinare punti di minimo globale è molto difficile e gli algoritmi analizzati determinano punti di minimo locale. Talvolta, si dispone di conoscenze aggiuntive sulla funzione obiettivo che permettono di identificare un punto di minimo globale. **È il caso delle funzioni convesse, per le quali i punti di minimo locale sono anche punti di minimo globale**.

### Metodi di discesa

Trovare la **direzione** lungo la quale la funzione diminuisce il suo valore (direzione di discesa) e poi decidere quanto lontano muoversi lungo questa direzione (lunghezza del passo, step-length):
$$
x_{k+1} = x_k + \alpha_k p_k
$$
$p_k$ è una **direzione di ricerca**, $\alpha_k$ è uno scalare chiamato **lunghezza del passo**. Questi due parametri sono scelti in modo da garantire la decrescita di $f(x)$ ad ogni iterazione:
$$
f(x_{k+1}) < f(x_k) \quad k= 0,1,2,...
$$

#### Metodi di tipo gradiente

Il gradiente rappresenta la direzione di massima crescita (o decrescita, se con segno negativo) della funzione:
$$
p_k = - \nabla f(x_k)
$$

Il gradiente indica la direzione della salita più rapida, e di conseguenza il gradiente negato indica la discesa. La funzione cresce più velocemente lungo la direzione del gradiente, decresce più velocemente lungo meno gradiente (direzione dell'anti gradiente).

Il fatto di scegliere la direzione in discesa non garantisce che l'algoritmo converga. Dove mi fermo?

Più il passo è piccolo, più ho garanzie che il passo almeno decresca. Ma se lo prendo molto piccolo, la minimizzazione può essere molto molto lenta. Non è detto che in quel punto la funzione decresca. Per un po' la funzione decresce, ma dopo un po' può tornare a crescere.

##### Condizioni di Wolfe

Cercano di risolvere il problema che se si sceglie un passo preciso, unicamente determinato dal requisito $f(x_{k + 1} < f(x_k)$, la funzione $x_{k+1}$ rimane in un punto di stallo.

Ci serve $\alpha_k$ che soddisfa le seguenti condizioni

##### Prima condizione di Wolfe (o condizione di Armijo)

Limita inferiormente la diminuzione ottenuta ad ogni passaggio
$$
f (x_{k+1}) \leq f(x_k) - \alpha_k \gamma
$$

##### Seconda condizione di Wolfe

Mi assicura che non vengono fatti passi troppo corti.

##### L'algoritmo di BackTracking

Si mettono insieme le due condizioni di Wolfe. La prima condizione da sola non è sufficiente per garantire la lunghezza del passo, perché la lunghezza del passo può diventare troppo piccola. Ma se uso una tecnica euristica, ricorsiva, per cui diminuisco il passo ma a un certo punto mi fermo in modo che non sia troppo corto, allora riesco effettivamente a mettere insieme le due cose.

La tecnica consiste, nel porre $\alpha_0 = 1$. Riduco questo valore finché non si trova un valore $\alpha_k$ per cui non sia soddisfatta la condizione di Armijo.

Solitamente $\rho$ si dimezza sempre, funziona così il backtracking. Per evitare che il passo sia troppo piccolo si mette una seconda condizione, di tollerabilità.

**Proposizione**

Sotto "opportune" ipotesi sulla funzione $f$ il metodo di discesa che calcola gli iterati come $x_{k+1} = x_k + \alpha_k p_k$ con $p_k$ direzione di discesa, $\alpha_k$ calcolato con algoritmo di backtracking (nel caso in cui sono soddisfate la prima condizione di wolfe) converge ad un punto di minimo locale $x^*$ di $f(x)$ tale che $\nabla f (x) = 0$

