import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def machine_precision ():
    esp, counter = 1, 0
    while esp + 1 > 1:
        esp /= 2.0
        counter += 1

    print ("Machine precision: " + esp)
    print ("Iterations: " + counter)

def matplotlib_ex_1 (accuracy = 100, leftRange = -5, rightRange = 5):
    """
        Matplotlib is a plotting library for the Python programming language and its
        numerical mathematics extension NumPy, from https://matplotlib.org/
        Create a figure combining together the cosine and sine curves, from 0 to 10:
        - add a legend
        - add a title
        - change the default colors
        """
    x = np.linspace(leftRange, rightRange, accuracy)
    plt.subplots(constrained_layout = False)[1].secondary_xaxis(0.5);   # Stampa l'asse x
    plt.plot(x, np.cos(x), color = 'red', linestyle = '--');
    plt.plot(x, np.sin(x), color = 'blue', linestyle = '--');

    plt.legend(['sin'], ['cos'])
    plt.title('Sine and cosine from -5 to 5')
    plt.show()



def matplotlib_ex_2 (n):
    """
        Write a script that, given an input number n, computes the numbers of the
        Fibonacci sequence that are less than n.
        """

    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        start, post = 0, 1
        counter = 2
        while start + post < n:
            post += start
            start = post - start
            counter += 1
            #print (start)
        return counter

#matplotlib_ex_2 (int(input("Enter a number: ")))


def matplotlib_ex_2_bis (k):
    """
        Write a code computing, for a natural number k, the ratio r(k)=F(k+1)/F(k),
        where F(k) are the Fibonacci numbers.
        Verify that, for a large k, {{rk}}k converges to the value φ=1+√5
        create a plot of the error (with respect to φ)
        """
    return (matplotlib_ex_2(n + 1) / matplotlib_ex_2 (n))

    arange = np.arange(50)                                                      # lista di 50 valori, da 0 a 49
    plt.plot(arange, [relative_error(i) for i in arange])                       # grafico con i valori arange nelle ascisse 
                                                                                # e l'errore relativo al valore i-esimo di arange calcolato con la funzione r(k)
    plt.legend(['relative error'])
    plt.show()


def r(k): # assuming k > 0
    if k <= 0:
        return 0
    a, b = 0, 1
    for _ in range(k):
        b += a
        a = b - a
    print(b / a)
    return b / a

def relative_error(k):
    phi = (1.0 + 5 ** 0.5) / 2.0
    return abs(r(k) - phi) / phi

"""
questa funzione traccia un grafico nel quale l'asse x è un set di valori da 0 a 49
e l'asse y l'errore relativo di r(k), che calcola phi, quindi, al crescere di k,
r(k) viene approssimato con più precisione, e il relativo 
errore decresce, come si può vedere dal grafico che viene stampato
"""
