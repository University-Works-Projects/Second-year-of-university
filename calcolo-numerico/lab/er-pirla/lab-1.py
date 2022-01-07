import numpy as np
import scipy.linalg.decomp_lu as LUdec
import scipy
import matplotlib.pyplot as plt


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


