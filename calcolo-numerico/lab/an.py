import numpy as np
import scipy.linalg.decomp_lu as LUdec
import scipy
import matplotlib.pyplot as plt

"""
# Numeri finiti
    #1. Calcolo della precisione macchina in python
base = 2
my_esp = 1
counter = 0
while 1 + my_esp != 1:
    my_old_esp = my_esp
    my_esp = my_esp / base
    counter += 1

print (my_esp)  # Dovrebbe stampare 1.1102230246251565e-16

def epsilon(var_type):
  pow = int(0)
  eps = var_type(1)
  while eps + 1.0 != 1.0:
    eps /= 2
    pow -= 1
  return pow + 1

print(f"La precisione del float è {epsilon(float)}")#numpy.float128
"""



"""

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
    phi = (1.0 + 5 ** 0.5) / 2.0     # ** è l'elevamento a potenza
    return abs(r(k) - phi) / phi

arange = np.arange(50)#1 2 3 4 5 6 7 8 9..
plt.plot(arange, [relative_error(i) for i in arange]) #LAMBDA
plt.legend(['relative error'])
plt.show() #già quando x = 10 la funzione tende a 0
"""



x = np.ones((5,1))
y = np.linspace(1,5,5)
y = y.reshape((1,5))
#print(x.shape)
#print(y.shape)
#print(x)
#print(y)

z= x+y+1
#print(z)

