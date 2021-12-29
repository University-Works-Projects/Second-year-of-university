import numpy as np
import scipy.linalg.decomp_lu as LUdec
import scipy
import matplotlib.pyplot as plt

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

