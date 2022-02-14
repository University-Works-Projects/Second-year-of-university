## 1 - PLOTTING 
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
from skimage import data, metrics

W, H = 50, 60

# Impostare formattazione print

# plt.figure(figsize=(width, height)) - Dimensioni della finestra del grafico
plt.figure(figsize=(W, H))

# plt.title("")
plt.title("TITLE IS HERE")

# plt.{x|y}label("")
plt.xlabel("Label of x axis")
plt.ylabel("Label of y axis")

# plt.plot(A, B, color = '', linestyle = '--')

# plt.subplot(numero_righe, numero_colonne, posizione_figura_corrente)

# Plot disuperfici