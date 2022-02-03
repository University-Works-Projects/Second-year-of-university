## ? - IMMAGINI 
 
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
from skimage import data, metrics

# Utilities
plt.imread('imageName,png')                                          # Caricamenti immagine da file
#C, R = np.shape(IMG) # Dimensione dell'immagine

# Caricamento immainge da dataset
from skimage import data                                            # set di immagini precaricate e salvate come matrici
A = data.camera()                                                   # Camera-man

plt.imshow(A, cmap = 'gray')                                        # Scala di grigio
