import matplotlib.pyplot as plt
import numpy as np

leftRange = -10
rigthRange = 30
accuracy = 100

x = np.linspace(leftRange, rigthRange, num = accuracy)
y_exp = np.exp(x)

plt.plot(x, y_exp, color = 'red', linestyle = '--')

plt.legend(['exp'])

plt.show()
