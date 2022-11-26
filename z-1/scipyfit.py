import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def line(x, m, c):
    return c+m*x

v='Vout'
y, err_y, x, err_x= np.loadtxt('newer/'+v+".txt", unpack=True)
param, param_cov = curve_fit(line, x, y, absolute_sigma=True)
perr = np.sqrt(np.diag(param_cov))
y_result = param[0]*x+param[1]
print("parameters: ",param)
print("covariance of parameters: ",param_cov)
print('error: ',perr)
plt.plot(x, y, 'o', label ="data")
plt.plot(x, y_result, '--', label ="fit")
plt.legend()
plt.grid()
plt.show()

