import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import brentq
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats

f, U = np.genfromtxt('1.txt', unpack=True)
U = U*10**(-3)
U_0=8*10**(-3) # volt
V=U/U_0



def f1(x, a, b):
    return x*a+b

ParamsI, CovarianceI = curve_fit(f1, np.log(f[15:]), np.log(V[15:]))
i = np.linspace(10**4, 10**5, 100000)
ErrorsI = np.sqrt(np.diag(CovarianceI))
a = ufloat(ParamsI[0], ErrorsI[0])
b = ufloat(ParamsI[1], ErrorsI[1])

plt.plot(i, f1(i, *ParamsI), 'r-', label='linear regression')
#plt.plot(f, V, 'b.', label='data')
plt.xscale("log")
plt.yscale("log")
plt.savefig('Probe2.pdf')
