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

f_log = np.log(f)
V_log = np.log(U/U_0)

V_0 = np.mean(V_log[:10])

print('V_0:',np.exp(V_0))

# lineare Regression
def f1(x, a, b):
    return x*a+b

ParamsI, CovarianceI = curve_fit(f1, f_log[15:], V_log[15:])
i = np.linspace(11, 14, 10000)
ErrorsI = np.sqrt(np.diag(CovarianceI))
a = ufloat(ParamsI[0], ErrorsI[0])
b = ufloat(ParamsI[1], ErrorsI[1])
print('a:', a)
print('b:', b)
v= unp.exp((np.log(V_0/np.sqrt(2))-b)/a)
print('Frequenz:', v)

plt.plot(f_log, V_log, 'b.', label='data')
plt.plot(i, f1(i, *ParamsI), 'r-', label='linear regression')
plt.xlabel(r"$ln(\nu)$")
plt.ylabel(r"$ln(V')$")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('A1.pdf')
plt.clf()
print('V*v', V_0*v)

plt.plot(f, U/U_0, 'b.')
plt.savefig('Probe.pdf')
