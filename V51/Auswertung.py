import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import brentq
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats


########### A1 ################################################################
f1, A1 = np.genfromtxt('1.txt', unpack=True)
f1_log = np.log(f1)
A1_log = np.log(A1)

def f_1(x, a1, b1):
    return x*a1+b1

ParamsI, CovarianceI = curve_fit(f_1, f1_log[16:], A1_log[16:])
i1 = np.linspace(11, 14, 1000000)
ErrorsI = np.sqrt(np.diag(CovarianceI))
a1 = ufloat(ParamsI[0], ErrorsI[0])
b1 = ufloat(ParamsI[1], ErrorsI[1])
null1 = brentq(f_1, 0, 100, args=(ParamsI[0], ParamsI[1]))

plt.plot(i1, f_1(i1, *ParamsI), 'b-', label='linear regression')
plt.plot(f1_log, A1_log, 'r.', label='data')
plt.xlabel(r"$\mathrm{ln}(f)$")
plt.ylabel(r"$\mathrm{ln}(U)$")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('1.pdf')
plt.clf()
print('    - A1 -    ')
print('a1: ', a1)
print('b1: ', b1)
print('Nullstelle 1: ', np.exp(null1))
R_N1=50*10**3
R_11=10*10**3
V1 = -R_N1/R_11
print('V1= ', V1)
print('V1*v1= ', V1*np.exp(null1))
########### A2 ################################################################

f2, A2 = np.genfromtxt('2.txt', unpack=True)
f2_log = np.log(f2)
A2_log = np.log(A2)

def f_2(x, a2, b2):
    return x*a2+b2

ParamsII, CovarianceII = curve_fit(f_2, f2_log[8:], A2_log[8:])
i2 = np.linspace(9, 13, 10000)
ErrorsII = np.sqrt(np.diag(CovarianceII))
a2 = ufloat(ParamsII[0], ErrorsII[0])
b2 = ufloat(ParamsII[1], ErrorsII[1])
null2 = brentq(f_2, 0, 100, args=(ParamsII[0], ParamsII[1]))
plt.plot(i2, f_2(i2, *ParamsII), 'b-', label='linear regression')
plt.plot(f2_log, A2_log, 'r.', label='data')
plt.xlabel(r"$\mathrm{ln}(f)$")
plt.ylabel(r"$\mathrm{ln}(U)$")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('2.pdf')
plt.clf()
print('    - A2 -    ')
print('a2: ', a2)
print('b2: ', b2)
print('Nullstelle 2: ', np.exp(null2))
R_N2=100*10**3
R_12=1*10**3
V2 = -R_N2/R_12
print('V2= ', V2)
print('V2*v2= ', V2*np.exp(null2))
print('Verhältnis:', V1*np.exp(null1)/(V2*np.exp(null2)))


### Integrator###############################################################

f3, A3 = np.genfromtxt('3.txt', unpack=True)
w3 = 2*np.pi*f3 *10**(-3) #1/w => millisekunden
U_0=0.59
U=A3/U_0
def f_3(x, a3, b3):
    return a3*x+b3

ParamsIII, CovarianceIII = curve_fit(f_3, 1/w3, U)
i3 = np.linspace(0, 0.3, 10000)
ErrorsIII = np.sqrt(np.diag(CovarianceIII))
a3 = ufloat(ParamsIII[0], ErrorsIII[0])
b3 = ufloat(ParamsIII[1], ErrorsIII[1])

plt.plot(i3, f_3(i3, *ParamsIII), 'b-', label='linear regression')
plt.plot(1/w3, U, 'r.', label='data')

plt.xlabel(r"$\frac{1}{\omega}  / \mathrm{ms}$")
plt.ylabel(r"$\frac{U}{U_0}$")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('3.pdf')
plt.clf()
a3=a3*10**3
print('    - A3 -    ')
print('a3: ', a3)
print('b3: ', b3)
R= 476.8
C= 12.9*10**(-9)
print('1/RC_Theorie : ', 1/(R*C))
print('Abweichung: ', 1/(R*C)/(a3))
