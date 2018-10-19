import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats

I, B = np.genfromtxt('Magnetfeld.txt', unpack=True)

L1plot = np.linspace(0, 22.5)
def f1(I, a0, a1, a2, a3):
    return a0+a1*I+a2*I**2+a3*I**3
paramsI, covarianceI = curve_fit(f1, I, B)
errorsI = np.sqrt(np.diag(covarianceI))
a0 = ufloat(paramsI[0], errorsI[0])
a1 = ufloat(paramsI[1], errorsI[1])
a2 = ufloat(paramsI[2], errorsI[2])
a3 = ufloat(paramsI[3], errorsI[3])
plt.plot(L1plot, f1(L1plot, *paramsI) , 'k-', label = "Regression")
plt.plot(I, B, 'b.', label = "Messwerte")
plt.xlabel(r"$ I / \mathrm{A}$")
plt.ylabel(r"$ B / \mathrm{mT}$")
print('a0:', a0)
print('a1:', a1)
print('a2:', a2)
print('a3:', a3)
plt.savefig('Magnetfeld.pdf')
