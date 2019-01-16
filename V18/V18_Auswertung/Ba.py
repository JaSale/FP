#alle benoetigten pakete laden
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats



y = np.genfromtxt('133Ba.txt' , unpack=True)



L1plot = np.linspace(0, 3000)
plt.title('Plot Barium')
plt.plot(y,'r--')
plt.yscale('log')

plt.savefig('Ba.pdf')
