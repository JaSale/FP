#alle benoetigten pakete laden
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats



y = np.genfromtxt('152Eu.txt' , unpack=True)



L1plot = np.linspace(0, 8200)
plt.title('Plot Eu')
plt.plot(y,'r--')
plt.yscale('log')

plt.savefig('Eu.pdf') #erstellt und speichert automatisch den Plot als pdf datei
