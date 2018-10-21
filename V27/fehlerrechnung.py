import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats

a, b = np.genfromtxt('rot_mitB.txt', unpack=True)
c, d = np.genfromtxt('rot_ohneB.txt', unpack=True)
n=0
b(b[n]-b[n+1])
