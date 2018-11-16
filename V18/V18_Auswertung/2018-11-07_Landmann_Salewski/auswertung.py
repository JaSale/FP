import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds #Standardabweichungen
from uncertainties import ufloat
from scipy.stats import stats
import sympy
from uncertainties import correlated_values
import scipy.integrate as int
import scipy.constants as con
from scipy.constants import physical_constants as phycon #physikalische konstanten
import sympy
from scipy.signal import find_peaks #peakfinder
from astropy.io import ascii


#Peaks finden.Durchlaufen lassen, mit 13,14 beginnen, weil die so dicht nebneinander liegen.
#Aus plot die Peakhöhe ablesen, bei heights eintragen. Ich hab dann noch mal in der Datei gecheckt, dass
#das wirklich die höchsten in der Umgebung sind.
Eu = np.genfromtxt('152Eu.txt', unpack=True)
peaks = find_peaks(Eu, height=5, distance=10 )
index=peaks[0]
peak_height=peaks[1]
##
##print(index)
##print(peak_height)
##
##linearer fit durch energie gegen bins:
E, W, bin, c = np.genfromtxt('EuEnergy.txt', unpack=True)
ascii.write(
[E, W, bin, Eu[bin.astype('int')]],
'eu.tex', format = 'latex', overwrite='True')

def lin(x, m, n):
    return m*x+n

paramsI, covarianceI= curve_fit(lin, bin, E )
errorsI= np.sqrt(np.diag(covarianceI))
m= ufloat(paramsI[0], errorsI[0])
n= ufloat(paramsI[1], errorsI[1])
#print('Kalibrationsmesswerte:'m,n)

#Ergebnis für Steigung m=0.4015+/-0.0010, Ergebnis für Achsenabschnitt n= 1.1+/-2.1

#Plotte die Messweerte mit markierten Peaks:
#x= np.linspace(1, 8192, 8192)
#plt.plot(x, Eu,'g--',label='Messwerte des Detektors')
#plt.plot(x,Eu[index],'rx',label='Detektierte Peaks')
#plt.xlabel(r"E / keV")
#plt.ylabel(r"counts / $N$")
#plt.tight_layout()
#plt.legend(loc="best")
#
#plt.savefig("Detektormesswerte.pdf")

#Plotte den linearen Fit zur Kalibrierung
x= np.linspace(1, 8192, 8192)
plt.plot(x, lin(x, *paramsI), "b--", label=" Fit")
plt.plot(bin, E, "yx", label="Energie-Bin")
plt.xlim(0, 3900)
plt.xlabel(r"Bin")
plt.ylabel(r"Energie")
plt.legend(loc="best")
plt.savefig("Kalibrierung.pdf")
plt.clf()

#Effizienzberechnung
A_0= ufloat(4130,60)
t_h= ufloat(4943.5, 5) #Tage
t= 18*365+45 #Tage nach 1.10.2000
A=A_0*unp.exp(-unp.log(2)*t/t_h)

print('Aktivität am Versuchstag:',A)
#A=1634+/-24

#Gaussfits über jeden Peak:
def gauss (x, sigma, h, a, mu) :
    return a+h*np.exp(-(x-mu)**2/(2*sigma**2))

def gaussian_fit_peaks(test_ind) :
        peak_inhalt = []
        index_fit = []
        hoehe = []
        unter = []
        sigma = []
        for i in test_ind:
            a=i-20
            b=i+20

            params_gauss, covariance_gauss=curve_fit(gauss, np.arange(a,b+1), Eu[a:b+1],p0=[1,Eu[i],0,i-0.1])
            errors_gauss = np.sqrt(np.diag(covariance_gauss))

            sigma_fit=ufloat(params_gauss[0], errors_gauss[0])
            h_fit=ufloat(params_gauss[1], errors_gauss[1])
            a_fit=ufloat(params_gauss[2], errors_gauss[2])
            mu_fit=ufloat(params_gauss[3], errors_gauss[3])

            print('Z_i', h_fit*sigma_fit*np.sqrt(2*np.pi))

            index_fit.append(mu_fit)
            hoehe.append(h_fit)
            unter.append(a_fit)
            sigma.append(sigma_fit)
            peak_inhalt.append(h_fit*sigma_fit*np.sqrt(2*np.pi))
        return index_fit, peak_inhalt, hoehe, unter, sigma

index_f, peakinhalt, hoehe, unter, sigma = gaussian_fit_peaks(bin.astype('int'))

print('Binindizes ====', bin)
print('mu_i =====',index_f)
print('Peakinhalt =====',peakinhalt)
print('sigma_i =====',sigma)
print('Untergrund =====',unter)
print('Höhe=====', hoehe)

E_det = []
for i in range(len(index_f)):
    E_det.append(lin(index_f[i],*paramsI))

print('Energie ===== ', E_det)

a=7.31+1.5
r=2.25

omega_4pi= (1-a/np.sqrt(a**2+r**2))/2
print('Raumwinkel omega =====', omega_4pi)

Q=[peakinhalt[i]/(omega_4pi*A*W[i]) for i in range(len(W))]
print('Effizienz =====',Q)

ascii.write(
[sigma, index_f, hoehe, unter, peakinhalt ],
'effizienz.tex', format='latex', overwrite='True'
)

ascii.write(
[E_det, Q],
'effizienz2.tex', format='latex', overwrite='True'
)

Q=Q[1:]
index_f = index_f[1:]
peakinhalt = peakinhalt[1:]
W = W[1:]
E=E[1:]
E_det=E_det[1:]

def potenz(x, b, c, d, e):
    return b*(x-c)**d+e

params2, covariance2= curve_fit(potenz, noms(E_det), noms(Q), sigma=stds(Q))
errors2= np.sqrt(np.diag(covariance2))

print('Kalibrationswerte der Potenzfunktion:')
print('Steigung b=', params2[0], '±', errors2[0] )
print('Verschiebung c=', params2[1], '±', errors2[1] )
print('Verschiebung e=', params2[2], '±', errors2[2] )
print('Exponent d=', params2[3], '±', errors2[3] )

#print(noms(Q))

b=ufloat(params2[0],errors2[0])
c=ufloat(params2[1],errors2[1])
d=ufloat(params2[2],errors2[2])
e=ufloat(params2[3],errors2[3])

x=np.linspace(0,1600,10000)
plt.plot(x, potenz(x,*params2), 'g--', label='Energie-Effizienz-Fit')
plt.plot(E, noms(Q),'rx', label='Effizienz-Energie')
plt.legend(loc='best')
plt.xlabel(r'E / keV')
plt.ylabel(r'Q(E)')
plt.savefig('effizienz.pdf')
plt.clf()
