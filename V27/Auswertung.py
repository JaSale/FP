import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats
import scipy.constants as const

print('ROTE LINIE:')
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

np.savetxt('TabelleMagnetfeld.txt', np.column_stack([np.round_(I, 0), np.round_(B, 0)]),
            delimiter=' & ', newline= r' \\'+'\n', fmt='%.2f')



#ROTE LINIE
#Mittelwert bilden
a, b = np.genfromtxt('rot_mitB.txt', unpack=True)
c, deltaS = np.genfromtxt('rot_ohneB.txt', unpack=True)
delS = np.array([])


for i in range(len(b)-1):
    m = (b[i]+b[i+1])/2
    m_fehler = ( ((b[i]-m)**2 + (b[i+1]-m)**2 ) /2)**(1/2)
    #delS = np.append(delS, ufloat(m, m_fehler))
    delS = np.append(delS, m)

delLambdaD = 48.91 #pm -12
delLambda = 0.5*delS/deltaS*delLambdaD #pm
delLambda_mittelwert = np.mean(delLambda) #pm


z=0
for i in range(len(delLambda)):
    z = z + (delLambda[i]-delLambda_mittelwert)**2


delLambda_fehler = (z/(len(delLambda)*(len(delLambda)-1)))**(1/2)
delLambda_mittelwert = ufloat(delLambda_mittelwert, delLambda_fehler)
B= f1(9, *paramsI)

mg_rot = (const.c * const.h)/( (643.8*10**(-9))**2 * 0.5*const.e/const.m_e *const.hbar*B*10**(-3)) * delLambda_mittelwert *10**(-12)
print('delLambda_mittelwert: ', delLambda_mittelwert)
print('B: ', B)
print('mg_rot: ', mg_rot)

#Tabelle
np.savetxt('Tabelle_rot.txt', np.column_stack([deltaS, delS, delLambda]),
            delimiter=' & ', newline= r' \\'+'\n', fmt='%.2f')


#BLAUE LINIE SIGMA
print('BLAUE LINIE SIGMA')
a, b_blau = np.genfromtxt('blau_sigma_mitB.txt', unpack=True)
c, deltaS_blau = np.genfromtxt('blau_sigma_ohneB.txt', unpack=True)
delS_blau = np.array([])

for i in range(len(b_blau)-1):
    m_blau = (b_blau[i]+b_blau[i+1])/2
    #m_blau_fehler = ( ((b_blau[i]-m_blau)**2 + (b_blau[i+1]-m_blau)**2 ) /2)**(1/2)
    #delS = np.append(delS, ufloat(m, m_fehler))
    delS_blau = np.append(delS_blau, m_blau)


delLambdaD_blau = 26.95 #pm
delLambda_blau = 0.5*delS_blau/deltaS_blau*delLambdaD_blau #pm
delLambda_blau_mittelwert = np.mean(delLambda_blau) #pm


z_blau=0
for i in range(len(delLambda_blau)):
    z_blau = z_blau + (delLambda_blau[i]-delLambda_blau_mittelwert)**2


delLambda_blau_fehler = (z_blau/(len(delLambda_blau)*(len(delLambda_blau)-1)))**(1/2)
delLambda_blau_mittelwert = ufloat(delLambda_blau_mittelwert, delLambda_blau_fehler)
B_blau= f1(5.5, *paramsI)

mg_blau = (const.c * const.h)/( (480*10**(-9))**2 * 0.5*const.e/const.m_e *const.hbar*B_blau*10**(-3)) * delLambda_blau_mittelwert *10**(-12)
print('delLambda_blau_mittelwert: ', delLambda_blau_mittelwert)
print('B_blau: ', B_blau)
print('mg_blau: ',mg_blau)

#Tabelle
np.savetxt('Tabelle_blau.txt', np.column_stack([deltaS_blau, delS_blau, delLambda_blau]),
            delimiter=' & ', newline= r' \\'+'\n', fmt='%.2f')


# BLAUE LINIE PI

print('BLAUE LINIE PI')
a, b_b = np.genfromtxt('blau_pi_mitB.txt', unpack=True)
c, deltaS_b = np.genfromtxt('blau_pi_ohneB.txt', unpack=True)
delS_b = np.array([])

for i in range(len(b_b)-1):
    m_b = (b_b[i]+b_b[i+1])/2
    #m_b_fehler = ( ((b_b[i]-m_b)**2 + (b_b[i+1]-m_b)**2 ) /2)**(1/2)
    #delS = np.append(delS, ufloat(m, m_fehler))
    delS_b = np.append(delS_b, m_b)


delLambdaD_b = 26.95 #pm
delLambda_b = 0.5*delS_b/deltaS_b*delLambdaD_b #pm
delLambda_b_mittelwert = np.mean(delLambda_b) #pm


z_b=0
for i in range(len(delLambda_b)):
    z_b = z_b + (delLambda_b[i]-delLambda_b_mittelwert)**2


delLambda_b_fehler = (z_b/(len(delLambda_b)*(len(delLambda_b)-1)))**(1/2)
delLambda_b_mittelwert = ufloat(delLambda_b_mittelwert, delLambda_b_fehler)
B_b= f1(21, *paramsI)

mg_b = (const.c * const.h)/( (480*10**(-9))**2 * 0.5*const.e/const.m_e *const.hbar*B_b*10**(-3)) * delLambda_b_mittelwert *10**(-12)
print('delLambda_b_mittelwert: ', delLambda_b_mittelwert)
print('B_b: ', B_b)
print('mg_b: ',mg_b)

#Tabelle
np.savetxt('Tabelle_b.txt', np.column_stack([deltaS_b, delS_b, delLambda_b]),
            delimiter=' & ', newline= r' \\'+'\n', fmt='%.2f')
