import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
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
#print('Europiumindex =====>', index)
#print('Europium Peakhoehe =====>',peak_height)
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
print('Kalibrationsmesswerte =====>',m,n)

#Ergebnis für Steigung m=0.4015+/-0.0010, Ergebnis für Achsenabschnitt n= 1.1+/-2.1
#Plotte die Messweerte mit markierten Peaks:
x= np.linspace(1, 8192, 8192)
plt.plot(x, Eu,'r--',label='Messwerte des Detektors')
#plt.plot(lin(index,*paramsI),Eu[index], 'g+',label='Detektierte Peaks')
plt.axvline(309, linestyle=':')
plt.text(309.3,4700,'121,78 keV',rotation=90)
plt.axvline(614, linestyle=':')
plt.text(614.3,4700,'244,70 keV',rotation=90)
plt.axvline(741, linestyle=':')
plt.text(741.3,4700,'295,94 keV',rotation=90)
plt.axvline(861, linestyle=':')
plt.text(861.3,4700,'344,30 keV',rotation=90)
plt.axvline(1027, linestyle=':')
plt.text(1027.3,600,'411,12 keV',rotation=90)
plt.axvline(1108, linestyle=':')
plt.text(1108.3,4700,'443,96 keV',rotation=90)
plt.axvline(1689, linestyle=':')
plt.text(1689.3,600,'678,00 keV',rotation=90)
plt.axvline(1715, linestyle=':')
plt.text(1715.3,4700,'688,67 keV',rotation=90)
plt.axvline(1939, linestyle=':')
plt.text(1939.3,4700,'778,90 keV',rotation=90)
plt.axvline(2159, linestyle=':')
plt.text(2159.3,4700,'964,37 keV',rotation=90)
plt.axvline(2506, linestyle=':')
plt.text(2506.3,4700,'1005,30 keV',rotation=90)
plt.axvline(2702, linestyle=':')
plt.text(2702.3,600,'1085,90 keV',rotation=90)
plt.axvline(2766, linestyle=':')
plt.text(2766.3,4700,'1112,10 keV',rotation=90)
plt.axvline(3228, linestyle=':')
plt.text(3228.3,4700,'1299,10 keV',rotation=90)
plt.axvline(3502, linestyle=':')
plt.text(3502.3,4700,'1408,00 keV',rotation=90)
plt.axvline(3633, linestyle=':')
plt.text(3633.3,4700,'1457,60 keV',rotation=90)
plt.xlim(0,3800)
plt.yscale('log')
plt.xlabel(r"Kanalnumer")
plt.ylabel(r"counts $N$")
plt.tight_layout()
plt.legend(loc="best")

plt.savefig("Detektormesswerte.pdf")
plt.clf()

#Plotte den linearen Fit zur Kalibrierung
x= np.linspace(1, 8192, 8192)
plt.plot(x, lin(x, *paramsI), "r--", label="linearer Fit")
plt.plot(bin, E, "g+", label="zugeordnete Bin-Indizes")
plt.xlim(0, 3900)
plt.ylim(-50, 1700)
plt.xlabel(r"Bin-Index")
plt.ylabel(r"Energie / keV")
plt.legend(loc="best")
plt.savefig("Kalibrierung.pdf")
plt.clf()

#Effizienzberechnung
A_0= ufloat(4130,60)
t_h= ufloat(4943.5, 5) #Tage
t= 18*365+45 #Tage nach 1.10.2000
A=A_0*unp.exp(-unp.log(2)*t/t_h)

print('Aktivität am Versuchstag =====>',A)
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
            a=i-17
            b=i+17

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

#print('Binindizes ====>', bin)
#print('mu_i =====>',index_f)
#print('Peakinhalt =====>',peakinhalt)
#print('sigma_i =====>',sigma)
#print('Untergrund =====>',unter)
#print('Höhe=====>', hoehe)

E_det = []
for i in range(len(index_f)):
    E_det.append(lin(index_f[i],*paramsI))

#print('Energie =====> ', E_det)

a=7.31+1.5
r=2.25

omega_4pi= (1-a/np.sqrt(a**2+r**2))/2
print('Raumwinkel omega =====>', omega_4pi)

Q=[peakinhalt[i]/(omega_4pi*A*(W[i]*0.01)*3851) for i in range(len(W))]

ascii.write(
[sigma, index_f, hoehe, unter ],
'effizienz.tex', format='latex', overwrite='True'
)

ascii.write(
[peakinhalt, E_det, Q],
'effizienz2.tex', format='latex', overwrite='True'
)

Q=Q[1:]
index_f = index_f[1:]
peakinhalt = peakinhalt[1:]
W = W[1:]
E=E[1:]
E_det=E_det[1:]

Q=[peakinhalt[i]/(omega_4pi*A*(W[i]*0.01)*3851) for i in range(len(W))]
print('Effizienz2 =====>',Q)

def potenz(x, a, b):
    return a*(x)**b

params2, covariance2= curve_fit(potenz, noms(E_det), noms(Q), sigma=stds(Q))
errors2= np.sqrt(np.diag(covariance2))

print('Kalibrationswerte der Potenzfunktion:')
print('a=', params2[0], '±', errors2[0] )
print('b=', params2[1], '±', errors2[1] )


print(noms(Q))

a=ufloat(params2[0],errors2[0])
b=ufloat(params2[1],errors2[1])


x= np.linspace(200,1600, 10000)
plt.plot(x, potenz(x,*params2), 'r--', label='Energie-Effizienz-Fit')
plt.plot(E, noms(Q),'gx', label='Messwerte')
plt.legend(loc='best')
plt.xlabel(r'E / keV')
plt.ylabel(r'Q(E)')
plt.savefig('effizienz.pdf')
plt.clf()


################Teil b)##############################################
#
#Cs = np.genfromtxt('137Cs.txt', unpack=True)
#peaks_2= find_peaks(Cs, height=70, distance=20)
#index_2= peaks_2[0]
#peaks_heights= peaks_2[1]
#energie_2=lin(index_2, *paramsI)
##print('Peaks',index_2)
##print('Peakhöhe', peaks_heights)
##print('Energie_2=====', energie_2)
#e_rueck=energie_2[-4]
#e_comp=energie_2[-2]
#e_photo=energie_2[-1]
#
#ascii.write(
#[[index_2[-4], index_2[-2], index_2[-1]], [e_rueck, e_comp, e_photo]],
#'monochromat.tex', format='latex', overwrite='True')
#print('Energie des Photopeaks====', e_photo)
#
########################Vergleich der expermintell ermittelten Werte für die Comptonkante und Rueckstreupeak mit Theoriewerten
#m_e=511
#
#e_comp_th= 2*e_photo**2/(m_e*(1+2*e_photo/m_e))
#print('Theoretischer Wert Comptonkante =====>', e_comp_th)
#print('Vergleich mit exp.-Wert =====>', 1-e_comp/e_comp_th)
#e_rueck_th= e_photo/(1+2*e_photo/m_e)
#print('Theoretischer Wert für den Rückstreupeak =====>', e_rueck_th)
#print('Vergleich mit Theorie =====>', 1-e_rueck/e_rueck_th)
#
#
#
###########################halbwertsbreite#######################################
#links= 1641
#rechts= 1653
#print(Cs[links], Cs[rechts])
#lp = np.arange(links, index_2[-1]+1)
#rp = np.arange(index_2[-1], rechts+1)
#
#params_l, covariance_l= curve_fit(lin, Cs[links:index_2[-1]+1],lp)
#errors_l = np.sqrt(np.diag(covariance_l))
#m_l=ufloat(params_l[0], errors_l[0])
#n_l=ufloat(params_l[1], errors_l[1])
##print('Fitparameter für links Streuung(energie)=====>', 'm_l=', lin(m_l, *paramsI), 'n_l=',lin(n_l, *paramsI) )
#print('Fitparameter für links Streuung=====>', 'm_l=', m_l, 'n_l=', n_l )
#
#params_r, covariance_r= curve_fit(lin, Cs[index_2[-1]:rechts+1],rp)
#errors_r = np.sqrt(np.diag(covariance_r))
#m_r= ufloat(params_r[0], errors_r[0])
#n_r= ufloat(params_r[1], errors_r[1])
##print('Fitparameter für rechts Streuung (Energie)=====>', 'm_r=', lin(m_r, *paramsI), 'n_r=', lin(n_r,*paramsI) )
#print('Fitparameter für rechts Streuung=====>', 'm_r=', m_r, 'n_r=', n_r )
#
#half= m_r*0.5*Cs[index_2[-1]] +n_r  - (m_l*0.5*Cs[index_2[-1]] +n_l)
#ten = m_r*0.1*Cs[index_2[-1]] +n_r  - (m_l*0.1*Cs[index_2[-1]] +n_l)
#
#print('Halbwertsbreite =====>', lin(half, *paramsI))
#print('Zehntelwertsbreite =====>', lin(ten, *paramsI))
#print('Zehntel nach half =====>', lin((1.823*half),*paramsI))
#print('Verhältnis zehntel zu halbwertsbreite =====>', 1 - lin(ten,*paramsI)/lin((1.823*half),*paramsI))

#######################################################################################

#x=np.linspace(1,8192,8192)
#plt.plot(lin(x, *paramsI), Cs, 'r--', label = 'Messwerte des Detektors')
##plt.plot(lin(index_2,*paramsI), Cs[index_2], 'g+', label = 'Detektierte Peaks')
##plt.axhline(y=0.5*Cs[index_2[-1]], color='y', linestyle='dashed')
##print('Halbwertshöhe =====>', 0.5*Cs[index_2[-1]])
##print('Zehntelwertshöhe =====>', 0.1*Cs[index_2[-1]])
##plt.axhline(y=0.1*Cs[index_2[-1]], color='y', linestyle='dashed')
#plt.axvline(195.44, linestyle=':')
#plt.text(196.44, 2000, 'Rückstreupeak', rotation=90)
#plt.axvline(467.87, linestyle=':')
#plt.text(468.87,2000,'Comptonkante', rotation=90)
#plt.axvline(662.35, linestyle=':')
#plt.text(663.35,2000,'Photopeak', rotation=90)
#plt.xlim(0,2000)
#plt.xlabel(r'E / keV')
#plt.yscale('log')
#plt.ylabel(r'counts $N$')
#plt.legend(loc='best')
#plt.savefig('Cs_fit.pdf')
#plt.clf()

################################Inhalt Compton und Photo######################################
#a= index_2[-1].astype('int')-55
#b= index_2[-1].astype('int')+55
#
#params_gauss_b, covariance_gauss_b=curve_fit(gauss, np.arange(a, b+1), Cs[a:b+1], p0=[1, Cs[index_2[-1]], 0, index_2[-1]-0.1])
#errors_gauss_b= np.sqrt(np.diag(covariance_gauss_b))
#
#sigma_fit=ufloat(params_gauss_b[0], errors_gauss_b[0])
#h_fit=ufloat(params_gauss_b[1], errors_gauss_b[1])
#a_fit=ufloat(params_gauss_b[2], errors_gauss_b[2])
#mu_fit=ufloat(params_gauss_b[3], errors_gauss_b[3])
#
#inhalt_photo= h_fit*sigma_fit*np.sqrt(2*np.pi)
#print('Inhalt des Photopeaks =====>', inhalt_photo)

#def compton(E, eps):
#    a_c = Cs[index_2[-2]]/ (1/eps**2 * (2+ e_comp**2 /(e_comp-e_photo)**2*(1/eps**2+(e_photo-e_comp)/e_photo-2/eps*(e_photo - e_comp)/e_photo)))
#    return a_c/eps**2 * (2+ E**2/(E-e_photo)**2*(1/eps**2+(e_photo-E)/e_photo-2/eps*(e_comp-e_photo)/e_photo))
#
#params_comp, covariance_comp= curve_fit(compton, lin(np.arange(1, index_2[-2]+1), *paramsI), Cs[0:index_2[-2]])
#errors_comp= np.sqrt(np.diag(covariance_comp))
#
#eps=ufloat(params_comp[0], errors_comp[0])
#def compton_2(E):
#    eps2 = noms(eps)
#    a_c = Cs[index_2[-2]] / (1/eps2**2 *(2+ e_comp**2/(e_comp-e_photo)**2*(1/eps2**2+(e_photo-e_comp)/e_photo-2/eps2*(e_photo-e_comp)/e_photo)))
#    return a_c/eps2**2 *(2+ E**2/(E-e_photo)**2*(1/eps2**2+(e_photo-E)/e_photo-2/eps2*(e_comp-e_photo)/e_photo))
#print(eps)
#
#inhalt_comp = quad(compton_2, a=lin(0,*paramsI),b=lin(index_2[-2],*paramsI))
#index_compton_kante = index_2[-2]
#lower_index_compton = 52
#inhalt_compton_spektrum = ufloat(sum(Cs[lower_index_compton:index_compton_kante]),sum(np.sqrt(Cs[lower_index_compton:index_compton_kante])))
#
#
#print('Inhalt des Comptonkontinuums =====>', inhalt_compton_spektrum)
#
#print('Vergleich Compton zu Photo Inhalt =====> ', inhalt_compton_spektrum/inhalt_photo)
#
#mu_ph= 0.002
#mu_comp= 0.38
#l=3.9
#abs_prob_photo= 1-np.exp(-mu_ph*l)
#abs_prob_comp= 1-np.exp(-mu_comp*l)
#print('Absorptionswahrscheinlichkeit für den Photoeffekt =====>', abs_prob_photo)
#print('Absorptionswahrscheinlichkeit für den Comptoneffekt =====>', abs_prob_comp)
#print('Vergleich Compton zu Photo = ', abs_prob_comp/abs_prob_photo)


#Teil d), c) gibt es nicht, weil es keinen Positronenstrahler gab

Ba= np.genfromtxt('133Ba.txt', unpack=True)
peaks_3= find_peaks(Ba, height=5, distance= 4)
index_3= peaks_3[0]
peak_heights_3= peaks_3[1]
energie_3= lin(index_3, *paramsI)
#print(index_3)
#print(peak_heights_3)
#print(energie_3)

x=np.linspace(1,8192,8192)
plt.plot(lin(x, *paramsI), Ba,'r--',label='Messwerte des Detektors')
#plt.plot(lin(index_3, *paramsI),Ba[index_3],'g+',label='Detektierte Peaks')

plt.xlim(0,1000)
plt.axvline(55.86, linestyle=':')
plt.text(56.2,3300,'55,86 keV',rotation=90)
plt.axvline(83.56, linestyle=':')
plt.text(84.0,3300,'83,58 keV',rotation=90)
plt.axvline(163, linestyle=':')
plt.text(163.4,3300,'163,00 keV',rotation=90)
plt.axvline(276.4, linestyle=':')
plt.text(277.0,3300,'276,4 keV',rotation=90)
plt.axvline(304.78, linestyle=':')
plt.text(305.2,3300,'304,78 keV',rotation=90)
plt.axvline(357.81, linestyle=':')
plt.text(358.2,3300,'357,81 keV',rotation=90)
plt.axvline(385.62, linestyle=':')
plt.text(386,3300,'385,62 keV',rotation=90)
plt.xlabel(r'E / keV')
plt.ylabel(r'counts $N$')
plt.legend(loc='best')
plt.savefig('Ba_plot_peaks.pdf')
plt.clf()

E_ba, W_ba, bin_ba = np.genfromtxt('BaEnergy.txt', unpack=True)

def gaussian_fit_peaks_d(test_ind) :
    peak_inhalt= []
    index_fit= []
    hoehe= []
    sigma= []
    unter= []
    for i in test_ind:
        a=i-30
        b=i+30

        params_gauss,covariance_gauss= curve_fit(gauss,np.arange(a,b+1),Ba[a:b+1],p0=[1,Ba[i],0,i-0.1])
        errors_gauss= np.sqrt(np.diag(covariance_gauss))

        sigma_fit=ufloat(params_gauss[0], errors_gauss[0])
        h_fit=ufloat(params_gauss[1], errors_gauss[1])
        a_fit=ufloat(params_gauss[2], errors_gauss[2])
        mu_fit=ufloat(params_gauss[3], errors_gauss[3])


        index_fit.append(mu_fit)
        hoehe.append(h_fit)
        unter.append(a_fit)
        sigma.append(sigma_fit)
        peak_inhalt.append(h_fit*sigma_fit*np.sqrt(2*np.pi))
    return index_fit, peak_inhalt, hoehe, unter, sigma

index_ba, peakinhalt_ba, hoehe_ba, unter_ba, sigma_ba = gaussian_fit_peaks_d(bin_ba.astype('int'))

print('Binindizes Ba ====>', bin_ba)
print('mu_i  Ba =====>',index_ba)
print('Peakinhalt Ba =====>',peakinhalt_ba)
print('sigma_i Ba =====>',sigma_ba)
print('Untergrund Ba =====>',unter_ba)
print('Höhe Ba =====>', hoehe_ba)

ascii.write(
[sigma_ba, hoehe_ba, lin(bin_ba, *paramsI), unter_ba],
'BaTab.tex', format='latex', overwrite='True')

E_ba_det = []
for i in range(len(index_ba)):
    E_ba_det.append(lin(index_ba[i], *paramsI))
tba=3790
Q_ba=[0,0,0,0]
Q_ba.append(potenz(E_ba_det[4],*params2))
Q_ba.append(potenz(E_ba_det[5],*params2))
Q_ba.append(potenz(E_ba_det[6],*params2))

sigma=ufloat(1417,23)
A_ba= peakinhalt_ba[3:]/(tba*omega_4pi*W_ba[3:]*0.01*potenz(E_ba_det[3:], *params2))
A_det = [0,0,0,0]
#A_ba_2= sigma/(tba*omega_4pi*7.2*potenz(278.466,*params2))
#print('Aktivität von 276 =========>',A_ba_2)
print('Aktivität vom Rest ==========================>',A_ba)
for i in A_ba:
    A_det.append(i)

ascii.write(
[E_ba, W_ba, bin_ba, E_ba_det],
'BaTab2.tex', format='latex', overwrite='True')

A_mittel= ufloat(np.mean(noms(A_ba)), np.mean(stds(A_ba)))
print('gemittelte Aktivität der Bariumquelle =====>', A_mittel)


#Teil e)
unbe = np.genfromtxt('unbekannt.txt', unpack=True)

peaks_4= find_peaks(unbe, height=100, distance=15)
index_4= peaks_4[0]
peak_heights_4= peaks_4[1]
energie_4= lin(index_4, *paramsI)

ascii.write(
[index_4, unbe[index_4], energie_4],
'unbekanntTab.tex', format='latex', overwrite='True')
print('Energie des unbekannten Strahlers =====>', energie_4)


x= np.linspace(1,8192,8192)
plt.plot(lin(x, *paramsI), unbe, 'r--', label='Messwerte des Detektors')
plt.plot(lin(index_4,*paramsI), unbe[index_4], 'g+', label='Detektierte Peaks')
plt.xlabel(r'E / keV')
plt.ylabel(r'counts $N$')
plt.legend(loc='best')
plt.savefig('unbekannterStrahler.pdf')
plt.clf()

###########################GAUSSFIT########################################
print('------------------------------------------------------------------')

y_photo = np.genfromtxt('137Cs.txt' , unpack=True)
a = np.arange(1,len(y_photo)+1)
a_lin = lin(a, *paramsI)


def gauss_photo (a_lin, sigma, h, a, mu):
    return a+h*np.exp(-(a_lin-mu)**2/(2*sigma**2))

ParamsI_photo, CovarianceI_photo = curve_fit(gauss_photo, a_lin[1640:1656], y_photo[1640:1656], p0=[1,1,1,660])
i = np.linspace(658, 665, 1000)

def funktion_flanken(x, m, b):
    return m*x+b

ParamsI_photo2, CovarianceI_photo2 = curve_fit(funktion_flanken, a_lin[1644:1648], y_photo[1644:1648])
i2= np.linspace(659.7, 661.7, 1000)
errors_photo2= np.sqrt(np.diag(CovarianceI_photo2))
m_l=ufloat(ParamsI_photo2[0], errors_photo2[0])
n_l=ufloat(ParamsI_photo2[1], errors_photo2[1])
print('m_l = ', m_l, 'n_l= ',n_l )

ParamsI_photo3, CovarianceI_photo3 = curve_fit(funktion_flanken, a_lin[1649:1652], y_photo[1649:1652])
i3= np.linspace(662, 663.5, 1000)
errors_photo3= np.sqrt(np.diag(CovarianceI_photo3))
m_r=ufloat(ParamsI_photo3[0], errors_photo3[0])
n_r=ufloat(ParamsI_photo3[1], errors_photo3[1])
print('m_r = ', m_r, 'n_r = ', n_r)

#half= m_r*0.5*Cs[index_2[-1]] +n_r  - (m_l*0.5*Cs[index_2[-1]] + n_l)
#ten = m_r*0.1*Cs[index_2[-1]] +n_r  - (m_l*0.1*Cs[index_2[-1]] + n_l)
#
#print('Halbwertsbreite =====>', lin(half, *paramsI))
#print('Zehntelwertsbreite =====>', lin(ten, *paramsI))
#print('Zehntel nach half =====>', lin((1.823*half),*paramsI))
#print('Verhältnis zehntel zu halbwertsbreite =====>', 1 - lin(ten,*paramsI)/lin((1.823*half),*paramsI))

plt.plot(i, gauss_photo(i, *ParamsI_photo), 'g-', label='Gauß-Fit')
plt.plot(i2, funktion_flanken(i2, *ParamsI_photo2), 'b--')
plt.plot(i3, funktion_flanken(i3, *ParamsI_photo3), 'b--', label='lineare Regression')
plt.plot(a_lin[1640:1656], y_photo[1640:1656],'r+', label='Messwerte')
plt.xlabel(r'E / keV')
plt.ylabel(r'counts $N$')
plt.legend(loc='best')
plt.savefig('Cs_photo.pdf')
