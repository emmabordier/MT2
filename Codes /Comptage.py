
#@authors: Ana Fiallos & Emma Bordier

import astropy
from astropy.io import fits
from astropy.table import Table
import scipy.ndimage.interpolation as sni
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import stats  
import os as os
from scipy import signal


direc='/Users/bordieremma/Documents/Magistere_3/MT2/'

#Champ8R=fits.open(direc+'champ8R_mag0.cat',format='fits')[0].data

Champ8R=fits.open(direc+'CODES/FINAL_calibfluxradius.cat')[2].data
Champ3R=fits.open(direc+'CODES/champ3R-2.cat')[2].data
Capak=np.loadtxt(direc+'CODES/Comparaison_Capak.txt')
Sloan=np.loadtxt(direc+'CODES/Sloan2.txt')
CFHT=fits.open(direc+'cfht2.cat')[2].data
#Sloan=Sloan.astype(np.float)

Champ8R=np.array(Champ8R)    #704 objets
Champ3R=np.array(Champ3R)    #947 objets
CFHT=np.array(CFHT)

mag8R=[]
mag3R=[]
magCFH=[]
r=0
p=0
q=0
for i in range (len(Champ8R)):
	r=Champ8R[i][16]
	mag8R.append(r)

for i in range(len(Champ3R)):
	p=Champ3R[i][16]
	mag3R.append(p)

for i in range(len(CFHT)):
	q=CFHT[i][16]
	magCFH.append(q)

mag8R=np.array(mag8R)
mag3R=np.array(mag3R)
magCFH=np.array(magCFH)

mini8R=min(mag8R)
maxi8R=max(mag8R)

mini3R=min(mag3R)
maxi3R=max(mag3R)

miniCFH=min(magCFH)
maxiCFH=max(magCFH)

binmag3R=np.arange(int(mini3R), int(maxi3R)+1, 0.5)
binmag8R=np.arange(int(mini8R), int(maxi8R)+1, 0.5)
binmagCFH=np.arange(int(miniCFH), int(maxiCFH)+1, 0.5)
nombre_objets8R=np.zeros(len(binmag8R)-1)
nombre_objets3R=np.zeros(len(binmag3R)-1)
nombre_objetsCFH=np.zeros(len(binmagCFH)-1)

for i in range(len(binmag8R)-1):
	#nombre_objets[i]=np.shape(np.where(mag>binmag[i]) and np.where(mag<binmag[i+1]))[1]
	nombre_objets8R[i]=len(mag8R[(mag8R>=binmag8R[i]) & (mag8R<binmag8R[i+1])])

for i in range(len(binmag3R)-1):
	#nombre_objets[i]=np.shape(np.where(mag>binmag[i]) and np.where(mag<binmag[i+1]))[1]
	nombre_objets3R[i]=len(mag3R[(mag3R>=binmag3R[i]) & (mag3R<binmag3R[i+1])])

for i in range(len(binmagCFH)-1):
	nombre_objetsCFH[i]=len(magCFH[(magCFH>=binmagCFH[i]) & (magCFH<binmagCFH[i+1])])



#log=np.log(nombre_objets)

norm=13.1*13.1/(60*60)      #Passer de 13.1*13.1 arcmin à 1deg^2
nombre_objets8R=nombre_objets8R/norm
nombre_objets3R=nombre_objets3R/norm

#print(np.log(nombre_objets8R[:28]))

mask8R=(np.where(nombre_objets8R<1))[0][1]
mask3R=(np.where(nombre_objets3R<1))[0][1]
#maskCFH=(np.where(nombre_objetsCFH<1))[0][1]


plt.figure()
plt.subplot(1,2,1)
plt.scatter(binmag8R[:mask8R],np.log10(nombre_objets8R[:mask8R]),label='OHP T120 2019',marker="*")
plt.scatter(Capak[:,0]+0.23,Capak[:,1],label='Capak et al 2004',marker='^')
plt.scatter(Sloan[:,0],Sloan[:,1],label='Sloan Commissioning Yasuda et al 2001',marker='+')
#plt.yticks(range(0,90,5))
#plt.xticks(range(10,30,1))
plt.title(r'Comptage des objets Champ 8 Filtre R t$_{pose}$=4$^h$50mn')
plt.xlabel('Magnitude')
plt.ylabel(r'log$_{10}$ N$_{objets}$ par degré$^{2}$ par 0.5mag')
#plt.yscale('log')
plt.legend()
#plt.show()

#plt.figure()
plt.subplot(1,2,2)
plt.scatter(binmag3R[:mask3R],np.log10(nombre_objets3R[:mask3R]),label='OHP T120 2019',marker="*")
plt.scatter(Capak[:,0]+0.23,Capak[:,1],label='Capak et al 2004',marker='^')
plt.scatter(Sloan[:,0],Sloan[:,1],label='Sloan Commissioning Yasuda et al 2001',marker='+')
#plt.yticks(range(0,90,5))
#plt.xticks(range(10,30,1))
plt.title(r'Comptage des objets Champ 3 Filtre R t$_{pose}$=50mn')
plt.xlabel('Magnitude')
plt.ylabel(r'log$_{10}$ N$_{objets}$ par degré$^{2}$ par 0.5mag')
#plt.yscale('log')
plt.legend()

plt.show()

plt.figure()
plt.scatter(binmagCFH[:32], np.log10(nombre_objetsCFH[:32]),label='CFHT',marker='*')
plt.scatter(Capak[:,0]+0.23,Capak[:,1],label='Capak et al 2004',marker='^')
plt.scatter(Sloan[:,0],Sloan[:,1],label='Sloan Commissioning Yasuda et al 2001',marker='+')
plt.title(r'Comptage des objets CFH Filtre R t$_{pose}$=150s')
plt.xlabel('Magnitude')
plt.ylabel(r'log$_{10}$ N$_{objets}$ par degré$^{2}$ par 0.5mag')
#plt.yscale('log')
plt.legend()
plt.show()

plt.figure()
plt.scatter(binmagCFH[:32], np.log10(nombre_objetsCFH[:32]),label='CFHT,t$_{pose}$=150s ',marker='*')
plt.scatter(binmag3R[:mask3R],np.log10(nombre_objets3R[:mask3R]),label='OHP T120 2019, t$_{pose}$=50mn',marker="*")
plt.title(r'Comptage des objets CFH et OHP Filtre R')
plt.xlabel('Magnitude')
plt.ylabel(r'log$_{10}$ N$_{objets}$ par degré$^{2}$ par 0.5mag')
#plt.yscale('log')
plt.legend()
plt.show()

