

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
from astroML.correlation import two_point
from astroML.correlation import bootstrap_two_point

direc='/Users/bordieremma/Documents/Magistere_3/MT2/'

Champ8R=fits.open(direc+'CODES/FINAL_calibfluxradius.cat')[2].data
Champ3R=fits.open(direc+'CODES/champ3R-2.cat')[2].data

mag3R=[]
k=0
for i in range(len(Champ3R)):
	k=Champ3R[i][16]
	mag3R.append(k)

mag3R=np.array(mag3R)

mini3R=min(mag3R)
maxi3R=max(mag3R)

binmag3R=np.arange(int(mini3R), int(maxi3R)+1, 0.5)
nombre_objets3R=np.zeros(len(binmag3R)-1)

for i in range(len(binmag3R)-1):
	#nombre_objets[i]=np.shape(np.where(mag>binmag[i]) and np.where(mag<binmag[i+1]))[1]
	nombre_objets3R[i]=len(mag3R[(mag3R>=binmag3R[i]) & (mag3R<binmag3R[i+1])])

#On fait un mask pour prendre les objets dont 12<mag<mag_compl

mag_compl=binmag3R[np.where(nombre_objets3R==max(nombre_objets3R))[0][0]]

#index_final=np.where()

#Champ3R_mask

X=[]      #XWIN
Y=[]      #YWIN
r=0
p=0
for i in range (len(Champ3R)):
	r=Champ3R[i][2]
	p=Champ3R[i][3]
	X.append(r)
	Y.append(p)


Xfinal=[]
Yfinal=[]
for i in range(len(Champ3R)):
	if (mag3R[i]>12) and (mag3R[i]<mag_compl):
		Xfinal.append(X[i])
		Yfinal.append(Y[i])

Xfinal=np.array(Xfinal)
Yfinal=np.array(Yfinal)

data=np.zeros((len(Xfinal),2))
data[:,0]=Xfinal
data[:,1]=Yfinal


norm=np.sqrt(13.1*13.1/(1024*1024*60*60))     #(13.1*13.1/(1024*1024)) donne la dim d'un pixel en arcmin 
data=data*norm

#bins = np.logspace(-3, -2, 10)

data_R=np.random.uniform(np.max(data),np.min(data),(649,2))

plt.figure()
plt.scatter(data[:,0],data[:,1],marker='+',label='Donnees OHP')
plt.scatter(data_R[:,0],data_R[:,1],marker='.',label='Donnees aleatoires')
plt.title('Distribution des données')
plt.legend()
plt.show()

bins=np.logspace(-3,-2,10)


corr = two_point(data, bins, method='standard',data_R=data_R, random_state=None)
corr2= two_point(data,bins,method='landy-szalay',data_R=data_R,random_state=None)
#np.allclose(corr, 0, atol=0.02)
#True

bins2=[]
for i in range(len(bins)-1):
	bins2.append((bins[i]+bins[i+1])/2)

plt.figure()
plt.scatter(bins2,corr,marker='+',label='Method:Standard')
plt.scatter(bins2,corr2,marker='.',label='Method:landy-szalay')
plt.title('Corrélation entre une distribution aléatoire et distribution de nos données')
plt.xlabel('Bins')
plt.ylabel('Corrélation')
plt.show()


'''


#Pour une distribution gaussienne
corr2, dcorr2 = bootstrap_two_point(X, bins, Nbootstrap=5)
np.allclose(corr2, 0, atol=2 * dcorr2)
True'''

