#@authors: Ana Fiallos & Emma Bordier
#PRETRAITEMENT CHAMP3 07.01.2019 FILTRE B 

import astropy
from astropy.io import fits
import scipy.ndimage.interpolation as sni
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import stats  
import os as os
from scipy import signal


direc='/Users/bordieremma/Documents/Magistere_3/MT2'

def writefits(array, fname, overwrite=False):

  if (os.path.isfile(fname)):
    if (overwrite == False):
      print("File already exists!, skiping")
      return
    else:
      print("File already exists!, overwriting")
      os.remove(fname)
  try: 
    hdu = fits.PrimaryHDU(array)
    hdu.writeto(fname)
  except:
    print('b1') 
    hdu = pf.PrimaryHDU(fname)
    print('b2') 
    hdu.writeto(array)

  print("File: %s saved!" % fname)

  return

champ3B=np.zeros((1024,1024,5))
offset=np.zeros((1024,1024,10))           #CREATION OFFSET POUR DONNEES 08.01
offset_09_01=np.zeros((1024,1024,10))     #CREATION OFFSET POUR DONNEES 09.01
flatB=np.zeros((1024,1024,10))

for i in range(1,5):
    champ3B[:,:,i-1]=fits.open(direc+"/DONNEES_NUIT2/Champ3_FiltreB/champ3-00"+str(i)+"B.fit")[0].data

for i in range(1,11):
    offset[:,:,i-1]=fits.open(direc+"/DONNEES_NUIT1/Calibrations_2019/Bias-00"+str(i)+"SII.fit")[0].data
    offset_09_01[:,:,i-1]=fits.open(direc+"/DONNEES_NUIT3/OFFSET/offset-00"+str(i)+".fit")[0].data
    flatB[:,:,i-1]=fits.open(direc+"/DONNEES_NUIT1/Calibrations_2019/Flat_B-00"+str(i)+"B.fit")[0].data



offset_median=np.median(offset,axis=2)
offset09_median=np.median(offset_09_01,axis=2)

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

n1 , bins1, patches = plt.hist(offset09_median.flatten(), bins=60,range=(285,315) , histtype='bar' )
x1 = np.linspace(bins2[0], bins2[-1], 60)
popt1, pcov1 = curve_fit(Gauss, x2, n2, p0=(max(n2), np.mean(offset09_median),np.std(offset09_median)))

n2 , bins2, patches = plt.hist(offset_median.flatten(), bins=60,range=(285,315) , histtype='bar' )
x2 = np.linspace(bins2[0], bins2[-1], 60)
popt2, pcov2 = curve_fit(Gauss, x2, n2, p0=(max(n2), np.mean(offset_median),np.std(offset_median)))

Superbias=popt2[1]
Superbias09=popt1[1]

flatnorm=np.zeros((1024,1024,10))
flatnorm09=np.zeros((1024,1024,10))          #RELIE A OFFSET09   


for i in range(10):
	flatnorm[:,:,i]=(flatB[:,:,i]-Superbias)/np.mean(flatB[:,:,i]-Superbias)
  flatnorm09[:,:,i]=(flatB[:,:,i]-Superbias09)/np.mean(flatB[:,:,i]-Superbias09)


Superflat=np.mean(flatnorm,axis=2)  


champ_corrB=np.zeros((1024,1024,4))


for i in range(4):
    champ_corrB[:,:,i]=(champ3B[:,:,i]-Superbias)/Superflat
    #champ_corr2[:,:,i]=(champ3[:,:,i]-Superbias)/Superflat2
    writefits(champ_corrB[:,:,i],'ChampCorrB'+str(i)+'.fit')
    #writefits(champ_corr2[:,:,i],'ChampCorr'+str(i)+'_2.fit')
    writefits(champ3B[:,:,i],'Champ'+str(i)+'.fit')


########AJOUTER LES DONNEES DU 09.01 OUI OU NON?


