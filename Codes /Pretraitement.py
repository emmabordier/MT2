#@authors: Ana Fiallos & Emma Bordier

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


direc='/Users/bordieremma/Documents/Magistere_3/MT2/DONNEES_NUIT1/'

##########CREER UN FICHIER FITS
def writefits(array, fname, overwrite=False):

  if (os.path.isfile(fname)):
    if (overwrite == False):
      print("File already exists!, skiping")
      return
    else:
      print("File already exists!, overwriting")
      os.remove(fname)
  try: 
    hdu = fits.PrimaryHDU(array.T)
    hdu.writeto(fname)
  except:
    print('b1') 
    hdu = pf.PrimaryHDU(fname)
    print('b2') 
    hdu.writeto(array.T)

  print("File: %s saved!" % fname)

  return

#champ3=fits.open(direc+"champ3bis-001R.fit")[0].data
champ3=np.zeros((1024,1024,5))
offset=np.zeros((1024,1024,10))
flatR=np.zeros((1024,1024,10))

for i in range(1,6):
    champ3[:,:,i-1]=fits.open(direc+"Champ3-06-01-19/champ3bis-00"+str(i)+"R.fit")[0].data

for i in range(1,11):
    offset[:,:,i-1]=fits.open(direc+"OFFSET120-nuit1/offset-00"+str(i)+".fit")[0].data
    flatR[:,:,i-1]=fits.open(direc+"Calibrations_2019/Flat-00"+str(i)+"R.fit")[0].data
    #writefits(flatR[:,:,i-1],'flatR'+str(i)+'.fit')


'''
###########VERIFICATION DIMINUTION DU RAPPORT SNR EN RACINE(NOMBRE IMAGES) 

champ_solo=champ3[:,:,1]
champ_mean=np.mean(champ3,axis=2)
champ_median=np.median(champ3,axis=2)

Var_champ_solo=np.std(champ_solo)
Var_champ_mean=np.std(champ_mean)
Var_champ_median=np.std(champ_median)

print('Ecart type 1image',Var_champ_solo)
print('Ecart type Moyenne des 5 images', Var_champ_mean)
print('Ecart type Mediane des 5 images', Var_champ_median)'''



###########SUPERBIAS

offset_mean=np.mean(offset,axis=2)
offset_median=np.median(offset,axis=2)


def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


n1 , bins1, patches = plt.hist(offset_mean.flatten(), bins=60,range=(285,315) , histtype='bar' )
x1 = np.linspace(bins1[0], bins1[-1], 60)

n2 , bins2, patches = plt.hist(offset_median.flatten(), bins=60,range=(285,315) , histtype='bar' )
x2 = np.linspace(bins2[0], bins2[-1], 60)

popt1, pcov1 = curve_fit(Gauss, x1, n1, p0=(max(n1), np.mean(offset_mean),np.std(offset_mean)))
popt2, pcov2 = curve_fit(Gauss, x2, n2, p0=(max(n2), np.mean(offset_median),np.std(offset_median)))

mean_1=popt1[1]
mean_2=popt2[1]

std_1=popt1[2]
std_2=popt2[2]

print('Moyenne Offset_mean',mean_1)
print('Moyenne Offset_median',mean_2)
print('Ecart type Offset_mean',std_1)
print('Ecart type Offset_median',std_2)


plt.figure()
plt.subplot(2,1,1)
plt.plot(x1, Gauss(x1, *popt1), 'r-', label='fit: max=%5.3f, moyenne=%5.3f, sigma=%5.3f' % tuple(popt1))
plt.legend()
plt.hist(offset_mean.flatten(), bins=60, range=(285,315) , histtype='bar' )
plt.title(r"Histogramme offset moyen($t_{pose} nul$) à $-60^\circ$C")
plt.xlabel("ADU")
plt.ylabel("Nombre de pixels")

#plt.figure()
plt.subplot(2,1,2)
plt.plot(x2, Gauss(x2, *popt2), 'r-', label='fit: max=%5.3f, moyenne=%5.3f, sigma=%5.3f' % tuple(popt2))
plt.legend()
plt.hist(offset_median.flatten(), bins=60, range=(285,315) , histtype='bar' )
plt.title(r"Histogramme offset median($t_{pose} nul$) à $-60^\circ$C")
plt.xlabel("ADU")
plt.ylabel("Nombre de pixels")

plt.tight_layout()
plt.show()
plt.clf()

Superbias=popt2[1]

###########SUPERFLAT

flatnorm=np.zeros((1024,1024,10))

for i in range(10):
	flatnorm[:,:,i]=(flatR[:,:,i]-Superbias)/np.mean(flatR[:,:,i]-Superbias)
	#writefits(flatnorm[:,:,i],'flatnorm'+str(i)+'.fit')
	#print('mean=',np.mean(flatnorm[:,:,i]))
	#print('std=',np.std(flatnorm[:,:,i]))

#Visualisation d'une tranche


Superflat=np.mean(flatnorm,axis=2)   

print('Superflat=',Superflat)

#Lissage du superflat 


#pixel=np.ones((1024,1024))

#M=signal.convolve2d(pixel,Superflat[::1024])
#NewSuperflat[i*1024::1024]=M[i]

#NewSuperflat=np.ones((1024,1024))
#moy_loc_flat=[]
#for i in range(0,1023,128):
#	moy_loc_flat.append(np.mean(Superflat[i:i+128,i:i+128]))

#for j in range(1024):
#	for i in range(1024):
#		Superflat
	#NewSuperflat[i:i+128,i:i+128]=Superflat[i:i+128,i:i+128]/np.mean(Superflat[i:i+128,i:i+128])
	
#print('NewSuperflat=',NewSuperflat)


###COUPE D'UN FLAT ET STATISTIQUE 


b=np.mean(np.mean(flatR-Superbias))
print(np.std(Superflat))

a=np.mean((flatR[:,:,3]-Superbias))
c=(a,b)
print('Normalisation par', a)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(Superflat,cmap='GnBu')
plt.title('Superflat total')
plt.xlabel('pixels')
plt.ylabel('intensité')
plt.subplot(2,2,3)
plt.plot(Superflat[750,:],label='tranche à 750')
plt.plot(Superflat[700,:],label='tranche à 700')
plt.plot(Superflat[600,:],label='tranche à 600')
plt.legend()
plt.subplot(1,2,2)
plt.hist((flatR[:,:,3]-Superbias).flatten(),range=(33000,37000),label='Moyenne=%5.3f, Superflat=%5.3f' %tuple(c) )# bins=60, range=(285,315) , histtype='bar')
plt.title('Histogramme flatR_3 ')
plt.xlabel('ADU')
plt.ylabel('Nombre de pixels')
plt.legend()


plt.tight_layout()
plt.show()
plt.clf()

#writefits(Superflat,'Superflat.fit')

difflat=flatnorm[:,:,0]-flatnorm[:,:,1]
difflat2=flatnorm[:,:,0]-Superflat

writefits(difflat,'difflat.fit')
writefits(difflat2,'difflat2.fit')

#flat_mean=np.mean(flatR,axis=2)
#flat_median=np.median(flatR,axis=2)

#Superflat_mean=(flat_mean-Superbias)/np.median(flat_mean-Superbias)
#Superflat_median=(flat_median-Superbias)/np.median(flat_median-Superbias)

'''
plt.figure()
plt.subplot(1,2,1)
plt.imshow(Superflat_mean,cmap='GnBu')
plt.title('Superflat moyen')
plt.subplot(1,2,2)
plt.imshow(Superflat_median,cmap='GnBu')
plt.title('Superflat median')
plt.show()'''


###########ENLEVER OFFSET ET DIVISER FLAT

champ_corr=np.zeros((1024,1024,5))

for i in range(5):
    champ_corr[:,:,i]=(champ3[:,:,i]-Superbias)/Superflat
    #champ_corrnew[:,:,i]=(champ3[:,:,i]-Superbias)/Superflat
    #writefits(champ_corr[:,:,i-1],'ChampCorr'+str(i)+'.fit')
    #writefits(champ3[:,:,i-1],'Champ'+str(i)+'.fit')

writefits(champ_corr[:,:,2],'ChampCorrNew.fit')
writefits(champ3[:,:,2],'Champ3_1.fit')

plt.figure()
plt.subplot(2,2,1)
plt.imshow(champ3[:,:,0],cmap='viridis')
plt.title('Champ profond 3 image brute')
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(champ_corr[:,:,0],cmap='viridis')
plt.title('Champ profond 3 image réduite')
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(champ_corr[:,:,0]-champ3[:,:,0],cmap='plasma')
plt.title('image corrigée-image brute')
plt.colorbar()

plt.tight_layout()
plt.show()
plt.clf()



#writefits(Superflat_median, 'SuperflatR.fits')
###########DIVISION PAR UN FLAT QUELCONQUE ET PAS UN SUPERFLAT

champ_corr2=np.zeros((1024,1024,5))

for i in range(5):
    champ_corr2[:,:,i]=(champ3[:,:,i]-Superbias)/((flatR[:,:,3]-Superbias)/np.mean((flatR[:,:,3]-Superbias)))
    writefits(champ_corr2[:,:,i],'ChampCorr'+str(i)+'_2.fit')

writefits(((flatR[:,:,3]-Superbias)/np.mean((flatR[:,:,3]-Superbias))),'Superflat_3.fit')



###########D'ABORD RECENTRER LES IMAGES



#~/Documents/Magistere_3/MT2/swarp-2.38.0/src/swarp Astrom_C3_B0.fits Astrom_C3_B1.fits Astrom_C3_B2.fits Astrom_C3_B3.fits  -IMAGE_OUT_NAME FILTREB.fits



###########PRENDRE LA MEDIANE 





