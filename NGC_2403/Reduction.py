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
direc1='/Users/bordieremma/Documents/Magistere_3/MT2/CODES/Comparaison/'

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

def reduction(filtre="nom"):   #filtre= C,R,B,Ha,SII
  galaxie=fits.open(direc+'M42/M42-001'+str(filtre)+'.fit')[0].data #'/NGC_2403/NGC_2403-001'+str(filtre)+'.fit')[0].data
  Superbias=300
  if filtre=="R":
    Superflat=fits.open(direc1+'master_flatbis_1024_R_Cousins.fits')[0].data
  else:
    Superflat=flat(filtre,Superbias)
  galaxie=(galaxie-Superbias)/Superflat
  writefits(galaxie,'NGC_2403_reduced_'+str(filtre)+'.fit')
  return 0


def flat(filtre,Superbias):
  flatnorm=np.zeros((1024,1024,10))
  flat=np.zeros((1024,1024,10))
  for k in range (1,11):
    flat[:,:,k-1]=fits.open(direc+"/DONNEES_NUIT1/Calibrations_2019/Flat_"+str(filtre)+"-00"+str(k)+str(filtre)+".fit")[0].data
  for i in range(10):
    flatnorm[:,:,i]=(flat[:,:,i]-Superbias)/np.mean(flat[:,:,i]-Superbias)
  Superflat=np.mean(flatnorm,axis=2)
  return(Superflat)













