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

direc='/Users/bordieremma/Documents/Magistere_3/MT2/RESULTATS_SWARP/'

tposeC3_R= 5*600 #FILTRE R: 600s
tposeC3_B= 4*1200    #FILTRE B: 1200s

tposeC8_R= 29*600
tposeC8_B= 4*1200

C3_R=fits.open(direc+'FinalR_C3.fits')[0].data
C3_B=fits.open(direc+'FinalB_C3.fits')[0].data

C8_R=fits.open(direc+'FinalR_C8.fits')[0].data
C8_B=fits.open(direc+'FinalB_C8.fits')[0].data

#NORMALISATION PAR LE TEMPS DE POSE 	

C3_R_Norm=C3_R/tposeC3_R
C3_B_Norm=C3_B/tposeC3_B

C8_R_Norm=C8_R/tposeC8_R
C8_B_Norm=C8_B/tposeC8_B

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

writefits(C3_R_Norm,'C3_R_Norm.fits')
writefits(C3_B_Norm,'C3_B_Norm.fits')
writefits(C8_R_Norm,'C8_R_Norm.fits')
writefits(C8_B_Norm,'C8_B_Norm.fits')

