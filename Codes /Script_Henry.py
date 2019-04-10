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


direc='/Users/bordieremma/Documents/Magistere_3/MT2/CODES/'

def plot_kcounts(figname):
	fl=direc+'Comparaison_Capak.txt'
	f = open(fl, 'w')
	histo=np.histogram(f[:,0],range=(15,25),bins=20)
	bins=(histo[1]+0.25)[:-1]
	ngals=histo[0]/1.66
	#(k_hjmcc,ngal_hjmcc)=np.loadtxt('/Users/hjmcc/cosmos/uvista/analysis/counts_all-hjmcc2010.dat', usecols=(0,2),unpack=True)
	#(k_wirds,ngalK,ngalK_err)=np.loadtxt('/Users/hjmcc/cosmos/uvista/analysis/wirds_counts_v2.txt', usecols=(0,8,9),unpack=True)
	#(k_quadri,ngalK_quadri)=np.loadtxt('/Users/hjmcc/cosmos/uvista/analysis/quadri-counts.txt', usecols=(0,1),unpack=True)
	#(k_will,ngalK_uds)=np.loadtxt('/Users/hjmcc/cosmos/uvista/analysis/DR8_counts.dat', usecols=(0,2),unpack=True)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.set_yscale('log')
	ax.set_xlim(17.5,25.5)
	ax.set_ylim(5e1,4e5)
	plt.xlabel(r'$K_{\rm s}$')
	plt.ylabel(r'$N_{\rm gal}~0.5~{\rm mag}^{-1} {\rm deg}^{-2}$')
	plt.plot(bins,ngals,'ro', markersize=10,label='UltraVISTA DR1')
	np.savetxt(f,np.transpose((bins,ngals)),fmt='%5.2f %5.2f')
	#plt.plot(k_hjmcc,10**ngal_hjmcc,'gs',mec='g',mfc='None',markersize=12,label='McCracken et al. 2010')
	#plt.plot(k_wirds,ngalK,'cp',mec='c',mfc='None',markersize=12,label='Bielby et al. 2011')
	#plt.plot(k_quadri+1.87,ngalK_quadri/2.0,'b^',mec='b',mfc='None',markersize=12,label='Quadri et al. 2007')
	#plt.plot(k_will,10**ngalK_uds/2.0,'mD',mec='m',mfc='None',markersize=12,label='UDS DR8')
	#plt.legend(loc='lower right',numpoints=1,fancybox=True)
	plt.savefig(figures+figname)

plot_kcounts(test)

