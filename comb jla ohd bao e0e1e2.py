# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 15:19:46 2018

@author: Lenovo
"""
import time
import h5py
#print(h5py.__version__)
#import os
#os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.integrate import odeint
from scipy.integrate import quad
import math
import emcee
from math import *
from numpy.linalg import inv


df1 = pd.read_csv("D:\\Amit\\python\\pych\\rjla.csv")
z = df1.loc[:, "z"]
za = z.as_matrix(columns=None)
mu = df1.loc[:, "mu"]
mua = mu.as_matrix(columns=None)
cov = pd.read_csv("D:\\Amit\\python\\pych\\jlacov.csv", header=None)
cova = cov.as_matrix(columns=None)
covm = np.asmatrix(cova)
covi = inv(np.matrix(covm))

from math import log10
from numpy.linalg import inv

nSN=31
f   = open('jla_mub_covmatrix.txt', 'r') # 'r' = read only
cov = np.loadtxt(f)
cov = np.reshape(cov,(nSN,nSN))
f.close()

InvC = np.empty([nSN,nSN])
InvC = np.linalg.inv(cov)

covib = pd.read_csv("D:\\Amit\\python\\pych\\baocmb.csv", header=None)
covab = covib.as_matrix(columns=None)
Y = np.asmatrix(covab)

from math import log10
from numpy.linalg import inv

c = 299792.458;
#e2=0
#e0=2.10450755
H0=67.8
zs=1090
m_t=2



def Sn(e0,e1,e2,wd,al,M):
    LCDMf= lambda x,e0,e1,e2,wd,al : 1/H1(x,e0,e1,e2,wd,al)
    LCDMfint= lambda zb,e0,e1,e2,wd,al : quad(LCDMf, 0, zb,args=(e0,e1,e2,wd,al),full_output=1)[0]
    LCDMfint=np.vectorize(LCDMfint)
    r= mua - M - 5*(np.log10((c)*(1 + za)*(LCDMfint(za,e0,e1,e2,wd,al))))
    return  np.dot(r,np.dot(InvC,r).T) 
    

def H1(x,e0,e1,e2,wd,al):
    c= ((-e0)+(np.power((1+x),(3*(1+2*al*wd-e1)/(2-3*e2+3*al*wd)))*((e0)+(-1+e1-2*al*wd))))/(-1+e1-2*al*wd)
    if c<=0:
        return 10000000000000000000
    return H0*c


zb1=[[0.106],
      [0.35],
      [0.57],
      [0.44],
      [0.6],
      [0.73]]

zbt=np.array(zb1)

mub1=[[30.84],
      [10.33],
      [6.72],
      [8.41],
      [6.66],
      [5.43]]     

mubt=np.array(mub1)


def Bao(e0,e1,e2,wd,al):
    rs=lambda x,e0,e1,e2,wd,al: c/H1(x,e0,e1,e2,wd,al)
    dA=lambda k,e0,e1,e2,wd,al: quad(rs, 0,k,args=(e0,e1,e2,wd,al),full_output=1)[0]
    Dv=lambda k,e0,e1,e2,wd,al: (((dA(k,e0,e1,e2,wd,al)**2)*c*k)/H1(k,e0,e1,e2,wd,al))**(1/3)
    Dv=np.vectorize(Dv)
    f=(dA(zs,e0,e1,e2,wd,al)/Dv(zb1,e0,e1,e2,wd,al))- mub1
    return np.dot(f.T,np.dot(Y,(f)))


zohd, Hohd, sigohd=np.genfromtxt('OHD.txt',unpack=True)


def Ohd(e0,e1,e2,wd,al):
    HH=lambda k,e0,e1,e2,wd,al: (H1(zohd[k], e0,e1,e2,wd,al))
    ohd=lambda k: ((Hohd[k] - HH(k,e0,e1,e2,wd,al))/sigohd[k])**2
    num=0
    for j in range(len(zohd)):
        num = num + ohd(j)
        
    part= np.sum(num)
    return part
    

def lnlike(theta):
    e0,e1,e2,wd,al,M = theta
    c= -0.5*(Ohd(e0,e1,e2,wd,al))  -Bao(e0,e1,e2,wd,al)/2  -Sn(e0,e1,e2,wd,al,M) / 2.0 # 
    if np.isnan(c):
        return  -np.inf
    return c


def lnprior(theta):
    e0,e1,e2,wd,al,M  = theta
    if 0< e0 <30 and 0<(2-3*e2+3*al*wd)<20 and 0<(2+3*al*wd)<20 and -20< e2<20 and -20< e1<20  and -20< wd <0.0 and 0< al <20 and -20< (3*e0 -3 + 3*e1 - 6*al*wd)<0 and 0< 2*e1 - 3*e2 + 3*al*e1*wd - 6*al*e2*wd<20 : #and -20<(2-3*e2+3*al*wd)<0  : #and 2<al<3 and 1.8<wd<2.8 :#and 0<(2-3*e2+3*al*wd)<10 and -10<((3*e0/H0)-1+3*e1-3*al*wd-3*e2) <0 :# and 0<al<2 and -2<wd<0: #and 0<((e0/H0)-1+e1-2*al*wd)<10
        return 0.0
    return -np.inf

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

print(lnprob([4.021,-6.972,-4.656,-0.089,0.01,24.995]))
print(lnprob([3.988,-6.911,-4.615,-0.068,0.014,24.994]))
#print(lnlike(59[0.0098,-1.24,-4.6,-2.5,0.19,24.867], za, mua, InvC))
result1=[4.021,-6.972,-4.656,-0.089,0.01,24.95]


ndim, nwalkers = 6, 2000
pos = [result1 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
#pos = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))


nsteps=3000

        
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from multiprocessing import Pool
from IPython.display import display, Math

if __name__ == '__main__':
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,pool=pool)
        start = time.time()
        sampler.run_mcmc(pos,nsteps, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        samples = sampler.get_chain(discard=500, thin=200, flat=True)
        names = ["e0","e1","e2","wd","al","M"]
        labels = ["e0","e1","e2","wd","al","M"]
        cut_samps = samples[samples[:,0]>0,:]
        cut_samps1 = cut_samps[cut_samps[:,3]<0,:]
        cut_samps2 = cut_samps1[cut_samps1[:,4]>0,:]
        cut_samples = MCSamples(samples=cut_samps2, names = names, labels = labels, 
                                ranges={'e0':(0.0, None),'e1':( None, None),'e2':( None, None),'wd':(-0.2, 0),'al':(0, 0.2)})
#,'M':(None, None)
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            
        axes[-1].set_xlabel("step number");

        g = plots.getSubplotPlotter()
        g.settings.num_plot_contours = 2
        g.triangle_plot([cut_samples], filled=True)
                        #,legend_labels=['68.3% confidence region','95.4% confidence region'], 
                        #legend_loc='upper right')
        #g.add_legend(['68.3 % confidence region', '95.4 % confidence region'], legend_loc='upper right', line_offset=0, legend_ncol=None, colored_text=True)
        g.export('triangle1_plot.png')
        g.export('triangle1_plot.pdf')
        
        
        for i in range(ndim):
            mcmc = np.percentile(cut_samps2[:, i], [15.87, 50, 84.13])
            q = np.diff(mcmc)
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            display(Math(txt))
        
        print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))
        
        tau = sampler.get_autocorr_time()
        print(tau)




