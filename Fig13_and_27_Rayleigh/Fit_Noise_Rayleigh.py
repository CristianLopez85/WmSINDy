# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 15:34:16 2025

@author: crist
"""

import numpy as np
import scipy.io
# !pip install fitter
from fitter import Fitter

#%%  Noise provided by python
Noise = scipy.io.loadmat('Noise.mat')['Noise']
Wm_NoiseID = scipy.io.loadmat('Wm_NoiseID.mat')['Wm_NoiseID']
mS_NoiseID = scipy.io.loadmat('mS_NoiseID.mat')['mS_NoiseID']

#%%  Noise provided by python

distx=Fitter(Noise[:,0],bins=100)
# distx.distributions = ['norm','beta','uniform','gamma','cauchy','dweibull','rayleigh']
distx.distributions = ['rayleigh']

distx.fit()
distx.summary()
print(distx.get_best())

disty=Fitter(Noise[:,1],bins=100)
# disty.distributions = ['norm','beta','uniform','gamma','cauchy','dweibull','rayleigh']
disty.distributions = ['rayleigh']

disty.fit()
disty.summary()
print(disty.get_best())

distz=Fitter(Noise[:,2],bins=100)
# distz.distributions = ['norm','beta','uniform','gamma','cauchy','dweibull','rayleigh']
distz.distributions = ['rayleigh']

distz.fit()
distz.summary()
print(distz.get_best())

#%% Change for x, y or z

# Extract the parameters
best_fit = distx.get_best()
rayleigh_params = best_fit['rayleigh']
loc = rayleigh_params['loc']
scale = rayleigh_params['scale']

# Calculate the mean and standard deviation from the fitted parameters
fitted_mean = loc + scale * np.sqrt(np.pi/2)
fitted_std = scale * np.sqrt((4-np.pi)/2)

# Calculate actual mean and std directly from the data
actual_mean = np.mean(Noise[:,0])
actual_std = np.std(Noise[:,0])

print(f"Fitted parameters: loc={loc}, scale={scale}")
print(f"Theoretical mean from fit: {fitted_mean}, Actual mean: {actual_mean}")
print(f"Theoretical std from fit: {fitted_std}, Actual std: {actual_std}")
# print(f"Target std (NoiseMag[0]): {NoiseMag[0]}")

#%%  Noise provided by WmSINDy

distx=Fitter(Wm_NoiseID[:,0],bins=100)
# distx.distributions = ['norm','beta','uniform','gamma','cauchy','dweibull','rayleigh']
distx.distributions = ['rayleigh']
distx.fit()
distx.summary()
print(distx.get_best())

#%
disty=Fitter(Wm_NoiseID[:,1],bins=100)
# disty.distributions = ['norm','beta','uniform','gamma','cauchy','dweibull','rayleigh']
disty.distributions = ['rayleigh']
disty.fit()
disty.summary()
print(disty.get_best())

#%
distz=Fitter(Wm_NoiseID[:,2],bins=100)
# distz.distributions = ['norm','beta','uniform','gamma','cauchy','dweibull','rayleigh']
distz.distributions = ['rayleigh']
distz.fit()
distz.summary()
print(distz.get_best())

#%%  Noise provided by mSINDy

distx=Fitter(mS_NoiseID[:,0],bins=100)
# distx.distributions = ['norm','beta','uniform','gamma','cauchy','dweibull','rayleigh']
distx.distributions = ['rayleigh']
distx.fit()
distx.summary()
print(distx.get_best())

#%
disty=Fitter(mS_NoiseID[:,1],bins=100)
# disty.distributions = ['norm','beta','uniform','gamma','cauchy','dweibull','rayleigh']
disty.distributions = ['rayleigh']
disty.fit()
disty.summary()
print(disty.get_best())

#%
distz=Fitter(mS_NoiseID[:,2],bins=100)
# distz.distributions = ['norm','beta','uniform','gamma','cauchy','dweibull','rayleigh']
distz.distributions = ['rayleigh']
distz.fit()
distz.summary()
print(distz.get_best())

