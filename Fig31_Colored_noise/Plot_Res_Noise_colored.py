# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 15:33:41 2025

@author: crist
"""

# =============================================================================
# This file will plot the merged result of swiping the noise level
# =============================================================================
#%% Import packages
import numpy as np
from scipy.integrate import odeint
from scipy.stats import dweibull
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from utils_NSS_SINDy import *
#%% Load result file

# Define the plot parameters
def ChangeFontSize(fontSize):
    plt.rc('xtick', labelsize=fontSize)    
    plt.rc('ytick', labelsize=fontSize) 
    return None

def CustomizeViolin(data,line_color,edge_color,lw1):
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
    inds = np.arange(0,len(medians))
    plt.plot(medians,linewidth=lw1,color=line_color,alpha=0.6,zorder=0)
    
    return None

# Define the color
color_WmS=[0.58, 0.44, 0.86]
color_NSS_SINDy='c'
color_WSINDy=[0.5,0.3,0.1]
edge_color="black"

# Define the transparancy
transAlpha=0.95

# Define the line width
lw1=8
lw2=2
lw3=1.5  # 3.5

width_violin=0.25

# Define the lable
labelNSS_SINDy="NSS-SINDy"

# Define the marker properties
MarkerSize_Violin=6
MarkerEdgeWidth=3
MarkerSize=200

# LengthToSwipe=LengthToSwipe[0,:]
LengthToSwipe=['Noise','Vector Field','Short Pred.','Parameter']                          # Added by CL
              
#%%  Load Identified Noises

#---------------  Load the MATLAB .mat file into Python
Res_mSINDy = scipy.io.loadmat('mLorenz_blue.mat') 
Res_WmSINDy = scipy.io.loadmat('WmLorenz_blue.mat')  

#---- Get simulated noises

def colored_noise_generator(N):
    """Generate colored noise of length N"""
    X_white = np.fft.rfft(np.random.randn(N))
    f = np.fft.rfftfreq(N)
    S = np.sqrt(f)                                      # Blue noise has f^(1/2) power spectrum
    # S = f                                             # Violet noise
    # S = 1/np.where(f == 0, float('inf'), np.sqrt(f))  # Pink noise
    # S = 1/np.where(f == 0, float('inf'), f)           # Brown noise
    # Normalize S
    S = S / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)

NoisePerc=[0,5,10,15,20,25,30,35,40,45,50]   
NoiseNum = len(NoisePerc)

Noise_level = 2                   # 10-> 50%
NoisePercentage = NoisePerc[Noise_level]  # -----> match with: j = NoiseNum - 1

Nloop = 5

# blue: 2, pink: 4, brown 1~, violet~1, en 8 lig mejor
#%%
fig= plt.subplots(figsize=(2,3))
ChangeFontSize(20)
# Labels for the bars
Wm_S_R = np.squeeze(Res_WmSINDy['Rate_success'])[:,:,Nloop][:,Noise_level];  
m_S_R = np.squeeze(Res_mSINDy['Rate_success'])[:,:,Nloop][:,Noise_level];  

labels = ['mSINDy', 'WmSINDy']

# Adding labels and title
plt.ylabel('Success rate %', fontsize=20)
# Heights of the bars
heights = [2*np.sum(m_S_R), 2*np.sum(Wm_S_R)]
bar_width = 0.3  # Adjust this value to change the width

# Plotting bars
plt.bar(labels, heights, color=[color_NSS_SINDy, color_WmS], width=bar_width, edgecolor='black')
plt.yticks([25,50,75,100])
plt.grid(axis='y', color='gray', linestyle='-', alpha=0.7)

plt.savefig("p30_blue_left.pdf")

#%%  Process Identified Noises

N_run = 50

Wm_NoiseIDList = Res_WmSINDy['NoiseID']
Wm_NoiseIDList = Wm_NoiseIDList.reshape(N_run, NoiseNum, 6, 2500, 3)

# 10 for 50% noise,                                      5 for last Nloop
Wm_noise_ids_at_50_percent_last_k = Wm_NoiseIDList[:, Noise_level, 5, :, :]

# Extract x component 
Wm_noise_ids_x = Wm_noise_ids_at_50_percent_last_k[:, :, 0].flatten()
Wm_noise_ids_y = Wm_noise_ids_at_50_percent_last_k[:, :, 1].flatten()
Wm_noise_ids_z = Wm_noise_ids_at_50_percent_last_k[:, :, 2].flatten()

# NoiseID = np.hstack((Wm_noise_ids_x.reshape(-1, 1),Wm_noise_ids_y.reshape(-1, 1),Wm_noise_ids_z.reshape(-1, 1)))

m_NoiseIDList = Res_mSINDy['NoiseID']
m_NoiseIDList = m_NoiseIDList.reshape(N_run, NoiseNum, 6, 2500, 3)

# 10 for 50% noise,                                    5 for last Nloop
m_noise_ids_at_50_percent_last_k = m_NoiseIDList[:, Noise_level, 5, :, :]

# Extract x component 
m_noise_ids_x = m_noise_ids_at_50_percent_last_k[:, :, 0].flatten()
m_noise_ids_y = m_noise_ids_at_50_percent_last_k[:, :, 1].flatten()
m_noise_ids_z = m_noise_ids_at_50_percent_last_k[:, :, 2].flatten()

# NoiseID = np.hstack((m_noise_ids_x.reshape(-1, 1),m_noise_ids_y.reshape(-1, 1),m_noise_ids_z.reshape(-1, 1)))

#%%

# Define the simulation parameters
p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])

# Define the time points
T=25.0
dt=0.0002  

# Define the initial conditions
x0=np.array([5.0,5.0,25.0])

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)

stateVar,dataLen=np.transpose(x).shape

pin = 0

N_run = 1

# Initialize noise arrays
noise_x = np.zeros((dataLen, N_run))
noise_y = np.zeros((dataLen, N_run))
noise_z = np.zeros((dataLen, N_run))


        
for i in range(N_run): 
    
    pin=pin+1
    np.random.seed(pin)
        
        #-------------------------- Generate the noise ------------------------
        # These values (NoiseMag: standard deviation) are presented in the manuscript
        # NoiseMag = [np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
        
        #----------- Gaussian Noise
        # Noise = np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])    
                
        #----------- Colored Noise
        
    Noise = np.zeros_like(x)
    for ii in range(stateVar):
            # Generate blue noise
        noise_signal = colored_noise_generator(dataLen)
        # Scale the noise to match the desired percentage of the signal's standard deviation
        NoiseMag = np.std(x[:, ii]) * NoisePercentage * 0.01
        Noise[:, ii] = NoiseMag * noise_signal
        
    noise_x[:, i] = Noise[:,0]
    noise_y[:, i] = Noise[:,1]
    noise_z[:, i] = Noise[:,1]
        
    
    # rrr
    #     # Compute and plot PSD for each noise
    #     f, Pxx = welch(Wm_noise_ids_x, fs=fs)
    #     plt.semilogy(f, Pxx)

    #     # Compute and plot PSD for each noise
    #     f, Pxx = welch(m_noise_ids_x, fs=fs)
    #     plt.semilogy(f, Pxx)
        
# NoiseID = np.hstack((noise_x.reshape(-1, 1),noise_y.reshape(-1, 1),noise_z.reshape(-1, 1)))

colorNoise='black'
colorNoiseID_Wm='red'
colorNoiseID_mS='cornflowerblue'

lw = 8

plt.figure(figsize=(10,20))
ChangeFontSize(25)

Iter = 0

from scipy.signal import welch
# Density x
plt.subplot(3,1,1)
plt.grid()

# Compute and plot PSD for each noise
f, Pxx = welch(noise_x.flatten(), fs=100, nperseg=1024, scaling='density')
plt.semilogy(f, Pxx,  linewidth = lw , color=colorNoise)

# Compute and plot PSD for each noise
f, Pxx = welch(Wm_noise_ids_x, fs=100, nperseg=1024, scaling='density')
plt.semilogy(f, Pxx, linestyle='--', linewidth = lw , color=colorNoiseID_Wm)

# Compute and plot PSD for each noise
f, Pxx = welch(m_noise_ids_x, fs=100, nperseg=1024, scaling='density')
plt.semilogy(f, Pxx, linestyle='--', linewidth = lw , color=colorNoiseID_mS)

#%%
# Density y
plt.subplot(3,1,2)
plt.grid()
# Compute and plot PSD for each noise
f, Pxx = welch(noise_y.flatten(), fs=100, nperseg=1024, scaling='density')
plt.semilogy(f, Pxx, linewidth = lw ,  color=colorNoise)

# Compute and plot PSD for each noise
f, Pxx = welch(Wm_noise_ids_y, fs=100, nperseg=1024, scaling='density')
plt.semilogy(f, Pxx, linestyle='--', linewidth = lw ,  color=colorNoiseID_Wm)

# Compute and plot PSD for each noise
f, Pxx = welch(m_noise_ids_y, fs=100, nperseg=1024, scaling='density')
plt.semilogy(f, Pxx, linestyle='--', linewidth = lw ,  color=colorNoiseID_mS)

# plt.savefig("p15r_violet.pdf")

#%%
# Density y
plt.subplot(3,1,3)
plt.grid()
# Compute and plot PSD for each noise
f, Pxx = welch(noise_z.flatten(), fs=100, nperseg=1024, scaling='density')
plt.semilogy(f, Pxx, linewidth = lw ,  color=colorNoise)

# Compute and plot PSD for each noise
f, Pxx = welch(Wm_noise_ids_z, fs=100, nperseg=1024, scaling='density')
plt.semilogy(f, Pxx, linestyle='--', linewidth = lw ,  color=colorNoiseID_Wm)

# Compute and plot PSD for each noise
f, Pxx = welch(m_noise_ids_z, fs=100, nperseg=1024, scaling='density')
plt.semilogy(f, Pxx, linestyle='--', linewidth = lw ,  color=colorNoiseID_mS)

plt.savefig("p30_blue_right.pdf")