# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:37:29 2025

@author: crist
"""

# =============================================================================
# This file will plot the merged result of swiping the noise level
# =============================================================================
#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from utils_NSS_SINDy import *

#%% Load result file

# Define the plot parameters
def ChangeFontSize(fontSize):
    plt.rc('xtick', labelsize=fontSize)    
    plt.rc('ytick', labelsize=fontSize) 
    
    return None

# Define the color
color_WmS = [0.58, 0.44, 0.86]
color_NSS_SINDy = 'c'
edge_color="black"

# Define the transparancy
transAlpha=0.95

# Define the line width
lw1=6
lw2=2
lw3=1.5  # 3.5

width_violin=0.15

# Define the lable
labelNSS_SINDy="NSS-SINDy"

# Define the marker properties
MarkerSize_Violin=6
MarkerEdgeWidth=3
MarkerSize=200

# LengthToSwipe=LengthToSwipe[0,:]
LengthToSwipe=['Noise','Vector Field','Short Pred.','Parameter']                          # Added by CL

#%% Plot when Discrepancy model is used
NoiseNum = 4

import scipy.io
Res_mSINDy = scipy.io.loadmat('m_Id_Noise_n20.mat')  # Load the MATLAB .mat file into Python
Res_WmSINDy = scipy.io.loadmat('Wm_Id_Noise_n20.mat')  # Load the MATLAB .mat file into Python

Nloop = 5

mNoise_Error = np.squeeze(Res_mSINDy['E_noise'])[:,Nloop];  
mField_Error = np.squeeze(Res_mSINDy['E_vector_field'])[:,Nloop]; 
m_Sh_Error = np.squeeze(Res_mSINDy['E_short_traj'])[:,Nloop];  
m_Par_Error = np.squeeze(Res_mSINDy['E_parameter'])[:,Nloop];  

WmNoise_Error = np.squeeze(Res_WmSINDy['E_noise'])[:,Nloop];   
WmField_Error = np.squeeze(Res_WmSINDy['E_vector_field'])[:,Nloop];  
Wm_Sh_Error = np.squeeze(Res_WmSINDy['E_short_traj'])[:,Nloop];  
Wm_Par_Error = np.squeeze(Res_WmSINDy['E_parameter'])[:,Nloop];  

mSINDy = np.column_stack([mNoise_Error,mField_Error,m_Sh_Error,m_Par_Error])
WmSINDy = np.column_stack([WmNoise_Error,WmField_Error,Wm_Sh_Error,Wm_Par_Error])

quartile1a, median1, quartile1b = np.percentile(WmSINDy, [25, 50, 75], axis=0)
quartile3a, median3, quartile3b = np.percentile(mSINDy, [25, 50, 75], axis=0)

#%%  
fig, ax1 = plt.subplots(figsize=(14,5))
ChangeFontSize(30)
# Plotting on the first axis
line1, = ax1.plot(median1,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')

sns.violinplot(data=WmSINDy,cut=0,inner="box",width=width_violin,scale="width",color=color_WmS,linewidth=lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(NoiseNum),median1,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(NoiseNum))
ax1.set_xticklabels(LengthToSwipe)
plt.yscale('log')
plt.ylabel('Error', fontsize=30);
# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median3,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=mSINDy,cut=0,inner="box",width=width_violin,scale="width",color=color_NSS_SINDy,linewidth=lw2,saturation=transAlpha,zorder=0, ax=ax2)
ax2.scatter(range(NoiseNum),median3,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.set_xticks(range(NoiseNum))
# ax2.set_xticklabels(LengthToSwipe2)

plt.legend([line1, line2], ['WmSINDy', 'mSINDy'], loc='upper right', ncol = 2, fontsize=30)

offset = 0.4  # Adjust this value as needed

ax1.scatter(np.arange(NoiseNum)-offset, median1, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(NoiseNum)+offset, median3, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5])
plt.ylim(1e-5,10)

ax1.yaxis.grid(True)

plt.tight_layout()

line1.remove()
line2.remove()
plt.savefig("p10l_Id_noise.pdf")

#%%
fig= plt.subplots(figsize=(2,3))
ChangeFontSize(20)
# Labels for the bars
Wm_S_R = np.squeeze(Res_WmSINDy['Rate_success'])[:,Nloop];
m_S_R = np.squeeze(Res_mSINDy['Rate_success'])[:,Nloop];


labels = ['mSINDy', 'WmSINDy']

# Adding labels and title
plt.ylabel('Succes rate %', fontsize=20)
# Heights of the bars
heights = [2*np.sum(m_S_R), 2*np.sum(Wm_S_R)]
bar_width = 0.3  # Adjust this value to change the width

# Plotting bars
plt.bar(labels, heights, color=[color_NSS_SINDy, color_WmS], width=bar_width, edgecolor='black')
plt.yticks([25,50,75,100])
plt.grid(axis='y', color='gray', linestyle='-', alpha=0.7)

plt.savefig("p10c_Id_noise.pdf")


#%%  Upload Identified Noises
N_run = 50; Nloop = 8

Wm_NoiseIDList = Res_WmSINDy['NoiseID']
Wm_NoiseIDList  = Wm_NoiseIDList.reshape(N_run, 3, Nloop, 1000, 2)
# Extract NoiseID for all runs (i=0 to 49), jj=2, and k=7 (last iteration)
Wm_NoiseIDList = Wm_NoiseIDList[:, 2, 7, :, :]

# Extract x, y components
Wm_noise_ids_x = Wm_NoiseIDList[:, :, 0].flatten()
Wm_noise_ids_y = Wm_NoiseIDList[:, :, 1].flatten()

# NoiseID = np.hstack((Wm_noise_ids_x.reshape(-1, 1),Wm_noise_ids_y.reshape(-1, 1)))

m_NoiseIDList = Res_mSINDy['NoiseID']
m_NoiseIDList  = m_NoiseIDList.reshape(N_run, 3, Nloop, 1000, 2)
# Extract NoiseID for all runs (i=0 to 49), jj=2, and k=7 (last iteration)
m_NoiseIDList = m_NoiseIDList[:, 2, 7, :, :]

# Extract x, y components
m_noise_ids_x = m_NoiseIDList[:, :, 0].flatten()
m_noise_ids_y = m_NoiseIDList[:, :, 1].flatten()

# NoiseID = np.hstack((m_noise_ids_x.reshape(-1, 1),m_noise_ids_y.reshape(-1, 1)))

#%%  Get simulated noises
# Define the simulation parameters
p0 = np.array([0.5])

# Define the initial conditions
x0 = np.array([-2,1])

# Define the time points
T=10.0
dt=0.01

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x = odeint(VanderPol,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx = np.transpose(VanderPol(np.transpose(x), 0, p0))             # derivative of the trajectory 

stateVar,dataLen=np.transpose(x).shape

pin = 0
NoisePercentage = 20   # -----> match with: j = NoiseNum - 1

# Initialize noise arrays
noise_x = np.zeros((dataLen, N_run))
noise_y = np.zeros((dataLen, N_run))
noise_z = np.zeros((dataLen, N_run))
        
for i in range(N_run): 
    for j in range(NoiseNum):
        pin=pin+1
        np.random.seed(pin)
        
        #-------------------------- Generate the noise ------------------------
        # These values (NoiseMag: standard deviation) are presented in the manuscript
        NoiseMag = [np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
        
        #----------- Gaussian Noise
        # Noise = np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])    

        #----------- gamma Noise
        for jj in range(stateVar):
            # We want std = NoiseMag[i]
            desired_std = NoiseMag[jj]
            
            # Generate gamma samples with the k value 
            k = 1
            theta = desired_std / np.sqrt(k)  # scale parameter
            
            # Center the distribution (subtract the mean to get zero mean)
            centered_samples = np.random.gamma(k, theta, (dataLen, 1)) - k * theta

            if jj == 0:
                noise_x[:, i] = centered_samples[:,0]
            elif jj == 1:
                noise_y[:, i] = centered_samples[:,0]
            elif jj == 2:
                noise_z[:, i] = centered_samples[:,0]
                
noise_x = noise_x.reshape(-1)
noise_y = noise_y.reshape(-1)
noise_z = noise_z.reshape(-1)

# NoiseID = np.hstack((noise_x.reshape(-1, 1),noise_y.reshape(-1, 1),noise_z.reshape(-1, 1)))

colorNoise='black'
colorNoiseID_Wm='red'
colorNoiseID_mS='cornflowerblue'

lw3=30
lw4=30

plt.figure(figsize=(30,30))
ChangeFontSize(80)

Iter = 0

# Density x
plt.subplot(2,1,1)
plt.grid()
sns.kdeplot(data=noise_x, color=colorNoise, linewidth=lw4,linestyle='-')
sns.kdeplot(data=Wm_noise_ids_x, color=colorNoiseID_Wm,  linewidth=lw3-10, linestyle=':')
sns.kdeplot(data=m_noise_ids_x, color=colorNoiseID_mS,  linewidth=lw3-10, linestyle='dashed')

# Density y
plt.subplot(2,1,2)
plt.grid()
sns.kdeplot(data=noise_y, color=colorNoise, linewidth=lw4, linestyle='-')
sns.kdeplot(data=Wm_noise_ids_y, color=colorNoiseID_Wm, linewidth=lw3, linestyle=':')
sns.kdeplot(data=m_noise_ids_y, color=colorNoiseID_mS,  linewidth=lw3-10, linestyle='dashed')

plt.tight_layout()

plt.savefig("p10r_Id_noise.pdf")