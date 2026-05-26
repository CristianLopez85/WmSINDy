# -*- coding: utf-8 -*-

"""
Created on Sat Jul 11 17:53:17 2020

@author: kadikadi
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
              
#%% Plot the parameter error
Criterion = 4

#---------------  Load the MATLAB .mat file into Python
Res_mSINDy = scipy.io.loadmat('mLorenz_corr.mat') 
Res_WmSINDy = scipy.io.loadmat('WmLorenz_corr.mat')  

Nloop = 5
Noise_level = 10

mNoise_Error = np.squeeze(Res_mSINDy['E_noise'])[:,:,Nloop][:,Noise_level];  
mField_Error = np.squeeze(Res_mSINDy['E_vector_field'])[:,:,Nloop][:,Noise_level];  
m_Sh_Error = np.squeeze(Res_mSINDy['E_short_traj'])[:,:,Nloop][:,Noise_level];  
m_Par_Error = np.squeeze(Res_mSINDy['E_parameter'])[:,:,Nloop][:,Noise_level];  

WmNoise_Error = np.squeeze(Res_WmSINDy['E_noise'])[:,:,Nloop][:,Noise_level];  
WmField_Error = np.squeeze(Res_WmSINDy['E_vector_field'])[:,:,Nloop][:,Noise_level];  
Wm_Sh_Error = np.squeeze(Res_WmSINDy['E_short_traj'])[:,:,Nloop][:,Noise_level];  
Wm_Par_Error = np.squeeze(Res_WmSINDy['E_parameter'])[:,:,Nloop][:,Noise_level];  

mSINDy = np.column_stack([mNoise_Error,mField_Error,m_Sh_Error,m_Par_Error])
WmSINDy = np.column_stack([WmNoise_Error,WmField_Error,Wm_Sh_Error,Wm_Par_Error])

_, median1, _ = np.percentile(WmSINDy, [25, 50, 75], axis=0)
_, median3, _ = np.percentile(mSINDy, [25, 50, 75], axis=0)

#%%  
fig, ax1 = plt.subplots(figsize=(14,5))
ChangeFontSize(30)
# Plotting on the first axis
line1, = ax1.plot(median1,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')

sns.violinplot(data=WmSINDy,cut=0,inner="box",width=width_violin,scale="width",color=color_WmS,linewidth=lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(Criterion),median1,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(Criterion))
ax1.set_xticklabels(LengthToSwipe)
plt.yscale('log')
plt.ylabel('Error', fontsize=30);
# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median3,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=mSINDy,cut=0,inner="box",width=width_violin,scale="width",color=color_NSS_SINDy,linewidth=lw2,saturation=transAlpha,zorder=0, ax=ax2)
ax2.scatter(range(Criterion),median3,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.set_xticks(range(Criterion))
# ax2.set_xticklabels(LengthToSwipe2)

plt.legend([line1, line2], ['WmSINDy', 'mSINDy'], loc='upper right', ncol = 2, fontsize=30)

offset = 0.4  # Adjust this value as needed

ax1.scatter(np.arange(Criterion)-offset, median1, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(Criterion)+offset, median3, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e1,1e0,1e-1,1e-2,1e-3,1e-4])
plt.ylim(3*1e-4,70)

ax1.yaxis.grid(True)

plt.tight_layout()

line1.remove()
line2.remove()
plt.savefig("p10l_noise_corr.pdf")

#%%
fig= plt.subplots(figsize=(2,3))
ChangeFontSize(20)
# Labels for the bars
Wm_S_R = np.squeeze(Res_WmSINDy['Rate_success'])[:,:,Nloop][:,Noise_level];  
m_S_R = np.squeeze(Res_mSINDy['Rate_success'])[:,:,Nloop][:,Noise_level];  


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

plt.savefig("p16r_noise_corr.pdf")

#%%  Upload Identified Noises
NoisePercentageToSwipe=[0,5,10,15,20,25,30,35,40,45,50]   
NoiseNum = len(NoisePercentageToSwipe)

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

#%%  Get simulated noises
def generate_correlated_noise(Q):
    eig_vals, eig_vecs = np.linalg.eigh(Q)
    μ = np.sqrt(np.maximum(eig_vals, 0))  # Ensure non-negative
    r = np.random.randn(len(eig_vals))
    return eig_vecs @ (μ * r)

# Define the simulation parameters
p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])

# Define the time points
T=25.0
dt=0.01  

# Define the initial conditions
x0=np.array([5.0,5.0,25.0])

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)

stateVar,dataLen=np.transpose(x).shape

pin = 0
NoisePercentage = 50  # -----> match with: j = NoiseNum - 1

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
                
        #----------- Correlated Noise
        
        # 1. Define correlation matrix (e.g., 0.5 correlation between variables), defined by CL
        C = np.array([[1,   0.5,  0.3],   # State 1 correlations
                      [0.5, 1,  -0.2],   # State 2 correlations
                      [0.3, -0.2, 1]])   # State 3 correlations
        
        # 2. Compute Q
        Q = np.diag(NoiseMag) @ C @ np.diag(NoiseMag)
        
        # 3. Generate correlated noise
        Noise = np.vstack([generate_correlated_noise(Q) for _ in range(dataLen)])
        
        noise_x[:, i] = Noise[:,0]
        noise_y[:, i] = Noise[:,1]
        noise_z[:, i] = Noise[:,1]
                
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
plt.subplot(3,1,1)
plt.grid()
sns.kdeplot(data=noise_x, color=colorNoise, linewidth=lw4,linestyle='-')
sns.kdeplot(data=Wm_noise_ids_x, color=colorNoiseID_Wm,  linewidth=lw3-10, linestyle=':')
sns.kdeplot(data=m_noise_ids_x, color=colorNoiseID_mS,  linewidth=lw3-10, linestyle='dashed')

# Density y
plt.subplot(3,1,2)
plt.grid()
sns.kdeplot(data=noise_y, color=colorNoise, linewidth=lw4, linestyle='-')
sns.kdeplot(data=Wm_noise_ids_y, color=colorNoiseID_Wm, linewidth=lw3, linestyle=':')
sns.kdeplot(data=m_noise_ids_y, color=colorNoiseID_mS,  linewidth=lw3-10, linestyle='dashed')

# Density z
plt.subplot(3,1,3)
plt.grid()
sns.kdeplot(data=noise_z, color=colorNoise, linewidth=lw4, linestyle='-')
sns.kdeplot(data=Wm_noise_ids_z, color=colorNoiseID_Wm, linewidth=lw3, linestyle=':')
sns.kdeplot(data=m_noise_ids_z, color=colorNoiseID_mS,  linewidth=lw3-10, linestyle='dashed')

plt.tight_layout()

plt.savefig("p16_b.pdf")

#%%
# NoiseMag = [np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
# noise_x1,noise_y1,noise_z1 = np.zeros((dataLen, 1)), np.zeros((dataLen, 1)), np.zeros((dataLen, 1))

# # Generate uniform noise for each dimension
# for jj in range(stateVar):
#     loc = 0
#     c = NoiseMag[jj] * np.sqrt(3.0)
#     Noise_c = loc + np.random.uniform(-c, c, (dataLen, 1))

#     # Stack the samples
#     if jj == 0:
#         noise_x1 = np.expand_dims(Noise_c[:, 0], axis=1)
#     elif jj == 1:
#         noise_y1 = np.expand_dims(Noise_c[:, 0], axis=1)
#     elif jj == 2:
#         noise_z1 = np.expand_dims(Noise_c[:, 0], axis=1)
        
# Noise = np.hstack((noise_x1,noise_y1,noise_z1))
# xn = x + Noise 

# lw=5
# plt.figure(figsize=(20,16))
# pp4=plt.axes(projection='3d')
# pp4.plot3D(x[:,0],x[:,1],x[:,2], color='black',linewidth=lw)
# pp4.plot3D(xn[:,0],xn[:,1],xn[:,2], color='blue',linestyle='--',linewidth=lw)
# pp4.view_init(0, -30)
# pp4.grid(False)
# pp4.axis('off')
# # plt.xlabel('z');plt.ylabel('y');plt.xlabel('x');plt.grid(True)
# plt.savefig('p12_b_noisy.pdf')

#%% Histograms:
# n_bins=int(180/5)
# x_scale=np.linspace(-4*max(NoiseMag), 4*max(NoiseMag), 1000)
  
# plt.figure()
# plt.subplot(3,1,1)
# plt.grid(True)
# pp6=plt.hist(noise_x, bins = n_bins, color = 'blue', alpha=0.9,edgecolor = 'black',density=True)
# pp6=plt.hist(Wm_noise_ids_x, bins = n_bins, color = 'orange', alpha=0.75,edgecolor = 'black',density=True)
# #pp6=plt.plot(x_scale,0.5*Gaussian(x_scale, 0, NoiseMag[0]),color ='black',alpha=0.9,linewidth=2.5)
# plt.ylabel('Frequency')
# plt.xlabel('Noise:x')
# axes = plt.gca()
# axes.set_xlim([-4*max(NoiseMag),4*max(NoiseMag)])
# #axes.set_ylim([0,0.65])

# plt.subplot(3,1,2)
# plt.grid(True)
# pp6=plt.hist(noise_y, bins = n_bins, color = 'blue', alpha=0.9,edgecolor = 'black',density=True)
# pp6=plt.hist(Wm_noise_ids_y, bins = n_bins, color = 'orange', alpha=0.75,edgecolor = 'black',density=True)
# #pp6=plt.plot(x_scale,0.5*Gaussian(x_scale, 0, NoiseMag[1]),color ='black',alpha=0.9,linewidth=2.5)
# plt.ylabel('Frequency')
# plt.xlabel('Noise:y')
# axes = plt.gca()
# axes.set_xlim([-4*max(NoiseMag),4*max(NoiseMag)])
# #axes.set_ylim([0,0.65])

# plt.subplot(3,1,3)
# plt.grid(True)
# pp6=plt.hist(noise_z, bins = n_bins, color = 'blue', alpha=0.9,edgecolor = 'black', density=True)
# pp6=plt.hist(Wm_noise_ids_z, bins = n_bins, color = 'orange', alpha=0.75,edgecolor = 'black', density=True)
# #pp6=plt.plot(x_scale,0.5*Gaussian(x_scale, 0, NoiseMag[2]),color ='black',alpha=0.9,linewidth=2.5)
# plt.ylabel('Frequency')
# plt.xlabel('Noise:z')
# axes = plt.gca()
# axes.set_xlim([-4*max(NoiseMag),4*max(NoiseMag)])
# #axes.set_ylim([0,0.65])
# plt.tight_layout()
