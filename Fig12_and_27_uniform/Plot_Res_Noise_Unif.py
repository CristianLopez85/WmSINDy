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

NoisePercentageToSwipe=[0,5,10,15,20,25,30,35,40,45,50]        
              
#%% Plot the parameter error
NoiseNum = len(NoisePercentageToSwipe)

#---------------  Load the MATLAB .mat file into Python
Res_mSINDy = scipy.io.loadmat('mLorenz_Uniform.mat') 
Res_WmSINDy = scipy.io.loadmat('WmLorenz_Uniform.mat')  
Res_WSINDy = scipy.io.loadmat('WLorenz_Uniform.mat')

Nloop = 5

WmNoise_Error = Res_WmSINDy['E_noise'][:,:,Nloop]
mNoise_Error = Res_mSINDy['E_noise'][:,:,Nloop]

quartile1a, median1, quartile1b = np.percentile(WmNoise_Error, [25, 50, 75], axis=0)
quartile3a, median3, quartile3b = np.percentile(mNoise_Error, [25, 50, 75], axis=0)

fig, ax1 = plt.subplots(figsize=(14,5))
# Plotting on the first axis
line1, = ax1.plot(median1,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')

sns.violinplot(data=WmNoise_Error,cut=0,inner="box",width=width_violin,scale="width",color=color_WmS,linewidth=lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(NoiseNum),median1,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(NoiseNum))
ax1.set_xticklabels(NoisePercentageToSwipe)

plt.yscale('log')
plt.ylabel('Noise ID Error', fontsize=24);plt.xlabel('Noise Level (%)', fontsize=24);
# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median3,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')

sns.violinplot(data=mNoise_Error,cut=0,inner="box",width=width_violin,scale="width",color=color_NSS_SINDy,linewidth=lw2,saturation=transAlpha,zorder=0, ax=ax2)

ax2.scatter(range(NoiseNum),median3,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_xticklabels([])  # Remove the labels for the top x-axis

offset = 0.45  # Adjust this value as needed

ax1.scatter(np.arange(NoiseNum)-offset, median1, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(NoiseNum)+offset, median3, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e0,1e-2,1e-4])

ax1.yaxis.grid(True)

plt.savefig("p26a.pdf")

#%%
WmField_Error = Res_WmSINDy['E_vector_field'][:,:,Nloop]
mField_Error = Res_mSINDy['E_vector_field'][:,:,Nloop]
WVector_Field_Error = Res_WSINDy['E_vector_field']

quartile2a, median2, quartile2b = np.percentile(WmField_Error, [25, 50, 75], axis=0)
quartile4a, median4, quartile4b = np.percentile(mField_Error, [25, 50, 75], axis=0)
quartile5a, median5, quartile5b = np.percentile(WVector_Field_Error, [25, 50, 75], axis=0)

fig, ax1 = plt.subplots(figsize=(14,5))

comp_f = 0.5

# Plotting on the first axis
line1, = ax1.plot(median2,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WmField_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WmS,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(NoiseNum),median2,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(NoiseNum))
ax1.set_xticklabels(NoisePercentageToSwipe)
plt.yscale('log')
plt.ylabel('Vector Field Error', fontsize=24);plt.xlabel('Noise Level (%)', fontsize=24);

# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median4,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=mField_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_NSS_SINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax2)
ax2.scatter(range(NoiseNum),median4,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.set_xticks(range(NoiseNum))

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_xticklabels([])  # Remove the labels for the top x-axis

# Creating a second axis
ax3 = ax1.twiny()
# Plotting on the third axis
line3, =  ax3.plot(median5,linewidth=lw1,color=color_WSINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WVector_Field_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WSINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax3)
ax3.scatter(range(NoiseNum),median5,s=MarkerSize,marker='o',color=color_WSINDy,edgecolors=edge_color,linewidth=lw3)
ax3.set_xticks(range(NoiseNum))

ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.set_xticklabels([])  # Remove the labels for the top x-axis

ax3.xaxis.grid(True)

# plt.legend([line1, line2, line3], ['WmSINDy', 'mSINDy', 'WSINDy'], loc='lower right', fontsize=24, ncol=3)

offset = 0.45  # Adjust this value as needed

ax1.scatter(np.arange(NoiseNum)-offset, median2, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(NoiseNum)+offset, median4, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(NoiseNum)+0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(NoiseNum)-0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e-1,1e-3,1e-5,1e-7])
ax1.yaxis.grid(True)
plt.ylim([1e-7, 1e-1])   # plt.ylim([1e-4, 1e0])
plt.savefig("p26b.pdf")

#%%
WmShort_Pred_Error = Res_WmSINDy['E_short_traj'][:,:,Nloop]
mShort_Pred_Error = Res_mSINDy['E_short_traj'][:,:,Nloop]
WShort_Pred_Error = Res_WSINDy['E_short']

quartile1a, median2, quartile1b = np.percentile(WmShort_Pred_Error, [25, 50, 75], axis=0)
quartile3a, median4, quartile3b = np.percentile(mShort_Pred_Error, [25, 50, 75], axis=0)
quartile5a, median5, quartile5b = np.percentile(WShort_Pred_Error, [25, 50, 75], axis=0)

fig, ax1 = plt.subplots(figsize=(14,5))

comp_f = 0.5

# Plotting on the first axis
line1, = ax1.plot(median2,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WmShort_Pred_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WmS,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(NoiseNum),median2,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(NoiseNum))
ax1.set_xticklabels(NoisePercentageToSwipe)
plt.yscale('log')
plt.ylabel('Short Prediction Error', fontsize=24);plt.xlabel('Noise Level (%)', fontsize=24);

# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median4,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=mShort_Pred_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_NSS_SINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax2)
ax2.scatter(range(NoiseNum),median4,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.set_xticks(range(NoiseNum))

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_xticklabels([])  # Remove the labels for the top x-axis

# Creating a second axis
ax3 = ax1.twiny()
# Plotting on the third axis
line3, =  ax3.plot(median5,linewidth=lw1,color=color_WSINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WShort_Pred_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WSINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax3)
ax3.scatter(range(NoiseNum),median5,s=MarkerSize,marker='o',color=color_WSINDy,edgecolors=edge_color,linewidth=lw3)
ax3.set_xticks(range(NoiseNum))

ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.set_xticklabels([])  # Remove the labels for the top x-axis

ax3.xaxis.grid(True)

# plt.legend([line1, line2, line3], ['WmSINDy', 'mSINDy', 'WSINDy'], loc='lower right', fontsize=24, ncol=3)

offset = 0.45  # Adjust this value as needed

ax1.scatter(np.arange(NoiseNum)-offset, median2, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(NoiseNum)+offset, median4, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(NoiseNum)+0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(NoiseNum)-0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e-1,1e-3,1e-5,1e-7])
# plt.grid()
ax1.yaxis.grid(True)
plt.savefig("p26c.pdf")

#%%
WmParameter_Error = Res_WmSINDy['E_parameter'][:,:,Nloop]    
mParameter_Error = Res_mSINDy['E_parameter'][:,:,Nloop]    
WParameter_Error = Res_WSINDy['E_parameter'];      

quartile2a, median2, quartile2b = np.percentile(WmParameter_Error, [25, 50, 75], axis=0)
quartile4a, median4, quartile4b = np.percentile(mParameter_Error, [25, 50, 75], axis=0)
quartile5a, median5, quartile5b = np.percentile(WParameter_Error, [25, 50, 75], axis=0)

fig, ax1 = plt.subplots(figsize=(14,5))

comp_f = 0.5

# Plotting on the first axis
line1, = ax1.plot(median2,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WmParameter_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WmS,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(NoiseNum),median2,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(NoiseNum))
ax1.set_xticklabels(NoisePercentageToSwipe)
plt.yscale('log')
plt.ylabel('Parameter Error', fontsize=24);plt.xlabel('Noise Level (%)', fontsize=24);

# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median4,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=mParameter_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_NSS_SINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax2)
ax2.scatter(range(NoiseNum),median4,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.set_xticks(range(NoiseNum))

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_xticklabels([])  # Remove the labels for the top x-axis

# Creating a second axis
ax3 = ax1.twiny()
# Plotting on the third axis
line3, =  ax3.plot(median5,linewidth=lw1,color=color_WSINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WParameter_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WSINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax3)
ax3.scatter(range(NoiseNum),median5,s=MarkerSize,marker='o',color=color_WSINDy,edgecolors=edge_color,linewidth=lw3)
ax3.set_xticks(range(NoiseNum))

ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.set_xticklabels([])  # Remove the labels for the top x-axis

ax3.xaxis.grid(True)

offset = 0.45  # Adjust this value as needed

ax1.scatter(np.arange(NoiseNum)-offset, median2, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(NoiseNum)+offset, median4, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(NoiseNum)+0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(NoiseNum)-0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e1,1e0,1e-1,1e-2,1e-3,1e-4])
ax1.yaxis.grid(True)
plt.ylim([2*1e-4, 2])   # plt.ylim([1e-4, 1e0])
plt.savefig("p26d.pdf")

#%%
WmR_o_S_Error = Res_WmSINDy['Rate_success'][:,:,Nloop]  ; WmR_o_S_Error = 2.0*np.sum(WmR_o_S_Error,axis=0)  
mR_o_S_Error = Res_mSINDy['Rate_success'][:,:,Nloop]    ; mR_o_S_Error = 2.0*np.sum(mR_o_S_Error,axis=0) 
WSR_o_S_Error = Res_WSINDy['Rate_success'];     WSR_o_S_Error = 2.0*np.sum(WSR_o_S_Error,axis=0)

plt.figure(figsize=(14,5))
# Cahnge the Font size
ChangeFontSize(24)

# Plot the line
plt.plot(WmR_o_S_Error,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')
plt.plot(mR_o_S_Error,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
plt.plot(WSR_o_S_Error,linewidth=lw1,color=color_WSINDy,alpha=0.6,zorder=0,linestyle='solid')

# Plot the scatter
plt.scatter(range(NoiseNum),WmR_o_S_Error,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
plt.scatter(range(NoiseNum),mR_o_S_Error,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
plt.scatter(range(NoiseNum),WSR_o_S_Error,s=MarkerSize,marker='o',color=color_WSINDy,edgecolors=edge_color,linewidth=lw3)
# plt.scatter(range(LamdasNum),median5,s=MarkerSize,marker='o',color=color_WSINDy,edgecolors=edge_color,linewidth=lw3)

plt.xticks(range(NoiseNum),NoisePercentageToSwipe)

# plt.yscale('log')
plt.yticks([100,75,50,25,0])
plt.grid()
plt.ylim([-5, 105])

# plt.ylabel('Parameter Error', fontsize=24);plt.xlabel(r'$\lambda$', fontsize=24);plt.legend(['WmSINDy', 'mSINDy', 'WSINDy'],loc='lower right', fontsize=24)
plt.ylabel('Success Rate (%)', fontsize=24);plt.xlabel('Noise Level (%)', fontsize=24);plt.legend(['WmSINDy', 'mSINDy', 'WSINDy'],loc='upper right', fontsize=24)
plt.savefig("p12_a.pdf")


#%%  Upload Identified Noises
N_run = 50

Wm_NoiseIDList = Res_WmSINDy['NoiseID']
Wm_NoiseIDList = Wm_NoiseIDList.reshape(N_run, NoiseNum, 6, 2500, 3)

# 10 for 50% noise, 5 for last Nloop
Wm_noise_ids_at_50_percent_last_k = Wm_NoiseIDList[:, 10, 5, :, :]

# Extract x component 
Wm_noise_ids_x = Wm_noise_ids_at_50_percent_last_k[:, :, 0].flatten()
Wm_noise_ids_y = Wm_noise_ids_at_50_percent_last_k[:, :, 1].flatten()
Wm_noise_ids_z = Wm_noise_ids_at_50_percent_last_k[:, :, 2].flatten()

# NoiseID = np.hstack((Wm_noise_ids_x.reshape(-1, 1),Wm_noise_ids_y.reshape(-1, 1),Wm_noise_ids_z.reshape(-1, 1)))

m_NoiseIDList = Res_mSINDy['NoiseID']
m_NoiseIDList = m_NoiseIDList.reshape(N_run, NoiseNum, 6, 2500, 3)

# 10 for 50% noise, 5 for last Nloop
m_noise_ids_at_50_percent_last_k = m_NoiseIDList[:, 10, 5, :, :]

# Extract x component 
m_noise_ids_x = m_noise_ids_at_50_percent_last_k[:, :, 0].flatten()
m_noise_ids_y = m_noise_ids_at_50_percent_last_k[:, :, 1].flatten()
m_noise_ids_z = m_noise_ids_at_50_percent_last_k[:, :, 2].flatten()

# NoiseID = np.hstack((m_noise_ids_x.reshape(-1, 1),m_noise_ids_y.reshape(-1, 1),m_noise_ids_z.reshape(-1, 1)))

#%%  Get simulated noises
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
NoisePercentage = 50   # -----> match with: j = NoiseNum - 1

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
        
        # Generate uniform noise for each dimension
        for jj in range(stateVar):
            loc = 0
            c = NoiseMag[jj] * np.sqrt(3.0)
            Noise_c = loc + np.random.uniform(-c, c, (dataLen, 1))

            # Stack the samples
            if jj == 0:
                noise_x[:, i] = Noise_c[:, 0]
            elif jj == 1:
                noise_y[:, i] = Noise_c[:, 0]
            elif jj == 2:
                noise_z[:, i] = Noise_c[:, 0]
                
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

plt.savefig("p12_b.pdf")

#%%
NoiseMag = [np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
noise_x1,noise_y1,noise_z1 = np.zeros((dataLen, 1)), np.zeros((dataLen, 1)), np.zeros((dataLen, 1))

# Generate uniform noise for each dimension
for jj in range(stateVar):
    loc = 0
    c = NoiseMag[jj] * np.sqrt(3.0)
    Noise_c = loc + np.random.uniform(-c, c, (dataLen, 1))

    # Stack the samples
    if jj == 0:
        noise_x1 = np.expand_dims(Noise_c[:, 0], axis=1)
    elif jj == 1:
        noise_y1 = np.expand_dims(Noise_c[:, 0], axis=1)
    elif jj == 2:
        noise_z1 = np.expand_dims(Noise_c[:, 0], axis=1)
        
Noise = np.hstack((noise_x1,noise_y1,noise_z1))
xn = x + Noise 

lw=5
plt.figure(figsize=(20,16))
pp4=plt.axes(projection='3d')
pp4.plot3D(x[:,0],x[:,1],x[:,2], color='black',linewidth=lw)
pp4.plot3D(xn[:,0],xn[:,1],xn[:,2], color='blue',linestyle='--',linewidth=lw)
pp4.view_init(0, -30)
pp4.grid(False)
pp4.axis('off')
# plt.xlabel('z');plt.ylabel('y');plt.xlabel('x');plt.grid(True)
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
