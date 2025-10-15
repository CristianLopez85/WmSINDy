# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:15:58 2025

@author: crist
"""

# =============================================================================
# This file will plot the merged result of swiping the noise level
# =============================================================================
#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
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

# lambdaToSwipe=lambdaToSwipe[0,:]
# lambdaToSwipe=[0,2,4,6,8,10,12,14,16,18,20]                           # Added by CL
lambdaToSwipe=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1]                        # Added by CL
#%% Plot the parameter error
NoiseNum = len(lambdaToSwipe)

Res_mSINDy = scipy.io.loadmat('m_noAppNoise_CF.mat')  # Load the MATLAB .mat file into Python

Res_WmSINDy = scipy.io.loadmat('Wm_CF.mat')  # Load the MATLAB .mat file into Python

Nloop = 5

WmNoise_Error = Res_WmSINDy['E_noise'];         WmNoise_Error = WmNoise_Error[:,:,Nloop]
mNoise_Error = Res_mSINDy['E_noise'];           mNoise_Error = mNoise_Error[:,:,Nloop]

quartile1a, median1, quartile1b = np.percentile(WmNoise_Error, [25, 50, 75], axis=0)
quartile3a, median3, quartile3b = np.percentile(mNoise_Error, [25, 50, 75], axis=0)

fig, ax1 = plt.subplots(figsize=(14,5))
# Plotting on the first axis
line1, = ax1.plot(median1,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')

sns.violinplot(data=WmNoise_Error,cut=0,inner="box",width=width_violin,scale="width",color=color_WmS,linewidth=lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(NoiseNum),median1,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(NoiseNum))
ax1.set_xticklabels(lambdaToSwipe)

plt.yscale('log')
plt.ylabel('Noise ID Error', fontsize=24);plt.xlabel('lambda', fontsize=24);
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
plt.yticks([1e2,1e1,1e0,1e-1])

ax1.yaxis.grid(True)

plt.savefig("p4a.pdf")

#%%
WmField_Error = Res_WmSINDy['E_vector_field'];  WmField_Error = WmField_Error[:,:,Nloop]
mField_Error = Res_mSINDy['E_vector_field'];    mField_Error = mField_Error[:,:,Nloop]

quartile2a, median2, quartile2b = np.percentile(WmField_Error, [25, 50, 75], axis=0)
quartile4a, median4, quartile4b = np.percentile(mField_Error, [25, 50, 75], axis=0)

fig, ax1 = plt.subplots(figsize=(14,5))

comp_f = 0.5

# Plotting on the first axis
line1, = ax1.plot(median2,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WmField_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WmS,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(NoiseNum),median2,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(NoiseNum))
ax1.set_xticklabels(lambdaToSwipe)
plt.yscale('log')
plt.ylabel('Vector Field Error', fontsize=24);plt.xlabel('lambda', fontsize=24);

# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median4,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=mField_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_NSS_SINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax2)
ax2.scatter(range(NoiseNum),median4,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.set_xticks(range(NoiseNum))

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_xticklabels([])  # Remove the labels for the top x-axis

# plt.legend([line1, line2, line3], ['WmSINDy', 'mSINDy', 'WSINDy'], loc='lower right', fontsize=24, ncol=3)

offset = 0.45  # Adjust this value as needed

ax1.scatter(np.arange(NoiseNum)-offset, median2, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(NoiseNum)+offset, median4, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e2,1e1,1,1e-1,1e-2,1e-3,1e-4])
ax1.yaxis.grid(True)
# plt.ylim([1e-7, 20])   # plt.ylim([1e-4, 1e0])
plt.savefig("p4c.pdf")

#%%
WmParameter_Error = Res_WmSINDy['E_parameter'];    WmParameter_Error = WmParameter_Error[:,:,Nloop]    
mParameter_Error = Res_mSINDy['E_parameter'];      mParameter_Error = mParameter_Error[:,:,Nloop]    

quartile2a, median2, quartile2b = np.percentile(WmParameter_Error, [25, 50, 75], axis=0)
quartile4a, median4, quartile4b = np.percentile(mParameter_Error, [25, 50, 75], axis=0)

fig, ax1 = plt.subplots(figsize=(14,5))

comp_f = 0.5

# Plotting on the first axis
line1, = ax1.plot(median2,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WmParameter_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WmS,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(NoiseNum),median2,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(NoiseNum))
ax1.set_xticklabels(lambdaToSwipe)
plt.yscale('log')
plt.ylabel('Parameter Error', fontsize=24);plt.xlabel('lambda', fontsize=24);

# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median4,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=mParameter_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_NSS_SINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax2)
ax2.scatter(range(NoiseNum),median4,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.set_xticks(range(NoiseNum))

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_xticklabels([])  # Remove the labels for the top x-axis

offset = 0.45  # Adjust this value as needed

ax1.scatter(np.arange(NoiseNum)-offset, median2, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(NoiseNum)+offset, median4, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e1,1e0,1e-1,1e-2,1e-3,1e-4])
ax1.yaxis.grid(True)
plt.ylim([3e-3, 10])   # plt.ylim([1e-4, 1e0])
plt.savefig("p4b.pdf")

#%%
WmR_o_S_Error = Res_WmSINDy['Rate_success'][:,:,Nloop]  ; WmR_o_S_Error = 2.0*np.sum(WmR_o_S_Error,axis=0)  
mR_o_S_Error = Res_mSINDy['Rate_success'][:,:,Nloop]    ; mR_o_S_Error = 2.0*np.sum(mR_o_S_Error,axis=0) 

plt.figure(figsize=(14,5))
# Cahnge the Font size
ChangeFontSize(24)

# Plot the line
plt.plot(WmR_o_S_Error,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')
plt.plot(mR_o_S_Error,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')

# Plot the scatter
plt.scatter(range(NoiseNum),WmR_o_S_Error,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
plt.scatter(range(NoiseNum),mR_o_S_Error,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
# plt.scatter(range(LamdasNum),median5,s=MarkerSize,marker='o',color=color_WSINDy,edgecolors=edge_color,linewidth=lw3)

plt.xticks(range(NoiseNum),lambdaToSwipe)

# plt.yscale('log')
plt.yticks([100,75,50,25,0])
plt.grid()
plt.ylim([-5, 105])

# plt.ylabel('Parameter Error', fontsize=24);plt.xlabel(r'$\lambda$', fontsize=24);plt.legend(['WmSINDy', 'mSINDy', 'WSINDy'],loc='lower right', fontsize=24)
plt.ylabel('Success Rate (%)', fontsize=24);plt.xlabel('lambda', fontsize=24);plt.legend(['WmSINDy', 'mSINDy'],loc='upper right', fontsize=24)
plt.savefig("p4d.pdf")
