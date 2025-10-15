# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:27:51 2024

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

#%% Load result file

# Define the plot parameters
def ChangeFontSize(fontSize):
    plt.rc('xtick', labelsize=fontSize)    
    plt.rc('ytick', labelsize=fontSize) 
    
    return None

# Define the color
color_WmS=[0.58, 0.44, 0.86]
color_NSS_SINDy='c'
color_WSINDy=[0.5,0.3,0.1]
edge_color="black"

# Define the transparancy
transAlpha=0.95

# Define the line width
lw1=6
lw2=2
lw3=1.5  # 3.5

width_violin=0.20

# Define the lable
labelNSS_SINDy="NSS-SINDy"

# Define the marker properties
MarkerSize_Violin=6
MarkerEdgeWidth=3
MarkerSize=200

# LengthToSwipe=LengthToSwipe[0,:]
LengthToSwipe=['Noise','Vector Field','Short Pred.','Parameter']                          # Added by CL
#%% Plot the parameter error
NoiseNum = 4

import scipy.io
Res_mSINDy = scipy.io.loadmat('m_noAppNoise_Duffing_n35.mat')  # Load the MATLAB .mat file into Python
Res_WmSINDy = scipy.io.loadmat('Wm_noAppNoise_Duffing_n35.mat')  # Load the MATLAB .mat file into Python

Nloop = 4

mNoise_Error = np.squeeze(Res_mSINDy['E_noise'][:,:,Nloop]);           
mvector_field_Error = np.squeeze(Res_mSINDy['E_vector_field'][:,:,Nloop]);   
mshort_traj_Error = np.squeeze(Res_mSINDy['E_short_traj'][:,:,Nloop]);         
mparameter_Error = np.squeeze(Res_mSINDy['E_parameter'][:,:,Nloop]);

WmNoise_Error = np.squeeze(Res_WmSINDy['E_noise'][:,:,Nloop]);           
Wmvector_field_Error = np.squeeze(Res_WmSINDy['E_vector_field'][:,:,Nloop]);   
Wmshort_traj_Error = np.squeeze(Res_WmSINDy['E_short_traj'][:,:,Nloop]);         
Wmparameter_Error = np.squeeze(Res_WmSINDy['E_parameter'][:,:,Nloop]);

mNoise_Error = np.column_stack((mNoise_Error, mvector_field_Error,mshort_traj_Error,mparameter_Error))
WmNoise_Error = np.column_stack((WmNoise_Error, Wmvector_field_Error,Wmshort_traj_Error,Wmparameter_Error))

quartile3a, median3, quartile3b = np.percentile(mNoise_Error, [25, 50, 75], axis=0)
quartile1a, median1, quartile1b = np.percentile(WmNoise_Error, [25, 50, 75], axis=0)

#%%  
fig, ax1 = plt.subplots(figsize=(14,5))
ChangeFontSize(30)
# Plotting on the first axis
line1, = ax1.plot(median1,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')

sns.violinplot(data=WmNoise_Error,cut=0,inner="box",width=width_violin,scale="width",color=color_WmS,linewidth=lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(NoiseNum),median1,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(NoiseNum))
ax1.set_xticklabels(LengthToSwipe)
plt.yscale('log')
plt.ylabel('Error', fontsize=30);
# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median3,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=mNoise_Error,cut=0,inner="box",width=width_violin,scale="width",color=color_NSS_SINDy,linewidth=lw2,saturation=transAlpha,zorder=0, ax=ax2)
ax2.scatter(range(NoiseNum),median3,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.set_xticks(range(NoiseNum))
# ax2.set_xticklabels(LengthToSwipe2)

plt.legend([line1, line2], ['WmSINDy', 'mSINDy'], loc='lower left', fontsize=30)

offset = 0.4  # Adjust this value as needed

ax1.scatter(np.arange(NoiseNum)-offset, median1, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(NoiseNum)+offset, median3, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e0,1e-1,1e-2,1e-3,1e-4])

ax1.yaxis.grid(True)

plt.tight_layout()

line1.remove()
line2.remove()

plt.savefig("p7a_l.pdf")
#%%
fig= plt.subplots(figsize=(2,3))
ChangeFontSize(20)
# Labels for the bars
labels = ['mSINDy', 'WmSINDy']

# Adding labels and title
plt.ylabel('Succes rate %', fontsize=20)
# Heights of the bars
mS_R = np.sum(np.squeeze(Res_mSINDy['Rate_success'][:,:,Nloop]))
WmS_R = np.sum(np.squeeze(Res_WmSINDy['Rate_success'][:,:,Nloop]))
heights = [2*mS_R, 2*WmS_R]

bar_width = 0.3  # Adjust this value to change the width

# Plotting bars
plt.bar(labels, heights, color=[color_NSS_SINDy, color_WmS], width=bar_width, edgecolor='black')

plt.yticks([25,50,75,100])

plt.savefig("p7a_r.pdf")
# Display the plot
# plt.show()