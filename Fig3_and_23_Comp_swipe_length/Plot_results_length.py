# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:08:19 2025

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

# LengthToSwipe=LengthToSwipe[0,:]
# LengthToSwipe=[0,2,4,6,8,10,12,14,16,18,20]                           # Added by CL
LengthToSwipe=[5,7.5,10,12.5,15,17.5,20,22.5,25]                        # Added by CL
#%% Plot the parameter error
LengthNum = len(LengthToSwipe)

Res_mSINDy = scipy.io.loadmat('mLorenz_length_Exp50_q3_l02_noAppN.mat')  # Load the MATLAB .mat file into Python
Res_WmSINDy = scipy.io.loadmat('WmLorenz_length_Exp50_q3_l02_noAppN.mat')  # Load the MATLAB .mat file into Python

Res_WSINDy = scipy.io.loadmat('WLorenz_Length_Exp50_fullqp.mat')  # Load the MATLAB .mat file into Python
Nloop = 5

WmNoise_Error = Res_WmSINDy['E_noise'];         WmNoise_Error = WmNoise_Error[:,:,Nloop]
mNoise_Error = Res_mSINDy['E_noise'];           mNoise_Error = mNoise_Error[:,:,Nloop]

quartile1a, median1, quartile1b = np.percentile(WmNoise_Error, [25, 50, 75], axis=0)
quartile3a, median3, quartile3b = np.percentile(mNoise_Error, [25, 50, 75], axis=0)

fig, ax1 = plt.subplots(figsize=(14,5))
# Plotting on the first axis
line1, = ax1.plot(median1,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')

sns.violinplot(data=WmNoise_Error,cut=0,inner="box",width=width_violin,scale="width",color=color_WmS,linewidth=lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(LengthNum),median1,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(LengthNum))
ax1.set_xticklabels(LengthToSwipe)

plt.yscale('log')
plt.ylabel('Noise ID Error', fontsize=24);plt.xlabel('Training Data Length', fontsize=24);
# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median3,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')

sns.violinplot(data=mNoise_Error,cut=0,inner="box",width=width_violin,scale="width",color=color_NSS_SINDy,linewidth=lw2,saturation=transAlpha,zorder=0, ax=ax2)

ax2.scatter(range(LengthNum),median3,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_xticklabels([])  # Remove the labels for the top x-axis

offset = 0.45  # Adjust this value as needed

ax1.scatter(np.arange(LengthNum)-offset, median1, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(LengthNum)+offset, median3, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e0,1e-1,1e-2])
plt.ylim([.05, 1e2])   # pl

ax1.yaxis.grid(True)

plt.savefig("p3a.pdf")

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
ax1.scatter(range(LengthNum),median2,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(LengthNum))
ax1.set_xticklabels(LengthToSwipe)
plt.yscale('log')
plt.ylabel('Vector Field Error', fontsize=24);plt.xlabel('Training Data Length', fontsize=24);

# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median4,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=mField_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_NSS_SINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax2)
ax2.scatter(range(LengthNum),median4,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.set_xticks(range(LengthNum))

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_xticklabels([])  # Remove the labels for the top x-axis

# Creating a second axis
ax3 = ax1.twiny()
# Plotting on the third axis
line3, =  ax3.plot(median5,linewidth=lw1,color=color_WSINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WVector_Field_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WSINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax3)
ax3.scatter(range(LengthNum),median5,s=MarkerSize,marker='o',color=color_WSINDy,edgecolors=edge_color,linewidth=lw3)
ax3.set_xticks(range(LengthNum))

ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.set_xticklabels([])  # Remove the labels for the top x-axis

ax3.xaxis.grid(True)

# plt.legend([line1, line2, line3], ['WmSINDy', 'mSINDy', 'WSINDy'], loc='lower right', fontsize=24, ncol=3)

offset = 0.45  # Adjust this value as needed

ax1.scatter(np.arange(LengthNum)-offset, median2, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(LengthNum)+offset, median4, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(LengthNum)+0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(LengthNum)-0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e1,1e-1,1e-3,1e-5,1e-7])
ax1.yaxis.grid(True)
plt.ylim([1e-4, 10])   # plt.ylim([1e-4, 1e0])
plt.savefig("p3b.pdf")

#%%
WmShort_Pred_Error = Res_WmSINDy['E_short_traj'][:,:,Nloop]
mShort_Pred_Error = Res_mSINDy['E_short_traj'][:,:,Nloop]
WShort_Pred_Error = Res_WSINDy['E_short']

quartile1a, median1, quartile1b = np.percentile(WmShort_Pred_Error, [25, 50, 75], axis=0)
quartile3a, median3, quartile3b = np.percentile(mShort_Pred_Error, [25, 50, 75], axis=0)
quartile3a, median5, quartile3b = np.percentile(WShort_Pred_Error, [25, 50, 75], axis=0)

fig, ax1 = plt.subplots(figsize=(14,5))

comp_f = 0.5

# Plotting on the first axis
line1, = ax1.plot(median1,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WmShort_Pred_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WmS,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax1)
ax1.scatter(range(LengthNum),median1,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(LengthNum))
ax1.set_xticklabels(LengthToSwipe)
plt.yscale('log')
plt.ylabel('Short Prediction Error', fontsize=24);plt.xlabel('Training Data Length', fontsize=24);

# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median3,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=mShort_Pred_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_NSS_SINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax2)
ax2.scatter(range(LengthNum),median3,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.set_xticks(range(LengthNum))

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_xticklabels([])  # Remove the labels for the top x-axis

# Creating a second axis
ax3 = ax1.twiny()
# Plotting on the third axis
line3, =  ax3.plot(median5,linewidth=lw1,color=color_WSINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WShort_Pred_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WSINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax3)
ax3.scatter(range(LengthNum),median5,s=MarkerSize,marker='o',color=color_WSINDy,edgecolors=edge_color,linewidth=lw3)
ax3.set_xticks(range(LengthNum))

ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.set_xticklabels([])  # Remove the labels for the top x-axis

ax3.xaxis.grid(True)

offset = 0.45  # Adjust this value as needed

ax1.scatter(np.arange(LengthNum)-offset, median1, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(LengthNum)+offset, median3, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(LengthNum)+0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(LengthNum)-0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e-1,1e-2,1e-3,1e-4])
# plt.grid()
ax1.yaxis.grid(True)
plt.savefig("p3c.pdf")

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
ax1.scatter(range(LengthNum),median2,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
ax1.set_xticks(range(LengthNum))
ax1.set_xticklabels(LengthToSwipe)
plt.yscale('log')
plt.ylabel('Parameter Error', fontsize=24);plt.xlabel('Training Data Length', fontsize=24);

# Creating a second axis
ax2 = ax1.twiny()

# Plotting on the second axis
line2, =  ax2.plot(median4,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=mParameter_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_NSS_SINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax2)
ax2.scatter(range(LengthNum),median4,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
ax2.set_xticks(range(LengthNum))

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_xticklabels([])  # Remove the labels for the top x-axis

# Creating a second axis
ax3 = ax1.twiny()
# Plotting on the third axis
line3, =  ax3.plot(median5,linewidth=lw1,color=color_WSINDy,alpha=0.6,zorder=0,linestyle='solid')
sns.violinplot(data=WParameter_Error,cut=0,inner="box",width=comp_f*width_violin,scale="width",color=color_WSINDy,linewidth=comp_f*lw2,saturation=transAlpha,zorder=0, ax=ax3)
ax3.scatter(range(LengthNum),median5,s=MarkerSize,marker='o',color=color_WSINDy,edgecolors=edge_color,linewidth=lw3)
ax3.set_xticks(range(LengthNum))

ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.set_xticklabels([])  # Remove the labels for the top x-axis

ax3.xaxis.grid(True)

offset = 0.45  # Adjust this value as needed

ax1.scatter(np.arange(LengthNum)-offset, median2, s=MarkerSize, marker='o', color='none', edgecolors='none')
ax2.plot(np.arange(LengthNum)+offset, median4, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(LengthNum)+0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')
ax3.plot(np.arange(LengthNum)-0.23, median5, color='none', alpha=0.6, zorder=0, linestyle='solid')

plt.yscale('log')
plt.yticks([1e1,1e0,1e-1,1e-2,1e-3,1e-4])
ax1.yaxis.grid(True)
plt.ylim([2*1e-3, 5])   # plt.ylim([1e-4, 1e0])
plt.savefig("p3d.pdf")

#%%
WmR_o_S_Error = Res_WmSINDy['Rate_success'][:,:,Nloop]  ; WmR_o_S_Error = 2.0*np.sum(WmR_o_S_Error,axis=0)  
mR_o_S_Error = Res_mSINDy['Rate_success'][:,:,Nloop]    ; mR_o_S_Error = 2.0*np.sum(mR_o_S_Error,axis=0) 
WSR_o_S_Error = Res_WSINDy['Rate_success'];     WSR_o_S_Error = 2.0*np.sum(WSR_o_S_Error,axis=0)

# WmShort_Pred_Error = Res_WmSINDy['Short_Pred_Error']
# WmR_o_S_Error = Res_WmSINDy['Rate_of_Success']
# WmR_o_S_Error = WmR_o_S_Error.reshape((11,))

# # mShort_Pred_Error = Res_mSINDy['Short_Pred_Error']
# mR_o_S_Error = Res_mSINDy['Rate_of_Success']
# mR_o_S_Error = mR_o_S_Error.reshape((11,))

# # Res_WSINDy = scipy.io.loadmat('WSINDy50_k2500_tauhatm2.mat')  # Load the MATLAB .mat file into Python
# WSR_o_S_Error = Res_WSINDy['Rate_of_Success']
# WSR_o_S_Error = WSR_o_S_Error.reshape((11,))

plt.figure(figsize=(14,5))
# Cahnge the Font size
ChangeFontSize(24)

# Plot the line
plt.plot(WmR_o_S_Error,linewidth=lw1,color=color_WmS,alpha=0.6,zorder=0,linestyle='solid')
plt.plot(mR_o_S_Error,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')
plt.plot(WSR_o_S_Error,linewidth=lw1,color=color_WSINDy,alpha=0.6,zorder=0,linestyle='solid')

# Plot the scatter
plt.scatter(range(LengthNum),WmR_o_S_Error,s=MarkerSize,marker='o',color=color_WmS,edgecolors=edge_color,linewidth=lw3)
plt.scatter(range(LengthNum),mR_o_S_Error,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)
plt.scatter(range(LengthNum),WSR_o_S_Error,s=MarkerSize,marker='o',color=color_WSINDy,edgecolors=edge_color,linewidth=lw3)
# plt.scatter(range(LamdasNum),median5,s=MarkerSize,marker='o',color=color_WSINDy,edgecolors=edge_color,linewidth=lw3)

plt.xticks(range(LengthNum),LengthToSwipe)

# plt.yscale('log')
plt.yticks([100,75,50,25,0])
plt.grid()
plt.ylim([-5, 105])

# plt.ylabel('Parameter Error', fontsize=24);plt.xlabel(r'$\lambda$', fontsize=24);plt.legend(['WmSINDy', 'mSINDy', 'WSINDy'],loc='lower right', fontsize=24)
plt.ylabel('Success Rate (%)', fontsize=24);plt.xlabel('Training Data Length', fontsize=24);plt.legend(['WmSINDy', 'mSINDy', 'WSINDy'],loc='upper right', fontsize=24)
plt.savefig("p3e.pdf")

