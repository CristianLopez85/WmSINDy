# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:25:05 2023

@author: crist
"""
# =============================================================================
# The following code will swipe the effect of noise level on using SINDy to perform noise signal speration 
# We set T=25, dt=0.01, q=10, x0=[-5.0,5.0,25.0], ro=0.9.
# The SINDy library has 9 nonlinear features. We also perform the loop multiple times to record the performance of SINDy.
# =============================================================================

#%% Import packages
import numpy as np
from scipy.integrate import odeint
from scipy.stats import dweibull
import tensorflow as tf
import matplotlib.pyplot as plt
from utils_NSS_SINDy_W import *
import time
# import tensorflow_probability as tfp
from datetime import datetime
import os
import importlib

import math
from utils_WSINDy import *
from scipy.io import savemat

#%% Define some parameters to swipe the different noise level
# Define how mant times you would like to run each noise level
N_run = 50

# Define the simulation parameters
p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])

# Define the initial conditions
#x0=np.array([-5.0,5.0,25.0])
x0=np.array([5.0,5.0,25.0])

# Define the time points
T=25.0
dt=0.01                                                                        # Originally 0.001

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(Lorenz(np.transpose(x), 0, p0))

M, nstates = x.shape

# Get the data size info
stateVar,dataLen=np.transpose(x).shape

# Define the data type
dataType=tf.dtypes.float32

dh=tf.constant(dt)


#%% WSINDy ---  parameters
# Library
polyorder = 2                                      # Added by CL
# polys = list(range(0, polyorder))                  # Monomials. DEFAULT: polys = 0:5                # Modified by CL
trigs = []                                         # sine / cosine terms. DEFAULT: trigs = []
filter_func = []                                   # {@(tags) tags[:,2]==0}
# custom_tags = []
custom_tags = np.empty((0, nstates))               # by CL

custom_fcns = []                                   # prodlib(nstates,build_poly_lib(nstates,0:1,[],filter),build_trig_lib(nstates,[0 2 6],[],filter))'
                                                   # custom terms. DEFAULT = {}

# Weak formulation
phi_class = 1                                      # @(t)exp(9./(t.^2-1-eps)); Choose test function class. DEFAULT: phi_class = 1.
                                                   # ---phi_class = 1: piecewise poly test functions,
                                                   # ---phi_class = 2: gaussian test functions
                                                   # ---phi_class = function_handle: use function handle and symbolically compute derivative
tau = 1e-16
tauhat = -2                                        # Choose test function params. DEFAULT: [tau,tauhat] = [10^-16,-1].
                                                   # ---tau > 1: [tau,tauhat] = [m,p] directly
                                                   # ---tauhat > 0: tauhat = width-at-half-max.
                                                   # ---tauhat < 0: use corner point (recommended)
K_frac = 2500      # results shown 500              # Choose number of test functions K. DEFAULT: K_frac = 1.                                 # Modified by CL, before:1000
                                                   # ---K_frac > 1: K = K_frac.
                                                   # ---K_frac <= 1: K = M*K_frac (M = number of time points)

# Optimization
scale_Theta = 0          # %% Orig. 2%%            # rescale data. DEFAULT: scale_theta = 2.
                                                   # ---scale_Theta < 0: rescale data: xobs -> xobs./(-scale_theta*rms(xobs))
                                                   # ---scale_Theta > 0: rescale data: xobs -> xobs/||xobs||_{2p} where p=max(polys)
                                                   # ---scale_Theta == 0: no rescaling
lambdas = 10 ** np.linspace(-4, 0, 100)            # sparsity factor(s). DEFAULT: lambda = 10.^(linspace(-4,0,100)).
alpha_loss = 0.8                                   # convex combination btw resid and sparsity in MSTLS loss. DEFAULT: 0.8.
                                                   # *(smaller libraries require alpha_loss closer to 1)
gamma = 0                                          # Tikhonov regularization. DEFAULT: gamma = 0.
                                                   # *(helps for ill-conditioned Theta, e.g. combined trig + poly libs)

# Data smoothing
smoothing_window = int(np.ceil(len(t) / 100))   # rough guess for smoothing window. DEFAULT: smoothing_window = ceil(length(tobs)/100).
                                                   # *(should over-estimate the optimal window)
                                                   # *(if smoothing not detected to be advantageous, window length will be set to 1)

#%% WSINDy --- additional params less important to tune

overlap_frac_ag = 0.8                              # (involved in mini-routine to find steep gradients)
max_d = 1                                          # use {test function, derivative pair} {phi, -phi'} vs {phi', -phi''}, etc.
overlap_frac = 1                                   # fraction of allowable support overlap between neighboring test functions
                                                   # often the adaptive grid will pack
                                                   # test functions very close together,
                                                   # leading to nearly redundant rows.
                                                   # restricting the support overlap prevents this.
relax_AG = 0                                       # convex combination between uniform and gradient-adapted placement of test functions
                                                   # adaptive grid helps for dynamics
                                                   # with sharp transitions (e.g. Van der Pol)
useGLS = 0                                         # useGLS = 0 calls OLS. useGLS > 0 calls GLS with cov (1-useGLS)*C + useGLS*diag(C), C = Vp*Vp'
                                                   # GLS is helpful when the test
                                                   # functions have very small support and when the jacobian of the true system
                                                   # is small (e.g. linear Dynamics)
                                                   
#%% Define the SINDy parameters
libOrder = polyorder
# lam = 0.12

# Check the GPU status
CheckGPU()

# Define the noise percent you would like to swipe
#NoisePercentageToSwipe=[25,30,35,40,45,50]
# NoisePercentageToSwipe=[0,2,4,6,8,10,12,14,16,18,20]  
NoisePercentageToSwipe=[0,5,10,15,20,25,30,35,40,45,50]          
NoiseNum=len(NoisePercentageToSwipe)  

# Set a pin to generate new noise every run
pin=0

# Set a list to store the noise value
# NoiseList=[]
NoiseEsList=[]
NoiseIDList=[]
TrainTimeList=np.zeros((N_run,NoiseNum))                                         # Removed Nloop
Evector_field_error = np.zeros((N_run,NoiseNum))
Epre_short = np.zeros((N_run,NoiseNum))
Epar_error_list=np.zeros((N_run,NoiseNum))                                        # by CL
SuccessOrNot_list=np.zeros((N_run,NoiseNum))                                        # by CL
x_sim_list=[]
Xi_List=[]
Xi0_List=[]

# Softstart?
Softstart=0

#%%
@tf.function
def LibGPU(x):
    # =============================================================================
    # Following is the 3nd order lib for 2 dimentional system   
    # =============================================================================
    # z1=tf.gather(x,[0],axis=1)
    # z2=tf.gather(x,[1],axis=1)
    
    # Theta=tf.concat([z1,z2,z1**2,z1*z2,z2**2,z1**3,(z1**2)*z2,z1*(z2**2),z2**3],axis=1)   
    
    # =============================================================================
    # Following is the 2nd order lib for 3 dimentional system   
    # =============================================================================
    z1=tf.gather(x,[0],axis=1)
    z2=tf.gather(x,[1],axis=1)
    z3=tf.gather(x,[2],axis=1)
    
    
    Theta=tf.concat([z1,z2,z3,z1**2,z1*z2,z1*z3,z2**2,z2*z3,z3**2],axis=1)                 
    
    return Theta

# =============================================================================
# Define a function that calculates the noise signal speration accuracy
# =============================================================================
def ID_Accuracy_SINDy_CL(x,dx,LibGPU,Xi,dataLen,dt):
    Evector_field_error=np.linalg.norm(dx-tf.linalg.matmul(LibGPU(tf.constant(x,dtype='float32')),Xi),'fro')**2/np.linalg.norm(dx,'fro')**2
    
    xpre=[]
    xpre=RK45_F_SINDy(tf.constant([x[0,:]],dtype="float32"),LibGPU,Xi,dt).numpy()
    
    try:
        for i in range(1,dataLen-1):
            dummy=RK45_F_SINDy(tf.constant([xpre[-1]],dtype="float32"),LibGPU,Xi,dt).numpy()
            xpre=np.append(xpre,[dummy[0]],axis=0)
                
        # Epre_error=np.linalg.norm(x[1:]-xpre,'fro')**2/np.linalg.norm(x,'fro')**2
    except:
        print("The simulation blows up...Current Neural Network is not stable...")
        # Epre_error=float('nan')

    return Evector_field_error, xpre

#%%   Define the true parameters for parameter error                           # By CL
Xi_base = np.zeros((9,3))

Xi_base[0,0]=-10; Xi_base[0,1]=28
Xi_base[1,0]=10; Xi_base[1,1]=-1
Xi_base[2,2]=-8/3
Xi_base[4,2]=1
Xi_base[5,1]=-1

#%%
# =============================================================================
# Start the noise level swip from here! Good luck!
# =============================================================================
for i in range(N_run):
    print("This is the run:",str(i+1),"\n")
    
    for j in range(NoiseNum):
        
        # print("\t Setting the noise percentage as:",NoisePercentageToSwipe[j],"%\n")             # Commented for SWAN
        # First, let's set the noise for this run
        # Define the random seed for the noise generation
        pin=pin+1
        np.random.seed(pin)
        
        #-------------------------- Generate the noise ------------------------
        NoiseMag = [np.std(x[:,i])*NoisePercentageToSwipe[j]*0.01 for i in range(stateVar)]
        
        #----------- Gaussian Noise
        # Noise = np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])    

        # #----------- Uniform Noise            
        # # For each dimension jj
        # for jj in range(stateVar):
        #     # We want std = NoiseMag[i]
        #     loc = 0
        #     c = NoiseMag[jj]*np.sqrt(3.0)
            
        #     Noise_c = loc + np.random.uniform(-c,c,(dataLen,1))
            
        #     # Stack the samples
        #     if jj == 0:
        #         Noise = Noise_c
        #     else:
        #         Noise = np.hstack((Noise, Noise_c))

        # #----------- Rayleigh Noise            
        # # For each dimension jj
        # for jj in range(stateVar):
        #     # We want std = NoiseMag[i]
        #     desired_std = NoiseMag[jj]
        #     s = desired_std / np.sqrt((4-np.pi)/2)
            
        #     # Generate Rayleigh samples with the calculated scale
        #     rayleigh_samples = np.random.rayleigh(s, (dataLen, 1))
            
        #     # Center the distribution (subtract the mean to get zero mean)
        #     rayleigh_mean = s * np.sqrt(np.pi/2)
        #     centered_samples = rayleigh_samples - rayleigh_mean
            
        #     # Stack the samples
        #     if jj == 0:
        #         Noise = centered_samples
        #     else:
        #         Noise = np.hstack((Noise, centered_samples))

        #----------- gamma Noise                  
        # for jj in range(stateVar):
        #     # We want std = NoiseMag[i]
        #     desired_std = NoiseMag[jj]
                    
        #     # Generate gamma samples with the k value 
        #     k = 1
        #     theta = desired_std / np.sqrt(k)  # scale parameter
                    
        #     # Center the distribution (subtract the mean to get zero mean)
        #     centered_samples = np.random.gamma(k, theta, (dataLen, 1)) - k * theta
            
        #     # Stack the samples
        #     if jj == 0:
        #         Noise = centered_samples
        #     else:
        #         Noise = np.hstack((Noise, centered_samples))
                
        #----------- Dweibull Noise                  
        for jj in range(stateVar):
            # We want std = NoiseMag[i]
                    
            # Create the distribution
            centered_samples = NoiseMag[jj]*dweibull.rvs(2.1,size=dataLen).reshape(-1, 1)
            
            # Stack the samples
            if jj == 0:
                Noise = centered_samples
            else:
                Noise = np.hstack((Noise, centered_samples))
        
        
        # Add the noise and get the noisy data
        xn = x + Noise
        
        # print("\t Getting initial guess...")            # Commented for SWAN
        
        # Get the derivative of the noisy data, we directly take the derivative here without smoothing
        if Softstart==1:
            # Do the smoothing
            NoiseEs,xes=approximate_noise(np.transpose(xn), 20)
            NoiseEs=np.transpose(NoiseEs)
            xes=np.transpose(xes)
            NoiseEsList.append(NoiseEs)
            
            # Get derivative and library
            dxes=CalDerivative(xes,dt,1)
            Theta=Lib(xes,libOrder)
            
            # Get initial guess
            Xi0, v, vp, mts,pts, grid_or, loss_wsindy, tags = wsindy_ode_fun(xn,t,
                                                      polyorder,    custom_tags,custom_fcns,
                                                      phi_class,max_d,tau,tauhat,K_frac,overlap_frac,relax_AG,
                                                      scale_Theta,useGLS,lambdas,gamma,alpha_loss,
                                                                                                0) # smoothing_window 
            
            # Update V,Vp,Grid
            V.update(v)
            Vp.update(vp)
            Grid.update(grid_or)
            
            Tags = tf.constant(tags, dtype=dataType)
            
            NoiseEs = np.zeros((xn.shape[0],xn.shape[1]))
            # Set up optimization parameter
            NoiseVar=tf.Variable(NoiseEs, dtype=tf.dtypes.float32)
        else:
            Theta=Lib(xn,libOrder)
            Xi0, v, vp, mts,pts, grid_or, loss_wsindy, tags = wsindy_ode_fun(xn,t,
                                                      polyorder,    custom_tags,custom_fcns,
                                                      phi_class,max_d,tau,tauhat,K_frac,overlap_frac,relax_AG,
                                                      scale_Theta,useGLS,lambdas,gamma,alpha_loss,
                                                                                                0) # smoothing_window 
        
        #------------------------ Criteria-------------------------------------
        Xi=tf.Variable(Xi0,dtype=dataType)
        Evector_field_error[i,j], x_sim = ID_Accuracy_SINDy_CL(x,dx,LibGPU,Xi,dataLen,dt)
        
        preLen = int(0.24*len(t))
        
        Epre_short[i,j] = np.linalg.norm(x[1:preLen+1]-x_sim[0:preLen],'fro')**2/np.linalg.norm(x[1:preLen+1],'fro')**2

        Epar_error_list[i,j] = np.linalg.norm(Xi_base-Xi0,2)/np.linalg.norm(Xi_base,2)                                    
        
        if np.all((Xi_base != 0) == (Xi0 != 0)):
            SuccessOrNot_list[i, j] = 1              

#%%

data_dict = {"E_vector_field": Evector_field_error, 
             "E_short": Epre_short,
             "E_parameter": Epar_error_list,
             "Rate_success": SuccessOrNot_list}
savemat("WLorenz_Dweibull.mat", data_dict)