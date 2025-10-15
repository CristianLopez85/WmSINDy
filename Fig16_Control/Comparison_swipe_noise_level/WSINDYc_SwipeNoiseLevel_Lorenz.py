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
from scipy.integrate import odeint, solve_ivp
from scipy.stats import dweibull
import tensorflow as tf
import matplotlib.pyplot as plt
from utils_NSS_SINDy_Wc import *
import time
# import tensorflow_probability as tfp
from datetime import datetime
import os
import importlib

import math
from utils_WSINDYc import *
from scipy.io import savemat

#%% Simulate

def sphs(Pf, K, t):
    i = np.arange(1, K + 1)
    theta = 2 * np.pi / K * np.cumsum(i)
    u = np.sqrt(2 / K) * np.cos(2 * np.pi * i[:, np.newaxis] * t / Pf + theta[:, np.newaxis])    
    return np.sum(u, axis=0)

def LorenzSys(t, x, u, p):
    """
    Lorenz system with forcing term.
    """
    return np.array([
        p['SIGMA'] * (-x[0] + x[1]) + u,
        p['RHO'] * x[0] - x[1] - x[0] * x[2],
        -p['BETA'] * x[2] + x[0] * x[1]
    ])

def getTrainingData():
    # System parameters
    p = {'SIGMA': 10, 'RHO': 28, 'BETA': 8/3}
    x0 = np.array([-8, 8, 27])                      # From Kaiser: x0
    dt = 0.002
    tspan = np.arange(0, 10 + dt, dt)
    
    # Forcing parameters
    Pf = 0.5*10                                                                # reduce freq: 10
    K = 8
    A = 10
    
    # Calculate forcing term for all time points
    u = A * sphs(Pf, K, tspan)
    
    # Including forcing term
    def system(t, x):
        # Interpolate forcing term for current time
        u_t = np.interp(t, tspan, u)
        return LorenzSys(t, x, u_t, p)
    
    # Solve system
    sol = solve_ivp(
        system,
        (tspan[0], tspan[-1]),
        x0,
        t_eval=tspan,
        rtol=1e-12,
        atol=1e-12 * np.ones(3),
        method='RK45',
        dense_output=True
    )
    
    # Extract solution
    t = sol.t
    x = sol.y.T
    
    # Calculate derivatives
    dx = np.zeros_like(x)
    for i in range(len(t)):
        dx[i] = LorenzSys(t[i], x[i], u[i], p)
    
    # Plotting
    plt.figure()
    plt.plot(t, u)
    plt.xlabel('Time')
    plt.ylabel('Input')
    plt.show()
    
    plt.figure()
    plt.plot(t, x[:, 0], linewidth=1.5, label='x')
    plt.plot(t, x[:, 1], linewidth=1.5, label='y')
    plt.plot(t, x[:, 2], linewidth=1.5, label='z')
    plt.xlabel('Time')
    plt.ylabel('Population size')
    plt.legend()
    plt.show()
    
    return t, x, u, dt, dx

# Run the simulation
t, x, u, dt, dx = getTrainingData()

#%% Define some parameters to swipe the different noise level
# Define how mant times you would like to run each noise level
N_run = 50

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
custom_tags = np.empty((0, stateVar))               # by CL

custom_fcns = []                                   # prodlib(stateVar,build_poly_lib(stateVar,0:1,[],filter),build_trig_lib(stateVar,[0 2 6],[],filter))'
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
K_frac = len(x)      # results shown 500              # Choose number of test functions K. DEFAULT: K_frac = 1.                                 # Modified by CL, before:1000
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
# @tf.function
# def LibGPU(x):
#     # =============================================================================
#     # Following is the 3nd order lib for 2 dimentional system   
#     # =============================================================================
#     # z1=tf.gather(x,[0],axis=1)
#     # z2=tf.gather(x,[1],axis=1)
    
#     # Theta=tf.concat([z1,z2,z1**2,z1*z2,z2**2,z1**3,(z1**2)*z2,z1*(z2**2),z2**3],axis=1)   
    
#     # =============================================================================
#     # Following is the 2nd order lib for 3 dimentional system   
#     # =============================================================================
#     z1=tf.gather(x,[0],axis=1)
#     z2=tf.gather(x,[1],axis=1)
#     z3=tf.gather(x,[2],axis=1)
    
    
#     Theta=tf.concat([z1,z2,z3,z1**2,z1*z2,z1*z3,z2**2,z2*z3,z3**2],axis=1)                 
    
#     return Theta

# =============================================================================
# Define a function that calculates the noise signal speration accuracy
# =============================================================================
def ID_Accuracy_SINDy_CL(x,dx,LibGPU,Xi,dataLen,dt,u):
    Evector_field_error=np.linalg.norm(dx-tf.linalg.matmul(LibGPU(tf.constant(x,dtype='float32'),u),Xi),'fro')**2/np.linalg.norm(dx,'fro')**2
    
    xpre=[]
    xpre=RK45_F_SINDy(tf.constant([x[0,:]],dtype="float32"),LibGPU,Xi,dt,u[0]).numpy()     # Added u
    
    try:
        for i in range(1,dataLen-1):
            dummy=RK45_F_SINDy(tf.constant([xpre[-1]],dtype="float32"),LibGPU,Xi,dt,u[i]).numpy()  # Added u
            xpre=np.append(xpre,[dummy[0]],axis=0)
                
        # Epre_error=np.linalg.norm(x[1:]-xpre,'fro')**2/np.linalg.norm(x,'fro')**2
    except:
        print("The simulation blows up...Current Neural Network is not stable...")
        # Epre_error=float('nan')

    return Evector_field_error, xpre

#%%   Define the true parameters for parameter error                           # By CL
Xi_base = np.zeros((14, 3))  # 14 library terms (no constant)
# Set true Lorenz parameters with control 
Xi_base[0, 0] = -10;  Xi_base[1, 0] = 10; Xi_base[3, 0] = 1       
Xi_base[0, 1] = 28;   Xi_base[1, 1] = -1; Xi_base[2, 1] = 0; Xi_base[6, 1] = -1      
Xi_base[2, 2] = -8/3; Xi_base[5, 2] = 1  

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
        Noise = np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])           
        
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
            Xi0, v, vp, mts,pts, grid_or, loss_wsindy, tags = wsindy_ode_fun(xn,u.reshape(-1, 1),t,
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
            Xi0, v, vp, mts,pts, grid_or, loss_wsindy, tags = wsindy_ode_fun(xn,u.reshape(-1, 1),t,
                                                      polyorder,    custom_tags,custom_fcns,
                                                      phi_class,max_d,tau,tauhat,K_frac,overlap_frac,relax_AG,
                                                      scale_Theta,useGLS,lambdas,gamma,alpha_loss,
                                                                                                0) # smoothing_window 
        
        #------------------------ Criteria-------------------------------------
        Xi=tf.Variable(Xi0,dtype=dataType)
        Evector_field_error[i,j], x_sim = ID_Accuracy_SINDy_CL(x,dx,LibGPU,Xi,dataLen,dt,u)
        
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
savemat("WSINDy_Res.mat", data_dict)