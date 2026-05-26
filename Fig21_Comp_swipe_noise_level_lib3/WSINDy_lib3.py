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
polyorder = 3                                      # Added by CL
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
    
    
    Theta=tf.concat([z1,z2,z3,z1**2,z1*z2,z1*z3,z2**2,z2*z3,z3**2,
                      z1**3,(z1**2)*z2,(z1**2)*z3,z1*(z2**2),(z1*z2)*z3,z1*(z3**2),
                      z2**3,(z2**2)*z3,z2*(z3**2),z3**3],axis=1)                 
    
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
Xi_base = np.zeros((19,3))

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
        # Set up the lambda
        # if j==0:
        #     lam=0.05
        # elif j<=5:
        #     lam=0.1
        # else:
        #     lam=0.15
        
        # Recompute computational graph by calling tf.function one more time
        # LibGPU,RK45_F_SINDy,RK45_B_SINDy,SliceNoise,Prediction_SINDy,CalDerivativeMatrix,WeightMSE,OneStepLoss_NSS_SINDy,Train_NSS_SINDy,NSS_SINDy,ID_Accuracy_SINDy=ReloadFunction_SINDy()
        
        # print("\t Setting the noise percentage as:",NoisePercentageToSwipe[j],"%\n")             # Commented for SWAN
        # First, let's set the noise for this run
        # Define the random seed for the noise generation
        pin=pin+1
        np.random.seed(pin)
        
        # Generate the noise
        NoiseMag = [np.std(x[:,i])*NoisePercentageToSwipe[j]*0.01 for i in range(stateVar)]
        Noise = np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])                # Gaussian Noise
        
        # Noise=np.hstack([NoiseMag[i]*np.random.uniform(-3.36,3.36,(dataLen,1))/2 for i in range(stateVar)])       # Uniform Noise      (-1,1)
        
        # Noise=np.hstack([NoiseMag[i]*np.random.gamma(1,1,(dataLen,1)) for i in range(stateVar)])              # Gamma Noise
        # if j==1:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.gamma(1,1,(dataLen,1))-0.376261) for i in range(stateVar)])              # Gamma Noise
        # if j==2:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.gamma(1,1,(dataLen,1))-0.752522) for i in range(stateVar)])              # Gamma Noise
        # if j==3:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.gamma(1,1,(dataLen,1))-1.128783) for i in range(stateVar)])              # Gamma Noise
        # if j==4:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.gamma(1,1,(dataLen,1))-1.505044) for i in range(stateVar)])              # Gamma Noise
        # if j==5:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.gamma(1,1,(dataLen,1))-1.881305) for i in range(stateVar)])              # Gamma Noise
        # if j==6:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.gamma(1,1,(dataLen,1))-2.257567) for i in range(stateVar)])              # Gamma Noise
        # if j==7:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.gamma(1,1,(dataLen,1))-2.633828) for i in range(stateVar)])              # Gamma Noise
        # if j==8:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.gamma(1,1,(dataLen,1))-3.010089) for i in range(stateVar)])              # Gamma Noise
        # if j==9:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.gamma(1,1,(dataLen,1))-3.386350) for i in range(stateVar)])              # Gamma Noise
        # if j==10:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.gamma(1,1,(dataLen,1))-3.762611) for i in range(stateVar)])              # Gamma Noise

        # # from scipy.special import gamma 
        # value_of_gamma = 0.9880333100398532                                     # gamma(1+1/1.03)
        
        # Noise=Noise=np.zeros((dataLen,stateVar))                                                              # Dweibull Noise
        # # # Assign noise
        # for ii in range(stateVar):
        #     Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(1.03,size=dataLen)  # 2.07
        # if j==1:                                                                                              # Dweibull Noise
        #     Noise=Noise=np.zeros((dataLen,stateVar))   
        #     for ii in range(stateVar):
        #         Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(1.03,size=dataLen)-0.376261*value_of_gamma  # 2.07
        # if j==2:                                                                                              # Dweibull Noise
        #     Noise=Noise=np.zeros((dataLen,stateVar))   
        #     for ii in range(stateVar):
        #         Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(1.03,size=dataLen)-0.752522*value_of_gamma  # 2.07
        # if j==3:                                                                                              # Dweibull Noise
        #     Noise=Noise=np.zeros((dataLen,stateVar))   
        #     for ii in range(stateVar):
        #         Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(1.03,size=dataLen)-1.128783*value_of_gamma  # 2.07  
        # if j==4:                                                                                              # Dweibull Noise
        #     Noise=Noise=np.zeros((dataLen,stateVar))   
        #     for ii in range(stateVar):
        #         Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(1.03,size=dataLen)-1.505044*value_of_gamma  # 2.07
        # if j==5:                                                                                              # Dweibull Noise
        #     Noise=Noise=np.zeros((dataLen,stateVar))   
        #     for ii in range(stateVar):
        #         Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(1.03,size=dataLen)-1.881305*value_of_gamma  # 2.07
        # if j==6:                                                                                              # Dweibull Noise
        #     Noise=Noise=np.zeros((dataLen,stateVar))   
        #     for ii in range(stateVar):
        #         Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(1.03,size=dataLen)-2.257567*value_of_gamma  # 2.07
        # if j==7:                                                                                              # Dweibull Noise
        #     Noise=Noise=np.zeros((dataLen,stateVar))   
        #     for ii in range(stateVar):
        #         Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(1.03,size=dataLen)-2.633828*value_of_gamma  # 2.07
        # if j==8:                                                                                              # Dweibull Noise
        #     Noise=Noise=np.zeros((dataLen,stateVar))   
        #     for ii in range(stateVar):
        #         Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(1.03,size=dataLen)-3.010089*value_of_gamma  # 2.07
        # if j==9:                                                                                              # Dweibull Noise
        #     Noise=Noise=np.zeros((dataLen,stateVar))   
        #     for ii in range(stateVar):
        #         Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(1.03,size=dataLen)-3.386350*value_of_gamma  # 2.07
        # if j==10:                                                                                              # Dweibull Noise
        #     Noise=Noise=np.zeros((dataLen,stateVar))   
        #     for ii in range(stateVar):
        #         Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(1.03,size=dataLen)-3.762611*value_of_gamma  # 2.07

        # Noise=Noise=np.zeros((dataLen,stateVar))
        # Assign noise
        # for ii in range(stateVar):
        #     Noise[:,ii]=NoiseMag[ii]*dweibull.rvs(2.07,size=dataLen)
        
        # Noise=np.hstack([NoiseMag[i]*np.random.rayleigh(1,(dataLen,1)) for i in range(stateVar)])           # Rayleigh Noise
        
        # Noise=np.hstack([(NoiseMag[i]*np.random.rayleigh(1,(dataLen,1))) for i in range(stateVar)])              # Rayleigh Noise
        # if j==1:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.rayleigh(1,(dataLen,1))-0.471573) for i in range(stateVar)])              # Rayleigh Noise
        # if j==2:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.rayleigh(1,(dataLen,1))-0.943147) for i in range(stateVar)])              # Rayleigh Noise
        # if j==3:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.rayleigh(1,(dataLen,1))-1.41472) for i in range(stateVar)])              # Rayleigh Noise
        # if j==4:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.rayleigh(1,(dataLen,1))-1.886294) for i in range(stateVar)])              # Rayleigh Noise
        # if j==5:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.rayleigh(1,(dataLen,1))-2.357867) for i in range(stateVar)])              # Rayleigh Noise
        # if j==6:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.rayleigh(1,(dataLen,1))-2.829441) for i in range(stateVar)])              # Rayleigh Noise
        # if j==7:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.rayleigh(1,(dataLen,1))-3.301014) for i in range(stateVar)])              # Rayleigh Noise
        # if j==8:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.rayleigh(1,(dataLen,1))-3.772588) for i in range(stateVar)])              # Rayleigh Noise
        # if j==9:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.rayleigh(1,(dataLen,1))-4.244161) for i in range(stateVar)])              # Rayleigh Noise
        # if j==10:
        #     Noise=np.hstack([(NoiseMag[i]*np.random.rayleigh(1,(dataLen,1))-4.715735) for i in range(stateVar)])              # Rayleigh Noise 
        # Store the generated noise 
        # NoiseList.append(Noise)
        
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
# Coll_errs = [Enoise_error_List,
#              Evector_field_error_list,
#              Epre_error_list,
#              Epar_error_list]

# print("\n Parameter Error:",Epar_error_list)

# print("\n\n\n\n Training finished! Please save the file using the Spyder variable explorer!")

#%%

# ind1= k-1 # 7

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(Epre_error_list[1,:,ind1])
# plt.title("Epre")
# plt.tight_layout
# plt.subplot(3,1,2)
# plt.plot(Evector_field_error_list[1,:,ind1])
# plt.title("Ef")
# plt.tight_layout
# plt.subplot(3,1,3)
# plt.plot(Enoise_error_List[1,:,ind1])
# plt.title("En")
# plt.tight_layout

# #%%
# ind2= k # 7
# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(Epre_error_list[0,ind2,:])
# #plt.yscale("log")
# plt.title("Epre")

# plt.subplot(3,1,2)
# plt.plot(Evector_field_error_list[0,ind2,:])
# #plt.yscale("log")
# plt.title("Ef")

# plt.subplot(3,1,3)
# plt.plot(Enoise_error_List[0,ind2,:])
# plt.yscale("log")
# plt.title("En")
# plt.tight_layout

data_dict = {"E_vector_field": Evector_field_error, 
             "E_short": Epre_short,
             "E_parameter": Epar_error_list,
             "Rate_success": SuccessOrNot_list}
savemat("WLorenz_noise_Exp50_fullqp.mat", data_dict)