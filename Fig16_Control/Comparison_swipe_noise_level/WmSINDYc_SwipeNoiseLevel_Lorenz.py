# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:07:49 2025

@author: crist
"""

# =============================================================================
# The following code will swipe the effect of noise level on using WSINDy to perform noise signal seperation 
# We set T=25, dt=0.01, q=10, x0=[5.0,5.0,25.0], ro=0.9.
# The WSINDy library has 9 nonlinear features. We also perform the loop multiple times to record the performance of SINDy.
# =============================================================================

#%% Import packages
import numpy as np
from scipy.integrate import odeint, solve_ivp
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

# Get the data size info
stateVar,dataLen=np.transpose(x).shape

# Define the data type
dataType=tf.dtypes.float32

dh=tf.constant(dt)

#%% Define some parameters to swipe the different noise level
# Define how mant times you would like to run each noise level
N_run = 50
# Define how many iterations you allow when training your model
N_train = 5000
# Define the number of SINDy loop
Nloop = 8
# Define the time for simulation
preLen = int(0.24*len(t))
#
libOrder = 2
lam = 0.2
# Define the prediction step
q = 1

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
K_frac = len(t)      # results shown 500              # Choose number of test functions K. DEFAULT: K_frac = 1.                                 # Modified by CL, before:1000
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

# Check the GPU status
# CheckGPU()

# Define the optimizer to that will updates the Neural Network weights
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-07)
        
# Get the weights for the error
ro=0.9
weights=tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)

# Define the noise percent you would like to swipe
#NoisePercentageToSwipe=[25,30,35,40,45,50]
NoisePercentageToSwipe=[0] 
# NoisePercentageToSwipe=[0,5,10,15,20,25,30,35,40,45,50]          
NoiseNum=len(NoisePercentageToSwipe)  

# Set a pin to generate new noise every run
pin=0

# Set a list to store the noise value
NoiseList=[]
NoiseEsList=[]
NoiseIDList=[]
TrainTimeList=np.zeros((N_run,NoiseNum,Nloop))
Enoise_error_List=np.zeros((N_run,NoiseNum,Nloop))
Evector_field_error_list=np.zeros((N_run,NoiseNum,Nloop))
Epre_error_list=np.zeros((N_run,NoiseNum,Nloop))
Epre_short_error_list=np.zeros((N_run,NoiseNum,Nloop))
Epar_error_list=np.zeros((N_run,NoiseNum,Nloop))
SuccessOrNot_list=np.zeros((N_run,NoiseNum,Nloop))
x_sim_list=[]
Xi_List=[]
Xi0_List=[]

# Softstart?
Softstart=0

#%%  Class to update V,Vp,and Grid

class VariableContainer:
    def __init__(self, initial_values):
        self.variables = []
        self.shapes = []
        self.dataType = tf.float32  # Add this line to specify the data type
        for value in initial_values:
            self.shapes.append(value.shape)
            self.variables.append(tf.Variable(initial_value=value, dtype=self.dataType))

    def update(self, new_values):
      for i, (variable, value) in enumerate(zip(self.variables, new_values)):
        if variable.shape != value.shape:
        # Create new variable using class dataType 
          new_var = tf.Variable(np.zeros(value.shape), dtype=self.dataType)  
          self.variables[i] = new_var
        self.variables[i].assign(value)

# Just to initialize the VariableContainer
_, v, vp, _,_, grid_or, _, _ = wsindy_ode_fun(x,u.reshape(-1, 1),t,
                                          polyorder,    custom_tags,custom_fcns,
                                          phi_class,max_d,tau,tauhat,K_frac,overlap_frac,relax_AG,
                                          scale_Theta,useGLS,lambdas,gamma,alpha_loss,
                                                                                    0) # smoothing_window 
# Just to initialize the VariableContainer
V = VariableContainer(v)
Vp = VariableContainer(vp)
Grid = VariableContainer(grid_or)
            
def v_vp_grid(xn_in ,NoiseID_in, K_frac, stateVar,custom_fcns, tobs, tau, tauhat, max_d, phi_class):
    
    # After the first few iterations, minus the noise identified from the noisy measurement data
    xes_d = xn_in - NoiseID_in
    
    m, n = xes_d.shape
    K = math.ceil(K_frac)
      
    mts, pts = findcorners(xes_d, tobs, tau, -tauhat, phi_class) 
    
    v1 = []; vp1 = []; grid_1 = []
    
    for r in range(n):
        
        v0 = []; vp0 = [];  grid_0 = []
        
    ### get test function weights
        mt = mts[r]      
        pt = pts[r]       
        dv = np.full_like(mts, dt)
        
        diffthresh = max(np.ceil((2 * mt + 1) * (1 - overlap_frac)), 1)
        Cfs = phi_int_weights(mt,max_d,-pt)
        v0 = Cfs[-2, :] * ((mt * dt) ** (-max_d + 1)) * dv[r]
        vp0 = Cfs[-1, :] * ((mt * dt) ** (-max_d)) * dv[r]

    ########## get_tf_centers to obtain grid_i, just relax_AG==0 ##################
        grid_0 = np.arange(0, m - 2 * mt, max(diffthresh, np.ceil((m - 2 * mt) / K))).astype(int)
        
        v1.append(v0)
        vp1.append(vp0)
        grid_1.append(grid_0) 
    
    return v1,vp1,grid_1

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

U = tf.constant(u,dtype=dataType)                                              # Added by CL
U0 = tf.constant(u[q:dataLen-q],dtype=dataType)                                # Added by CL

for i in range(N_run):
    # print("This is the run:",str(i+1),"\n")
    # run_prints(i+1)
    
    for j in range(NoiseNum):
        # # Set up the lambda
        # if j==0:
        #     lam=0.05
        # #elif j<=5:
        # #    lam=0.1
        # #else:
        # #    lam=0.15
        # elif j==1 or j==2: # else if j is equal to 1 or 2
        #     lam=0.1 # assign 0.1 to lam
        # elif j==3 or j==4 or j==5: # else if j is equal to 3 or 4 or 5
        #     lam=0.15 # assign 0.15 to lam
        # else: # else (if j is greater than 5)
        #     lam=0.3 # assign 0.2 to lam
        
        # Recompute computational graph by calling tf.function one more time
        # LibGPU,RK45_F_SINDy,RK45_B_SINDy,SliceNoise,Prediction_SINDy,CalDerivativeMatrix,WeightMSE,OneStepLoss_NSS_SINDy,Train_NSS_SINDy,NSS_SINDy,ID_Accuracy_SINDy=ReloadFunction_SINDy()
        
        print("\t Setting the noise percentage as:",NoisePercentageToSwipe[j],"%\n")             # Commented for SWAN
        # First, let's set the noise for this run
        # Define the random seed for the noise generation
        pin=pin+1
        np.random.seed(pin)
        
        # Generate the noise
        NoiseMag = [np.std(x[:,i])*NoisePercentageToSwipe[j]*0.01 for i in range(stateVar)]
        Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])                  # Gaussian Noise
        
        # Add the noise and get the noisy data
        xn = x + Noise
        
        # print("\t Getting initial guess...")            # Commented for SWAN
        
        # Get the derivative of the noisy data, we directly take the derivative here without smoothing
        if Softstart==1:
            # Do the smoothing
            NoiseEs,xes=approximate_noise(np.transpose(xn), 20)
            NoiseEs=np.transpose(NoiseEs)
            xes=np.transpose(xes)
            # NoiseEsList.append(NoiseEs)
            
            # Get derivative and library
            # dxes=CalDerivative(xes,dt,1)
            # Theta=Lib(xes,libOrder)
            
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
            # Theta=Lib(xn,libOrder)
            Xi0, v, vp, mts,pts, grid_or, loss_wsindy, tags = wsindy_ode_fun(xn,u.reshape(-1, 1),t,
                                                      polyorder,    custom_tags,custom_fcns,
                                                      phi_class,max_d,tau,tauhat,K_frac,overlap_frac,relax_AG,
                                                      scale_Theta,useGLS,lambdas,gamma,alpha_loss,
                                                                                                0) # smoothing_window 
            # print(Xi0)
            
            # Update V,Vp,Grid
            V.update(v)
            Vp.update(vp)
            Grid.update(grid_or)
            
            Tags = tf.constant(tags, dtype=dataType)
            
            # NoiseVar=tf.Variable(tf.random.normal((dataLen,stateVar), mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None))  # Commented by CL
            
            NoiseEs = np.zeros((xn.shape[0],xn.shape[1]))
            # Get the initial guess of noise, we make a random guess here. For beteer performance you could first smooth the data and guess the noise. 
            NoiseVar = tf.Variable(NoiseEs,dtype=tf.dtypes.float32)
            
        # Store initial guess
        # Xi0_List.append(Xi0)
        
        # print("\t The initial guess of the parameters are:")              # Commented for SWAN
        # print(Xi0)
        
        # Define the initial guess of the selection parameters
        Xi=tf.Variable(Xi0,dtype=dataType)
        
        # Set the initial active matrix (all active)
        Xi_act=tf.constant(np.ones(Xi0.shape),dtype=dataType)
        
        # print("\t Setting up the parameters...\n")            # Commented for SWAN
        # Get the middel part of the measurement data (it will be define as constant)
        Y=tf.constant(xn,dtype=dataType)
        Y0=tf.constant(GetInitialCondition(xn,q,dataLen),dtype=dataType)
        
        # Ge the forward and backward measurement data (it is a constant that wouldn't change)
        Ypre_F,Ypre_B=SliceData(xn,q,dataLen)
        Ypre_F=tf.constant(Ypre_F,dtype=dataType)
        Ypre_B=tf.constant(Ypre_B,dtype=dataType)
        
        # Satrt training!
        # print("\t Start training...\n\n")            # Commented for SWAN
         
        for k in range(Nloop):
            # print("Runing the loop ",str(k+1))            # Commented for SWAN
            
            # Denoise the signal
            # NoiseID,totalTime=Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,N_train)
            NoiseID,totalTime = Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,N_train,V,Vp, Grid, U, U0) 
            
            # print("\t Current loop takes ",totalTime)            # Commented for SWAN
            
            # Update test functions
            v_up, vp_up, grid_up = v_vp_grid(xn[q+1:-q-1,:] ,NoiseID[q+1:-q-1,:], K_frac, stateVar,custom_fcns, t, tau, tauhat, max_d, phi_class)
            
            V.update(v_up)
            Vp.update(vp_up)
            Grid.update(grid_up)
            
            # After the first iteration, minus the noise identified from the noisy measurement data
            xes=xn-NoiseID
            xes=xes[q+1:-q-1,:]
            
            u = u.reshape(-1, 1)
            u_s = u[q+1:-q-1]
            xes_aug = np.concatenate((xes, u_s), axis=1)                  # Augmented
            Theta = Lib(xes_aug,libOrder)

            # print("Current Xi result")            # Commented for SWAN
            # print(Xi)
                
            # Do WSINDy on the denoised data
            index_min=abs(Xi.numpy())>lam
            Xi_act_dum=Xi_act.numpy()*index_min.astype(int)
            Xi_num=Xi.numpy()
            Xi_num=Xi_num*Xi_act_dum
            index_min=Xi_act_dum.astype(bool)
            
            
            for r in range(stateVar):
                
                v0 = []; vp0 = [];  grid0 = []
                
            # ----------  get test functions ----------     
                v0 = v_up[r] 
                vp0 = vp_up[r] 
                # v0 = v[r] 
                # vp0 = vp[r] 

            # ----------  get_tf_centers to obtain grid_i, just relax_AG==0  ---------- 
                grid0 = grid_up[r] 
                # grid0 = grid_or[r] 
                
            ### get linear system
                b = np.convolve(xes[:, r], vp0.ravel(), mode='valid')
                # b = b[grid0]                                                         # Added -1, for phi_0
                G = convolve2d(Theta, np.outer(v0, [1]), mode='valid')
                # G = G[grid0]                                                         # Added -1, for phi_0

            # ----------  RT is not calculated, just useGLS = 0  ---------- 
                # Regress
                Xi_num[index_min[:,r],r] = solve_minnonzero(G[:,index_min[:,r]], b)
            
            # Print the new initial start point
            # print("New Xi result")            # Commented for SWAN
            # print(Xi_num)
            
            # Determine which term should we focus on to optimize next
            Xi_act=tf.constant(Xi_act_dum,dtype=tf.dtypes.float32)
            Xi=tf.Variable(Xi_num,dtype=tf.dtypes.float32)
            
            # Calculate the performance
            Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_SINDy(x,dx,Noise,NoiseID,LibGPU,Xi,dataLen,dt,u)
            
            # Print the performance
            # print("\t\t The error between the true noise and estimated noise is:",Enoise_error,"\n")                                         # Commented for SWAN
            # print("\t\t The error between the true vector field and estimated vector field is:",Evector_field_error,"\n")
            # print("\t\t The error between the true trajector and simulted trajectory is:",Epre_error,"\n")
            
            Epre_short=np.linalg.norm(x[1:preLen+1]-x_sim[0:preLen],'fro')**2/np.linalg.norm(x[1:preLen+1],'fro')**2
            
            ParameterError = np.linalg.norm(Xi_base-Xi_num,2)/np.linalg.norm(Xi_base,2)                                    # By CL
            
            # Store the result of identified noise and training time                     # Commented, and values saved out of Nloop, by CL
            # NoiseIDList.append(NoiseID)
            # x_sim_list.append(x_sim)
            TrainTimeList[i,j,k]=totalTime
            Enoise_error_List[i,j,k]=Enoise_error
            Evector_field_error_list[i,j,k]=Evector_field_error
            Epre_error_list[i,j,k]=Epre_error
            Epre_short_error_list[i,j,k]=Epre_short
            Epar_error_list[i,j,k]=ParameterError
            if np.all((Xi_base != 0) == (Xi.numpy() != 0)):
                SuccessOrNot_list[i,j,k] = 1 
            # Xi_List.append(Xi.numpy())
            
#%%
# data_dict = {"NoiseID": NoiseIDList, "x_sim": x_sim_list, "Time": TrainTimeList, 
#              "E_noise": Enoise_error_List, "E_vector_field": Evector_field_error_list, 
#              "E_traj": Epre_error_list,"E_short_traj": Epre_short_error_list,"E_parameter": Epar_error_list,
#              "Rata_succes": SuccessOrNot_list,"Xi_list": Xi_List}

data_dict = {"Time": TrainTimeList, 
             "E_noise": Enoise_error_List, "E_vector_field": Evector_field_error_list, 
             "E_traj": Epre_error_list,"E_short_traj": Epre_short_error_list,"E_parameter": Epar_error_list,
             "Rate_success": SuccessOrNot_list}
savemat("WmSINDy_Res.mat", data_dict)
