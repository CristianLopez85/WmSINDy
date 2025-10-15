# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 10:47:36 2020

@author: kahdi
"""

# =============================================================================
# The following code will swipe the effect of prediction steps on using SINDy to perform noise signal speration.
# We set dt=0.01, x0=[-5.0,5.0,25.0], ro=0.9.
# The WSINDy library has 9 nonlinear features. We also perform the loop multiple times to record the performance of SINDy.
# =============================================================================

#%% Import packages
import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
import matplotlib.pyplot as plt
from utils_NSS_SINDy_W_Duffing import *
import time
# import tensorflow_probability as tfp
from datetime import datetime
import os
import importlib
import math
from utils_WSINDy_Duffing import *
from scipy.io import savemat

#%% Define some parameters to swipe the different noise level
# Define the simulation parameters
np.random.seed(0)

# Define the parameters
p0 = np.array([0.2, 0.1, 1])

# Define the initial conditions
x0 = np.array([-2,2])

# Define how mant times you would like to run each noise level
N_run = 50

# Define how many iterations you allow when training your model
N_train= 5000

# Define the number of SINDy loop
Nloop = 5

# Define the time points
T=25.0
dt=0.01  
t=np.linspace(0.0,T,int(T/dt))

# Define the time for simulation
preLen = int(0.24*len(t))

# Define the prediction step
q=1

# Now simulate the system
x = odeint(Duffing,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx = np.transpose(Duffing(np.transpose(x), 0, p0))    

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

# Check the GPU status
# CheckGPU()

# Define the optimizer to that will updates the Neural Network weights
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-07)
        
# Get the weights for the error
ro=0.9

weights=tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)

# Define the data length you would like to swipe
Lambda_List=(np.array([0.05]))
Lambda_List_Num=len(Lambda_List)

# Set a pin to generate new noise every run
pin=0

# Define the noise level
NoisePercentage = 25

#%% Set a list to store the noise value
NoiseList=[]
NoiseEsList=[]
NoiseIDList=[]
TrainTimeList=np.zeros((N_run,Lambda_List_Num,Nloop))
Enoise_error_List=np.zeros((N_run,Lambda_List_Num,Nloop))
Evector_field_error_list=np.zeros((N_run,Lambda_List_Num,Nloop))
Epre_error_list=np.zeros((N_run,Lambda_List_Num,Nloop))
Epre_short_error_list=np.zeros((N_run,Lambda_List_Num,Nloop))
Epar_error_list=np.zeros((N_run,Lambda_List_Num,Nloop))
SuccessOrNot_list=np.zeros((N_run,Lambda_List_Num,Nloop))
cost_fun=np.zeros((N_run,Lambda_List_Num,Nloop))
Proj_cost=np.zeros((N_run,Lambda_List_Num,Nloop))
Overfit_cost=np.zeros((N_run,Lambda_List_Num,Nloop))
x_sim_list=[]
Xi_List=[]
Xi0_List=[]

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
_, v, vp, _,_, grid_or, _, _ = wsindy_ode_fun(x,t,
                                          polyorder,    custom_tags,custom_fcns,
                                          phi_class,max_d,tau,tauhat,K_frac,overlap_frac,relax_AG,
                                          scale_Theta,useGLS,lambdas,gamma,alpha_loss,
                                                                                    0) # smoothing_window 
# Just to initialize the VariableContainer
V = VariableContainer(v)
Vp = VariableContainer(vp)
Grid = VariableContainer(grid_or)
            
def v_vp_grid(xn_in ,NoiseID_in, tags, nstates,custom_fcns, tobs, tau, tauhat, max_d, phi_class):
    
    # After the first few iterations, minus the noise identified from the noisy measurement data
    xes_d = xn_in - NoiseID_in
    
    m, n = xes_d.shape
    K = math.ceil(len(xes_d))
              
    mts, pts,ks = findcorners(xes_d, tobs, tau, -tauhat, phi_class)
    
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
        grid_0 = grid_0 - 1
        
        v1.append(v0)
        vp1.append(vp0)
        grid_1.append(grid_0) 
    
    return v1,vp1,grid_1,ks

#%%   Define the true parameters for parameter error                           # By CL
Xi_base = np.zeros((9,2))

Xi_base[0,1]=-0.1
Xi_base[1,0]=1;Xi_base[1,1]=-0.2
Xi_base[5,1]=-1

#%%
# =============================================================================
# Start the noise level swip from here! Good luck!
# =============================================================================
for i in range(N_run):
    # print("This is the run:",str(i+1),"\n")
    # run_prints(i+1)
    # print("\t Setting the noise percentage as:",NoisePercentage,"%\n")
    # First, let's set the noise for this run
    # Define the random seed for the noise generation
    pin=pin+1
    np.random.seed(pin)
        
    # Generate the noise
    NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
    Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])
        
    # Add the noise and get the noisy data
    xn=x+Noise
    
    # Store the generated noise 
    # NoiseList.append(Noise)

    # Swipe the data length    
    for j in range(Lambda_List_Num):
        # print("\t Currently using: ",Lambda_List[j]," as lambda...")
        
        lam=Lambda_List[j]
        
        # Recompute computational graph by calling tf.function one more time
        # LibGPU,RK45_F_SINDy,RK45_B_SINDy,SliceNoise,Prediction_SINDy,CalDerivativeMatrix,WeightMSE,OneStepLoss_NSS_SINDy,Train_NSS_SINDy,NSS_SINDy,ID_Accuracy_SINDy=ReloadFunction_SINDy()
        
        # print("\t Getting initial guess...")
        # Get the initial guess of the SINDy parameters
        
        #-------------------- Do the smoothing --------------------------------
        NoiseEs,xes=approximate_noise(np.transpose(xn), 20)
        NoiseEs=np.transpose(NoiseEs)
        xes=np.transpose(xes)
        #-------------------- NO Smoothing  ----------------------------- by CL, so next line use xn instead of xes
        # NoiseEs = np.zeros((xn.shape[0],xn.shape[1]))
        #
        
        # Get the derivative of the noisy data, we directly take the derivative here without smoothing
        # NoiseEsList.append(NoiseEs)
        
        # Theta=Lib(xes,libOrder)
        
            # Get initial guess
        Xi0, v, vp, mts,pts, grid_or, loss_wsindy, tags = wsindy_ode_fun(xes,t,
                                                      polyorder,    custom_tags,custom_fcns,
                                                      phi_class,max_d,tau,tauhat,K_frac,overlap_frac,relax_AG,
                                                      scale_Theta,useGLS,lambdas,gamma,alpha_loss,
                                                                                                0) # smoothing_window 
            
            # Update V,Vp,Grid
        V.update(v)
        Vp.update(vp)
        Grid.update(grid_or)
            
        Tags = tf.constant(tags, dtype=dataType)
                    
        # Set up optimization parameter
            
        # Xi0_List.append(Xi0)
        
        # print("\t The initial guess of the parameters are:")
        # print(Xi0)
        
        # Define the initial guess of the selection parameters
        Xi=tf.Variable(Xi0,dtype=dataType)
        
        # Set the initial active matrix (all active)
        Xi_act=tf.constant(np.ones(Xi0.shape),dtype=dataType)
        
        # print("\t Setting up the parameters...\n")
        # Get the middel part of the measurement data (it will be define as constant)
        Y=tf.constant(xn,dtype=dataType)
        Y0=tf.constant(GetInitialCondition(xn,q,dataLen),dtype=dataType)
        
        # Ge the forward and backward measurement data (it is a constant that wouldn't change)
        Ypre_F,Ypre_B=SliceData(xn,q,dataLen)
        Ypre_F=tf.constant(Ypre_F,dtype=dataType)
        Ypre_B=tf.constant(Ypre_B,dtype=dataType)
        
        # Get the initial guess of noise, we make a random guess here. For beteer performance you could first smooth the data and guess the noise. 
        NoiseVar=tf.Variable(NoiseEs, dtype=tf.dtypes.float32)
        
        # Satrt training!
        # print("\t Start training...\n\n")
        
        for k in range(Nloop):
            # print("Runing the loop ",str(k+1))
            # Denoise the signal
            NoiseID,totalTime = Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,N_train,Tags, V,Vp, Grid) 
            
            # print("\t Current loop takes ",totalTime)            # Commented for SWAN
            
            # Update test functions
            v_up, vp_up, grid_up,kup = v_vp_grid(xn[q+1:-q-1,:] ,NoiseID[q+1:-q-1,:], tags, nstates,custom_fcns, t, tau, tauhat, max_d, phi_class)
            
            V.update(v_up)
            Vp.update(vp_up)
            Grid.update(grid_up)
            
            # print("\t Current loop takes ",totalTime)
            # After the first iteration, minus the noise identified from the noisy measurement data
            xes=xn-NoiseID
            xes=xes[q+1:-q-1,:]
            
            Theta=Lib(xes,libOrder)
            
            # print("Current Xi result")
            # print(Xi)
            
            # Do SINDy on the denoised data
            index_min=abs(Xi.numpy())>lam
            Xi_act_dum=Xi_act.numpy()*index_min.astype(int)
            Xi_num=Xi.numpy()
            Xi_num=Xi_num*Xi_act_dum
            index_min=Xi_act_dum.astype(bool)
            
            lossvals = []                                                      # For cost function
            proj_cost = []                                                     # For cost function
            overfit_cost = []                                                  # For cost function
            
            # Regress
            for r in range(nstates):
                
                v0 = []; vp0 = [];  grid0 = []
                
            # ----------  get test functions ----------     
                v0 = v_up[r] 
                vp0 = vp_up[r] 
                # v0 = v[r] 
                # vp0 = vp[r] 

            # ----------  get_tf_centers to obtain grid_i, just relax_AG==0  ---------- 
                grid0 = grid_up[r]                                                   # Added -1, because using wsindy_ode_fun
                # grid0 = grid_or[r] 
                
            ### get linear system
                b = np.convolve(xes[:, r], vp0.ravel(), mode='valid')
                #b = b[grid0]                                                         # Added -1, for phi_0
                G = convolve2d(Theta, np.outer(v0, [1]), mode='valid')
                #G = G[grid0]

            # ----------  RT is not calculated, just useGLS = 0  ---------- 
                # Regress
                Xi_num[index_min[:,r],r] = solve_minnonzero(G[:,index_min[:,r]], b)

                #----------------- For cost function --------------------------
                # W_ls = np.linalg.lstsq(G, b, rcond=None)[0]
                # GW_ls = np.linalg.norm(G @ W_ls)
                                
                # W = Xi_num[:,r]
                # proj_cost.append(np.linalg.norm(G @ (W - W_ls)) / GW_ls)
                # overfit_cost.append(np.count_nonzero(W) / W.shape[0])
                # lossvals.append(np.linalg.norm(G @ (W - W_ls)) / GW_ls + np.count_nonzero(W) / W.shape[0])
                

            # Print the new initial start point
            # print("New Xi result")
            # print(Xi_num)
            
            # Determine which term should we focus on to optimize next
            Xi_act=tf.constant(Xi_act_dum,dtype=tf.dtypes.float32)
            Xi=tf.Variable(Xi_num,dtype=tf.dtypes.float32)
            
            # Calculate the performance
            Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_SINDy(x,dx,Noise,NoiseID,LibGPU,Xi,dataLen,dt)
            
            # Print the performance
            # print("\t\t The error between the true noise and estimated noise is:",Enoise_error,"\n")
            # print("\t\t The error between the true vector field and estimated vector field is:",Evector_field_error,"\n")
            # print("\t\t The error between the true trajector and simulted trajectory is:",Epre_error,"\n")

            Epre_short=np.linalg.norm(x[1:preLen+1]-x_sim[0:preLen],'fro')**2/np.linalg.norm(x[1:preLen+1],'fro')**2
            ParameterError = np.linalg.norm(Xi_base-Xi_num,2)/np.linalg.norm(Xi_base,2)  # By CL
            
            # Store the result of identified noise and training time
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
            # Proj_cost[i,j,k]=np.sum(proj_cost)
            # Overfit_cost[i,j,k]=np.sum(overfit_cost)
            # cost_fun[i,j,k]=np.sum(lossvals)
            # Xi_List.append(Xi.numpy())

#%%
# data_dict = {"NoiseID": NoiseIDList, "x_sim": x_sim_list, "Time": TrainTimeList, 
#              "E_noise": Enoise_error_List, "E_vector_field": Evector_field_error_list, 
#              "E_traj": Epre_error_list,"E_short_traj": Epre_short_error_list,"E_parameter": Epar_error_list,
#              "Rate_succes": SuccessOrNot_list,"Xi_list": Xi_List}

data_dict = {"Time": TrainTimeList, "E_noise": Enoise_error_List, "E_vector_field": Evector_field_error_list, 
             "E_traj": Epre_error_list,"E_short_traj": Epre_short_error_list,"E_parameter": Epar_error_list,
             "Rate_success": SuccessOrNot_list}
savemat("Wm_Duffing.mat", data_dict)