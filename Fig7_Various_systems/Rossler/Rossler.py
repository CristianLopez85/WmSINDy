# -*- coding: utf-8 -*-
"""
Created on Thursday Jan 16 11:42:12 2025

@author: crist
"""

#%% Import packages
import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
import matplotlib.pyplot as plt
from utils_NSS_SINDy_W_Rossler import *
import time
# import tensorflow_probability as tfp
from datetime import datetime
import math

from itertools import permutations
from utils_WSINDy_Rossler import *

""" 
If using grid points, please change the following:
In the file 'utils_NSS_SINDy_W':
   1.- Check that LibGPU(x) is for Rossler
          Theta=np.ones((n,1),dtype=float)
          # Theta=x[:,[0]]
   2.- Line 96, modify as: for i in range(m)
In the file 'utils_WSINDy':
   3.- In def get_tags_SINDy(n): Use if n ==3:  # For Rossler
"""
#%% From mSINDy    
# Define how many percent of noise you need
NoisePercentage= 40

# Simulate
# Define the random seed for the noise generation
np.random.seed(0)

# Define the parameters
p0=np.array([0.2,0.2,5.7])

# Define the initial conditions
x0=np.array([3.0,5.0,0.0])

# Define the time points
T=25.0
dt=0.01

tobs=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(Rossler,x0,tobs,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(Rossler(np.transpose(x), 0, p0))

M, nstates = x.shape

# Get the size info
stateVar,dataLen=np.transpose(x).shape

# Generate the noise
NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])

# Add the noise and get the noisy data
xobs = x+Noise

#%% from WSINDy ---  Set parameters
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
K_frac = 0.1*len(tobs)      # results shown 500              # Choose number of test functions K. DEFAULT: K_frac = 1.                                 # Modified by CL, before:1000
                                                   # ---K_frac > 1: K = K_frac.
                                                   # ---K_frac <= 1: K = M*K_frac (M = number of time points)

# Optimization
scale_Theta = 0          # %% Orig. 2%%            # rescale data. DEFAULT: scale_theta = 2.
                                                   # ---scale_Theta < 0: rescale data: xobs -> xobs./(-scale_theta*rms(xobs))
                                                   # ---scale_Theta > 0: rescale data: xobs -> xobs/||xobs||_{2p} where p=max(polys)
                                                   # ---scale_Theta == 0: no rescaling
lambdas = 10 ** np.linspace(-4, 0, 100)            # sparsity factor(s). DEFAULT: lambda = 10.^(linspace(-4,0,100)).
alpha_loss = 0.8 #------0.8---------                 # convex combination btw resid and sparsity in MSTLS loss. DEFAULT: 0.8.
                                                   # *(smaller libraries require alpha_loss closer to 1)
gamma = 0                                          # Tikhonov regularization. DEFAULT: gamma = 0.
                                                   # *(helps for ill-conditioned Theta, e.g. combined trig + poly libs)

# Data smoothing
smoothing_window = int(np.ceil(len(tobs) / 100))   # rough guess for smoothing window. DEFAULT: smoothing_window = ceil(length(tobs)/100).
                                                   # *(should over-estimate the optimal window)
                                                   # *(if smoothing not detected to be advantageous, window length will be set to 1)

#%% from WSINDy --- additional params less important to tune

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

#%%    
    
# Test the wSINDy
libOrder = polyorder
lam = 0.08

# SoftStart?
softstart=0

#%% Now plot the result of Lorenz
xn = np.copy(xobs)                                                                           # To avoid update: by CL

plt.figure()

plt.subplot(3,1,1)
pp1=plt.plot(tobs,x[:,0],linewidth=0.5,color='k')
pp1=plt.scatter(tobs,xn[:,0],s=0.5,color='b')
plt.ylabel('x')
plt.grid(True)

plt.subplot(3,1,2)
pp2=plt.plot(tobs,x[:,1],linewidth=0.5,color='k')
pp1=plt.scatter(tobs,xn[:,1],s=0.5,color='b')
plt.ylabel('y')


plt.subplot(3,1,3)
pp3=plt.plot(tobs,x[:,2],linewidth=0.5,color='k')
pp1=plt.scatter(tobs,xn[:,2],s=0.5,color='b')
plt.ylabel('z')
plt.xlabel('t')
plt.tight_layout()
plt.grid(True)

#%%
lw=5
plt.figure(figsize=(20,16))
pp4=plt.axes(projection='3d')
pp4.plot3D(x[:,0],x[:,1],x[:,2], color='black',linewidth=lw)
pp4.plot3D(xn[:,0],xn[:,1],xn[:,2], color='blue',linestyle='--',linewidth=lw)
pp4.view_init(45, 260)
pp4.grid(False)
pp4.axis('off')
# plt.xlabel('z');plt.ylabel('y');plt.xlabel('x');plt.grid(True)
plt.savefig('Fig7a_Rossler_noisy.pdf')

#%% Define a neural network
# Check the GPU status
CheckGPU()

# Define the data type
dataType=tf.dtypes.float32

dh=tf.constant(dt)

#%% Define the data

# Define the prediction step
q = 1

# Get the middel part of the measurement data (it will be define as constant)
Y = tf.constant(xn,dtype=dataType)
Y0 = tf.constant(GetInitialCondition(xn,q,dataLen),dtype=dataType)

# Ge the forward and backward measurement data (it is a constant that wouldn't change)
Ypre_F,Ypre_B = SliceData(xn,q,dataLen)
Ypre_F = tf.constant(Ypre_F,dtype=dataType)
Ypre_B = tf.constant(Ypre_B,dtype=dataType)

# Get the weight for the error
ro = 0.9
weights = tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)

if softstart==1:
    # Soft Start
    NoiseEs,xes = approximate_noise(np.transpose(xn), 20)
    NoiseEs = np.transpose(NoiseEs)
    xes = np.transpose(xes)
else:
    # Hard Start
    NoiseEs = np.zeros((xn.shape[0],xobs.shape[1]))
    xes = xn - NoiseEs

# Get the initial guess of noise, we make a random guess here. For beteer performance you could first smooth the data and guess the noise. 
NoiseVar = tf.Variable(NoiseEs,dtype=tf.dtypes.float32)

# Get the initial guess of the WSINDy parameters

w_sparse, v, vp, mts,pts, grid_or, loss_wsindy, tags = wsindy_ode_fun(xobs,tobs,
                                                                      polyorder,    custom_tags,custom_fcns,
                                                                      phi_class,max_d,tau,tauhat,K_frac,overlap_frac,relax_AG,
                                                                      scale_Theta,useGLS,lambdas,gamma,alpha_loss,
                                                                                    0) # smoothing_window 

# From w to Xi
Xi0 = w_sparse
print(Xi0)

#%%
# Define the initial guess of the selection parameters
Xi=tf.Variable(Xi0,dtype=dataType)

# Set the initial active matrix
Xi_act=tf.constant(np.ones(Xi0.shape),dtype=dataType)

#%% First plot the noise: true v.s. identified
StartIndex=500 # Choose how many noise data point you would like to plot
EndIndex=700

plt.figure()
plt.subplot(3,1,1)
plt.title("Initial Guess")
pp1=plt.plot(tobs[StartIndex:EndIndex],Noise[StartIndex:EndIndex,0],linewidth=1.5,color='b')
pp1=plt.plot(tobs[StartIndex:EndIndex],NoiseVar.numpy()[StartIndex:EndIndex,0],linewidth=1.5,color='k',linestyle='--')
plt.ylabel('Nx')
plt.xlabel('t')
plt.legend(['Noise Truth:x', 'Noise Estimate:x'],loc='upper right')
plt.grid(True)

plt.subplot(3,1,2)
pp1=plt.plot(tobs[StartIndex:EndIndex],Noise[StartIndex:EndIndex,1],linewidth=1.5,color='b')
pp1=plt.plot(tobs[StartIndex:EndIndex],NoiseVar.numpy()[StartIndex:EndIndex,1],linewidth=1.5,color='k',linestyle='--')
plt.ylabel('Ny')
plt.xlabel('t')
plt.legend(['Noise Truth:y', 'Noise Estimate:y'],loc='upper right')
plt.grid(True)

plt.subplot(3,1,3)
pp1=plt.plot(tobs[StartIndex:EndIndex],Noise[StartIndex:EndIndex,2],linewidth=1.5,color='b')
pp1=plt.plot(tobs[StartIndex:EndIndex],NoiseVar.numpy()[StartIndex:EndIndex,2],linewidth=1.5,color='k',linestyle='--')
plt.ylabel('Nz')
plt.xlabel('t')
plt.legend(['Noise Truth:z', 'Noise Estimate:z'],loc='upper right')
plt.grid(True)
plt.tight_layout()
#plt.savefig('Result_NSS_SwipeNoiseLevel\Plots\Percent'+str(NoisePercentage)+'.pdf')

#%% Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-09)        
                                    
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

# # Initialize the VariableContainer
V = VariableContainer(v)
Vp = VariableContainer(vp)
Grid = VariableContainer(grid_or)

# Function to calculate V,Vp,and Grid

def v_vp_grid(xn_in ,NoiseID_in, tags, nstates,custom_fcns, tobs, tau, tauhat, max_d, phi_class, K_frac):
    
    # After the first few iterations, minus the noise identified from the noisy measurement data
    xes_d = xn_in - NoiseID_in
    
    m, n = xes_d.shape
              
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
        grid_0 = np.arange(0, m - 2 * mt, max(diffthresh, np.ceil((m - 2 * mt) / K_frac))).astype(int)
        
        v1.append(v0)
        vp1.append(vp0)
        grid_1.append(grid_0) 
    
    return v1,vp1,grid_1,ks


#%% Finally start training!
Nloop = 6
N_train = 5000
update_each = N_train

# Set a list to store the noise value
NoiseList = []
NoiseIDList_SingleRun = []
TrainTimeList_SingleRun = np.zeros((Nloop,1))
Enoise_error_List_SingleRun = np.zeros((Nloop,1))
Evector_field_error_list_SingleRun = np.zeros((Nloop,1))
Epre_error_list_SingleRun = np.zeros((Nloop,1))
x_sim_list_SingleRun = []
Xi_List_SingleRun = []

Tags = tf.constant(tags, dtype=dataType) 

#%%
N_train_mod = int(N_train/update_each)                  # So, total number of iterations in each k is N_train_mod * N_train

for k in range(Nloop):
    
    for kk in range(N_train_mod):
        
        #% Denoise the signal
        print("Running the loop ",str(k+1))
        
        NoiseID,totalTime = Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,update_each,Tags, V,Vp, Grid) 
        
        v_up, vp_up, grid_up,kup = v_vp_grid(xn[q+1:-q-1,:] ,NoiseID[q+1:-q-1,:], tags, nstates,custom_fcns, tobs, tau, tauhat, max_d, phi_class, K_frac)
        
        V.update(v_up)
        Vp.update(vp_up)
        Grid.update(grid_up)
    
    print("\t Current loop takes ",totalTime)
    # After the first iteration, minus the noise identified from the noisy measurement data
    xes = xn - NoiseID
    xes=xes[q+1:-q-1,:]
     
    print("Current Xi result")
    print(Xi)
    
    # Perform sparsity
    
    index_min = abs(Xi.numpy()) > lam
    Xi_act_dum=Xi_act.numpy()*index_min.astype(int)
    Xi_num=Xi.numpy()
    Xi_num=Xi_num*Xi_act_dum
    index_min=Xi_act_dum.astype(bool)
    
    ## ---------- Do WSINDy on the denoised data ----------       
    w_sparse1 = 0*w_sparse
    Theta = Lib(xes,libOrder)
              
    for r in range(nstates):
        
        v0 = []; vp0 = [];  grid0 = []
        
    ### get test functions
        v0 = v_up[r] 
        vp0 = vp_up[r] 
        # ----------  get_tf_centers to obtain grid_i, just relax_AG==0  ------
        grid0 = grid_up[r] 
        # rr
    ### get linear system
        b = np.convolve(xes[:, r], vp0.ravel(), mode='valid')
        # b = b[grid0]                                                         # Added -1, for phi_0
        G = np.zeros((len(Theta) - len(v0) + 1, len(Theta[0])))
        for i in range(Theta.shape[1]):
            G[:, i] = np.convolve(v0, Theta[:, i], mode='valid')
        # G = G[grid0]                                                         # Added -1, for phi_0

    ########## RT is not calculated, just useGLS = 0 ##################
        # Regress
        w_sparse0 = np.linalg.lstsq(G[:,index_min[:,r]], b, rcond=-1)[0]       
        w_sparse1[index_min[:,r],r] = w_sparse0.ravel()

    # From w to Xi
    Xi_num = w_sparse1
    
    # Print the new initial start point
    print("New Xi result")
    print(Xi_num)

    # Determine which term should we focus on to optimize next
    Xi_act=tf.constant(Xi_act_dum,dtype=tf.dtypes.float32)
    Xi=tf.Variable(Xi_num,dtype=tf.dtypes.float32)
    
    # Calculate the performance
    Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_SINDy(x,dx,Noise,NoiseID,LibGPU,Xi,dataLen,dt)
    
    # Print the performance
    print("\t\t The error between the true noise and estimated noise is:",Enoise_error,"\n")
    print("\t\t The error between the true vector field and estimated vector field is:",Evector_field_error,"\n")
    print("\t\t The error between the true trajectory and simulated trajectory is:",Epre_error,"\n")
    
    NoiseIDList_SingleRun.append(NoiseID)
    x_sim_list_SingleRun.append(x_sim)
    TrainTimeList_SingleRun[k]=totalTime
    Enoise_error_List_SingleRun[k]=Enoise_error
    Evector_field_error_list_SingleRun[k]=Evector_field_error
    Epre_error_list_SingleRun[k]=Epre_error
    Xi_List_SingleRun.append(Xi.numpy())    

# np.savetxt('result.txt', Xi_num, delimiter=',', fmt='%s')    
#%% Now plot the noise signal speration result

# First plot the noise: true v.s. identified
StartIndex=500 # Choose how many noise data point you would like to plot
EndIndex=700

plt.figure()
plt.subplot(3,1,1)
pp1=plt.plot(tobs[StartIndex:EndIndex],Noise[StartIndex:EndIndex,0],linewidth=1.5,color='b')
pp1=plt.plot(tobs[StartIndex:EndIndex],NoiseID[StartIndex:EndIndex,0],linewidth=1.5,color='k',linestyle='--')
plt.ylabel('Nx')
plt.xlabel('t')
plt.legend(['Noise Truth:x', 'Noise Estimate:x'],loc='upper right')
plt.grid(True)

plt.subplot(3,1,2)
pp1=plt.plot(tobs[StartIndex:EndIndex],Noise[StartIndex:EndIndex,1],linewidth=1.5,color='b')
pp1=plt.plot(tobs[StartIndex:EndIndex],NoiseID[StartIndex:EndIndex,1],linewidth=1.5,color='k',linestyle='--')
plt.ylabel('Ny')
plt.xlabel('t')
plt.legend(['Noise Truth:y', 'Noise Estimate:y'],loc='upper right')
plt.grid(True)

plt.subplot(3,1,3)
pp1=plt.plot(tobs[StartIndex:EndIndex],Noise[StartIndex:EndIndex,2],linewidth=1.5,color='b')
pp1=plt.plot(tobs[StartIndex:EndIndex],NoiseID[StartIndex:EndIndex,2],linewidth=1.5,color='k',linestyle='--')
plt.ylabel('Nz')
plt.xlabel('t')
plt.legend(['Noise Truth:z', 'Noise Estimate:z'],loc='upper right')
plt.grid(True)
plt.tight_layout()
#plt.savefig('Percent'+str(NoisePercentage)+'.pdf')

#%%
n_bins=int(180/5)
x_scale=np.linspace(-4*max(NoiseMag), 4*max(NoiseMag), 1000)

plt.figure()
plt.subplot(3,1,1)
plt.grid(True)
pp6=plt.hist(Noise[:,0], bins = n_bins, color = 'blue', alpha=0.9,edgecolor = 'black',density=True)
pp6=plt.hist(NoiseID[:,0], bins = n_bins, color = 'orange', alpha=0.75,edgecolor = 'black',density=True)
#pp6=plt.plot(x_scale,0.5*Gaussian(x_scale, 0, NoiseMag[0]),color ='black',alpha=0.9,linewidth=2.5)
plt.ylabel('Frequency')
plt.xlabel('Noise:x')
axes = plt.gca()
axes.set_xlim([-4*max(NoiseMag),4*max(NoiseMag)])
#axes.set_ylim([0,0.65])

plt.subplot(3,1,2)
plt.grid(True)
pp6=plt.hist(Noise[:,1], bins = n_bins, color = 'blue', alpha=0.9,edgecolor = 'black',density=True)
pp6=plt.hist(NoiseID[:,1], bins = n_bins, color = 'orange', alpha=0.75,edgecolor = 'black',density=True)
#pp6=plt.plot(x_scale,0.5*Gaussian(x_scale, 0, NoiseMag[1]),color ='black',alpha=0.9,linewidth=2.5)
plt.ylabel('Frequency')
plt.xlabel('Noise:y')
axes = plt.gca()
axes.set_xlim([-4*max(NoiseMag),4*max(NoiseMag)])
#axes.set_ylim([0,0.65])

plt.subplot(3,1,3)
plt.grid(True)
pp6=plt.hist(Noise[:,2], bins = n_bins, color = 'blue', alpha=0.9,edgecolor = 'black', density=True)
pp6=plt.hist(NoiseID[:,2], bins = n_bins, color = 'orange', alpha=0.75,edgecolor = 'black', density=True)
#pp6=plt.plot(x_scale,0.5*Gaussian(x_scale, 0, NoiseMag[2]),color ='black',alpha=0.9,linewidth=2.5)
plt.ylabel('Frequency')
plt.xlabel('Noise:z')
axes = plt.gca()
axes.set_xlim([-4*max(NoiseMag),4*max(NoiseMag)])
#axes.set_ylim([0,0.65])
plt.tight_layout()
#plt.savefig('Percent'+str(NoisePercentage)+'_Distribution.pdf')

#%%
plt.figure(figsize=(20,16))
pp4=plt.axes(projection='3d')
pp4.plot3D(x[:,0], x[:,1], x[:,2], color='black',linewidth=8)
pp4.plot3D((xn-NoiseID)[:,0], (xn-NoiseID)[:,1], (xn-NoiseID)[:,2], color='springreen',linestyle='--',linewidth=8)  # 150 100 50 100
pp4.view_init(45, 260)
pp4.grid(False)
pp4.axis('off')
#plt.savefig("Rossler_Denoised_NoiseLevel_"+str(NoisePercentage)+".pdf")
plt.savefig('Fig7a_Rossler_denoised.pdf')

#%%
Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_SINDy(x,dx,Noise,NoiseID,LibGPU,Xi,dataLen,dt)

print("The error between the true noise and estimated noise is:",Enoise_error)
print("The error between the true vector field and estimated vector field is:",Evector_field_error)
print("The error between the true trajectory and simulated trajectory is:",Epre_error)

#%%
fontSize=60
plt.rc('xtick', labelsize=fontSize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontSize) 

preLen = int(0.24*2500)
plt.figure(figsize=(20,16))
pp5=plt.axes(projection='3d')
pp5.plot3D(x[1:preLen+1,0], x[1:preLen+1,1], x[1:preLen+1,2], color='black',linewidth=8)
pp5.plot3D(x_sim[0:preLen,0], x_sim[0:preLen,1], x_sim[0:preLen,2], color='m',linestyle='--',linewidth=8)
pp5.plot([x[0,0]], [x[0,1]], [x[0,2]], markerfacecolor='black', markeredgecolor='orange', marker='*', markeredgewidth=5,markersize=60, alpha=1)
pp5.view_init(45, 260)
pp5.grid(True)
pp5.axis('on')
pp5.set_facecolor("white")

Epre_short=np.linalg.norm(x[1:preLen+1]-x_sim[0:preLen],'fro')**2/np.linalg.norm(x[1:preLen+1],'fro')**2

print("The short term prediction error is:",Epre_short)

# plt.savefig("Rossler_Pre_"+str(NoisePercentage)+".pdf")

#%%
fontSize=10
plt.rc('xtick', labelsize=fontSize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontSize) 

plt.figure()
plt.plot(Enoise_error_List_SingleRun)
plt.ylabel('NoiseID Error')
plt.xlabel('Loop Index')

plt.figure()
plt.plot(Evector_field_error_list_SingleRun)
plt.ylabel('Derivative Error')
plt.xlabel('Loop Index')

plt.figure()
plt.plot(Epre_error_list_SingleRun)
plt.ylabel('Simulation Error')
plt.xlabel('Loop Index')   
    

#%%        Parameter error                                                      # By CL
# Define the true parameters

Xi_base, _, _, _,_, _, _, _ = wsindy_ode_fun(x,tobs,
                                          polyorder,    custom_tags,custom_fcns,
                                          phi_class,max_d,tau,tauhat,K_frac,overlap_frac,relax_AG,
                                          scale_Theta,useGLS,lambdas,gamma,alpha_loss,
                                                                                    0) # smoothing_window 

ParameterError = np.linalg.norm(Xi_base-Xi_num,2)/np.linalg.norm(Xi_base,2)

Coll_Errs = [Enoise_error,Evector_field_error,Epre_error, Epre_short, ParameterError]

# print("Enoise_error,Evector_field_error,Epre_error, Epre_short, ParameterError:",Coll_Errs)   
  