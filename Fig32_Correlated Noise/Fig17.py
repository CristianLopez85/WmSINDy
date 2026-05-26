# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:43:55 2025

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
import tensorflow as tf
import matplotlib.pyplot as plt
from utils_NSS_SINDy_W import *
import time
# import tensorflow_probability as tfp
from datetime import datetime
import os
import importlib
import seaborn as sns
import math
from utils_WSINDy import *
from scipy.stats import dweibull
from scipy.linalg import eigh

#%% Define the plot parameters
def ChangeFontSize(fontSize):
    plt.rc('xtick', labelsize=fontSize)    
    plt.rc('ytick', labelsize=fontSize) 
    
    return None

def generate_correlated_noise(Q):
    eig_vals, eig_vecs = np.linalg.eigh(Q)
    μ = np.sqrt(np.maximum(eig_vals, 0))  # Ensure non-negative
    r = np.random.randn(len(eig_vals))
    return eig_vecs @ (μ * r)

#%% Define some parameters to swipe the different noise level
# Define how mant times you would like to run each noise level
N_run = 1

# Define how many iterations you allow when training your model
N_train= 5000

# Define the number of SINDy loop
Nloop = 6#, paper 6

# Simulate
# Define the random seed for the noise generation
# np.random.seed(4)

# Define the parameters
p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])

# Define the initial conditions
x0 = np.array([5,5,25])

# Define the time points
T=25.0
dt=0.01

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x = odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx = np.transpose(Lorenz(np.transpose(x), 0, p0))             # derivative of the trajectory

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
smoothing_window = 0   # rough guess for smoothing window. DEFAULT: smoothing_window = ceil(length(tobs)/100).
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
lam = 0.2

# Check the GPU status
CheckGPU()

# Define the prediction step
q = 1

# Define the optimizer to that will updates the Neural Network weights
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-09)
        
# Get the weights for the error
ro=0.9
weights=tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)

# Define the noise percent you would like to swipe
NoisePercentage = 50          # 50

# Set a pin to generate new noise every run
pin = 9                      # 9

# Set a list to store the noise value
# NoiseList=[]
NoiseEsList=[]
NoiseLoopID=[]
TrainTimeList=np.zeros((N_run,1))                                         # Removed Nloop
Enoise_error_List=np.zeros((N_run,1)) 
Evector_field_error_list=np.zeros((N_run,1)) 
Epre_error_list=np.zeros((N_run,1)) 
Epar_error_list=np.zeros((N_run,1))                                        # by CL
SuccessOrNot_List=np.zeros((N_run,1))                                        # by CL
x_sim_list=[]
Xi_List=[]
Xi0_List=[]
xn_list=[]

# Softstart?
softstart=0

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
w, v, vp, _,_, grid_or, _, _ = wsindy_ode_fun(x,t,
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
Xi_base, _, _, _,_, _, _, _ = wsindy_ode_fun(x,t,
                                          polyorder,    custom_tags,custom_fcns,
                                          phi_class,max_d,tau,tauhat,K_frac,overlap_frac,relax_AG,
                                          scale_Theta,useGLS,lambdas,gamma,alpha_loss,
                                                                                    0) # smoothing_window

#%%
# =============================================================================
# Start the noise level swip from here! Good luck!
# =============================================================================
for i in range(N_run):
    print("This is the run:",str(i+1),"\n")
    
    # Define the random seed for the noise generation
    pin=pin+1
    np.random.seed(pin)
        
    # #--------------------------------------------------------------------------

    #---------------------Generate correlated noise-----------------------------
    NoiseMag = [np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]

    # 1. Define correlation matrix (e.g., 0.5 correlation between variables), defined by CL
    C = np.array([[1,   0.5,  0.3],   # State 1 correlations
                  [0.5, 1,  -0.2],   # State 2 correlations
                  [0.3, -0.2, 1]])   # State 3 correlations
    
    # 2. Compute Q
    Q = np.diag(NoiseMag) @ C @ np.diag(NoiseMag)
    
    # 3. Generate correlated noise
    Noise = np.vstack([generate_correlated_noise(Q) for _ in range(dataLen)])
    
    # To verify:
    # 1.-# Compute empirical covariance matrix
    #   Q_empirical = np.cov(Noise, rowvar=False)
    #   Compare with theoretical Q
    #   print("Theoretical Q:\n", Q)
    #   print("Empirical Q:\n", Q_empirical)
    #   The empirical covariance Q should match Q
    # 2.- Check Eigenvalue Non-Negativity
    #   Q to be positive semi-definite (all eigenvalues ≥ 0). 
    #   assert np.all(np.linalg.eigvals(Q) >= -1e-10), "Q is not PSD!"
    # 3.- Verify Orthogonality of Eigenvectors
    # Check if D^T D ≈ I (identity matrix)
    #   eig_vals, eig_vecs = np.linalg.eigh(Q)
    #   Check if D^T D ≈ I (identity matrix)
    #   D = eig_vecs
    #   print("D^T D:\n", D.T @ D)
    #   The output should be approximately the identity matrix (e.g., off-diagonal entries < 1e-10).
    rr
    # Add the noise and get the noisy data
    xn = x + Noise 
    
    # Define the mean of the noise
    NoiseMean=np.array([0,0,0])    
    
    for jj in range(1):                                                        # 
        print("Iteration",str(jj+1))
        if jj==0:
            xn=xn-NoiseMean
        else:
            NoiseMean=np.mean(NoiseID,axis=0)
            xn=xn-NoiseMean
        
        # Store it
        # xn_list.append(xn)
    
        # Get the middel part of the measurement data (it will be define as constant)
        Y=tf.constant(xn,dtype=dataType)
        Y0=tf.constant(GetInitialCondition(xn,q,dataLen),dtype=dataType)
    
        # Get the forward and backward measurement data (it is a constant that wouldn't change)
        Ypre_F,Ypre_B=SliceData(xn,q,dataLen)
        Ypre_F=tf.constant(Ypre_F,dtype=dataType)
        Ypre_B=tf.constant(Ypre_B,dtype=dataType)
    
        if softstart==1:
        # Soft Start
            NoiseEs,xes=approximate_noise(np.transpose(xn), 20)
            NoiseEs=np.transpose(NoiseEs)
            xes=np.transpose(xes)
        else:
        # Hard Start
            NoiseEs=np.zeros((xn.shape[0],xn.shape[1]))
            xes=xn-NoiseEs
    
        # Get the initial guess of noise, we make a random guess here. For beteer performance you could first smooth the data and guess the noise. 
        NoiseVar=tf.Variable(NoiseEs,dtype=tf.dtypes.float32)
    
        if jj==0: 
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
        
        else:
            if softstart==1:
                Xi0=Xi.numpy()
            else:
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
            
        # print(Xi0)
        
        # Define the initial guess of the selection parameters
        Xi = tf.Variable(Xi0,dtype=dataType)
        
        # Set the initial active matrix (all active)
        Xi_act = tf.constant(np.ones(Xi0.shape),dtype=dataType)
        
        # Satrt training!
        # print("\t Start training...\n\n")            # Commented for SWAN
         
        for k in range(Nloop):
            print("Running the loop ",str(k+1))            # Commented for SWAN
            
            # Denoise the signal
            # NoiseID,totalTime=Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,N_train)
            NoiseID,totalTime = Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,N_train,Tags, V,Vp, Grid) 
            
            # print("\t Current loop takes ",totalTime)            # Commented for SWAN
            
            # Update test functions
            v_up, vp_up, grid_up,kup = v_vp_grid(xn[q+1:-q-1,:] ,NoiseID[q+1:-q-1,:], tags, nstates,custom_fcns, t, tau, tauhat, max_d, phi_class)
            
            V.update(v_up)
            Vp.update(vp_up)
            Grid.update(grid_up)
            
            # After the first iteration, minus the noise identified from the noisy measurement data
            xes=xn-NoiseID
            xes=xes[q+1:-q-1,:]
            
            Theta=Lib(xes,libOrder)
            
            # print("Current Xi result")            # Commented for SWAN
            # print(Xi)
                
            # Do WSINDy on the denoised data
            index_min=abs(Xi.numpy())>lam
            Xi_act_dum=Xi_act.numpy()*index_min.astype(int)
            Xi_num=Xi.numpy()
            Xi_num=Xi_num*Xi_act_dum
            index_min=Xi_act_dum.astype(bool)
            
            for r in range(nstates):
                
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
                #b = b[grid0]                                                         # Added -1, for phi_0
                G = convolve2d(Theta, np.outer(v0, [1]), mode='valid')
                #G = G[grid0]                                                         # Added -1, for phi_0

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
        Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_SINDy(x,dx,Noise,NoiseID,LibGPU,Xi,dataLen,dt)
            
            # Print the performance
            # print("\t\t The error between the true noise and estimated noise is:",Enoise_error,"\n")                                         # Commented for SWAN
            # print("\t\t The error between the true vector field and estimated vector field is:",Evector_field_error,"\n")
            # print("\t\t The error between the true trajector and simulted trajectory is:",Epre_error,"\n")
            
        preLen=len(t)                                                                                                  # By CL   
        Epre_short=np.linalg.norm(x[1:preLen+1]-x_sim[0:preLen],'fro')**2/np.linalg.norm(x[1:preLen+1],'fro')**2
            
        ParameterError = np.linalg.norm(Xi_base-Xi_num,2)/np.linalg.norm(Xi_base,2)                                    # By CL
            
            # Store the result of identified noise and training time                     # Commented, and values saved out of Nloop, by CL
            # NoiseIDList.append(NoiseID)
            # x_sim_list.append(x_sim)
            
    TrainTimeList[i,:]=totalTime
    Enoise_error_List[i,:]=Enoise_error
    Evector_field_error_list[i,:]=Evector_field_error
    Epre_error_list[i,:]= Epre_short
    Epar_error_list[i,:]=ParameterError
        # Xi_List.append(Xi.numpy())

    if np.all((Xi_base != 0) == (Xi.numpy() != 0)):
        SuccessOrNot_List[i, :] = 1            
    
    # Store both the noisy data and the result of identified noise and training time                     # Commented, and values saved out of Nloop, by CL
    xn_list.append(xn)
    NoiseLoopID.append(NoiseID)
    
#%%
Coll_errs = np.column_stack((Enoise_error_List,
              Evector_field_error_list,
              Epre_error_list,
              Epar_error_list,
              SuccessOrNot_List))

print("\n Noise Error:",Enoise_error_List)
print("\n Field Error:",Evector_field_error_list)
print("\n Prediction error:",Epre_error_list)
print("\n Parameter Error:",Epar_error_list)
print("\n Rate of Success:",SuccessOrNot_List)

# print("\n\n\n\n Training finished! Please save the file using the Spyder variable explorer!")
print("\n WmSINDy q:",q, ", L:",T, ", lambda:",lam, ", noise:",NoisePercentage, ", N_train:",N_train, ", Seeds:",pin)

#%%
lw=5
plt.figure(figsize=(20,16))
pp4=plt.axes(projection='3d')
pp4.plot3D(x[:,0],x[:,1],x[:,2], color='black',linewidth=lw)
pp4.plot3D(xn[:,0],xn[:,1],xn[:,2], color='blue',linestyle='--',linewidth=lw)
pp4.view_init(0, -30)
pp4.grid(False)
pp4.axis('off')
# plt.xlabel('z');plt.ylabel('y');plt.xlabel('x');plt.grid(True)
plt.savefig('Fig16a_noisy.pdf')

#%% Plot the distribution of noise
colorNoise='black'
colorNoiseID='red'

lw3=30
lw4=30

plt.figure(figsize=(30,30))
ChangeFontSize(80)

Iter = 0

# Density x
plt.subplot(3,1,1)
plt.grid()
sns.kdeplot(data=xn_list[Iter][:,0]-x[:,0], color=colorNoise, linewidth=lw4,linestyle='-')
sns.kdeplot(data=NoiseLoopID[Iter][:,0],    color=colorNoiseID,  linewidth=lw3, linestyle=':')

# Density y
plt.subplot(3,1,2)
plt.grid()
sns.kdeplot(data=xn_list[Iter][:,1]-x[:,1], color=colorNoise, linewidth=lw4, linestyle='-')
sns.kdeplot(data=NoiseLoopID[Iter][:,1], color=colorNoiseID, linewidth=lw3, linestyle=':')

# Density z
plt.subplot(3,1,3)
plt.grid()
sns.kdeplot(data=xn_list[Iter][:,2]-x[:,2], color=colorNoise, linewidth=lw4, linestyle='-')
sns.kdeplot(data=NoiseLoopID[Iter][:,2], color=colorNoiseID, linewidth=lw3, linestyle=':')

plt.tight_layout()

plt.savefig('Fig16a_NoiseDistID.pdf')