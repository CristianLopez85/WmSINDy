# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:10:32 2025

@author: crist
"""

# =============================================================================
# The following code will swipe the effect of prediction steps on using SINDy to perform noise signal speration.
# We set dt=0.01, x0=[-5.0,5.0,25.0], ro=0.9.
# =============================================================================

#%% Import packages
import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
import matplotlib.pyplot as plt
from utils_NSS_SINDy import *
import time
# import tensorflow_probability as tfp
from datetime import datetime
import os
import importlib
from scipy.io import savemat
#%% Create a path and folder to save the result
# FolderName="Result_NSS_SwipeLambda\\"
# FilePath=os.getcwd()
# SavePath=FilePath+'\\'+FolderName
# # Create the folder
# try:
#     os.mkdir(SavePath)
#     print("The file folder does not exist, will create a new one....\n")
# except:
#     print("The folder already exist, will store the new result in current folder...\n")

#%% Define some parameters to swipe the different noise level
# Define how mant times you would like to run each noise level
N_run = 50                                                                       

# Define how many iterations you allow when training your model
N_train = 5000                                                                     

# Define the number of SINDy loop
Nloop = 6                                                                          

# Define the simulation parameters
p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])

# Define the initial conditions
x0=np.array([5.0,5.0,25.0])
# x0=np.array([0, 1, 0])       # E. Lorenz
# x0=np.array([0.1, 0.1, 0.1])       # E. Lorenz
# Define the time points
T=25.0
dt=0.01

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(Lorenz(np.transpose(x), 0, p0))

preLen = int(0.24*len(t))

# Get the data size info
stateVar,dataLen=np.transpose(x).shape

# Define the data type
dataType=tf.dtypes.float32

dh=tf.constant(dt)

# Define the SINDy parameters
N_SINDy_Iter=15
disp=0
NormalizeLib=0
libOrder=2

# Check the GPU status
# CheckGPU()

# Define the optimizer to that will updates the Neural Network weights
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-07)
        
# Get the weights for the error
ro=0.9
q = 1#0                                                                          # Original: 10
weights=tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)

# Define the data length you would like to swipe
Lambda_List=(np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1]))
Lambda_List_Num=len(Lambda_List)

# Set a pin to generate new noise every run
pin=0

# Define the noise level
NoisePercentage = 40

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
Cost_fun=np.zeros((N_run,Lambda_List_Num,Nloop))
Proj_cost=np.zeros((N_run,Lambda_List_Num,Nloop))
Overfit_cost=np.zeros((N_run,Lambda_List_Num,Nloop))
x_sim_list=[]
Xi_List=[]
Xi0_List=[]

#%%        Parameter error                                                      # By CL
# Define the true parameters
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
    NoiseList.append(Noise)
    
    # Swipe the data length    
    for j in range(Lambda_List_Num):
        # print("\t Currently using: ",Lambda_List[j]," as lambda...")
        
        lam=Lambda_List[j]
        
        # Recompute computational graph by calling tf.function one more time
        # LibGPU,RK45_F_SINDy,RK45_B_SINDy,SliceNoise,Prediction_SINDy,CalDerivativeMatrix,WeightMSE,OneStepLoss_NSS_SINDy,Train_NSS_SINDy,NSS_SINDy,ID_Accuracy_SINDy=ReloadFunction_SINDy()
        
        # print("\t Getting initial guess...")
        # Get the initial guess of the SINDy parameters
        # Do the smoothing
        #-------------------- Do the smoothing --------------------------------
        # NoiseEs,xes=approximate_noise(np.transpose(xn), 20)
        # NoiseEs=np.transpose(NoiseEs)
        # xes=np.transpose(xes)
        #-------------------- NO Smoothing  ----------------------------- by CL, so next line use xn instead of xes
        NoiseEs = np.zeros((xn.shape[0],xn.shape[1]))
        
        # Get the derivative of the noisy data, we directly take the derivative here without smoothing
        dxes=CalDerivative(xn,dt,1)
        # NoiseEsList.append(NoiseEs)
        
        Theta=Lib(xn,libOrder)
        
        Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)
        Xi0_List.append(Xi0)
        
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
            # N_train = 5
            NoiseID,totalTime=Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,N_train)
            
            # print("\t Current loop takes ",totalTime)
            # After the first iteration, minus the noise identified from the noisy measurement data
            xes=xn-NoiseID
            dxes=CalDerivative(xes,dt,1)
            
            Theta=Lib(xes,libOrder)
            
            # print("Current Xi result")
            # print(Xi)
            
            # Do SINDy on the denoised data
            index_min=abs(Xi.numpy())>lam
            Xi_act_dum=Xi_act.numpy()*index_min.astype(int)
            Xi_num=Xi.numpy()
            Xi_num=Xi_num*Xi_act_dum
            index_min=Xi_act_dum.astype(bool)
            
            proj_cost = []
            
            # Regress
            for r in range(3):
                Xi_num[index_min[:,r],r]=solve_minnonzero(Theta[:,index_min[:,r]],dxes[:,r])
                
                
                #----------------- For cost function --------------------------
                # W_ls = np.linalg.lstsq(Theta, dxes[:,r], rcond=-1)[0]
                # GW_ls = np.linalg.norm(np.dot(Theta, W_ls))
                # W = W_full[:,r]
                # proj_cost = np.linalg.norm(Theta.dot(W - W_ls)) / GW_ls
                # overfit_cost = len(np.nonzero(W)[0]) / len(W)
                # lossvals.append(proj_cost + overfit_cost)
                
                # Compute W_ls (least squares solution)
                # W_ls = np.linalg.lstsq(Theta, dxes, rcond=None)[0]
                
                # W = Xi_num[:,r]
                # dx_calc = (np.gradient(xes[:, r])) / dt
                # proj_cost.append(np.linalg.norm(Theta @ W - dx_calc) / np.linalg.norm(dx_calc))
                
            
            # Least squares solution
            W_ls = np.linalg.lstsq(Theta, dxes, rcond=None)[0]
            GW_ls = np.linalg.norm(Theta @ W_ls)
            W = Xi_num
            
            # for _ in range(N_SINDy_Iter):
            #     smallinds = np.abs(W) < lam  # Find small coefficients
            #     W[smallinds] = 0  # Apply threshold
                
            #     for ind in range(W.shape[1]):  # Loop over state dimension
            #         biginds = ~smallinds[:, ind]
            #         if np.any(biginds):
            #             W[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], dxes[:, ind], rcond=None)[0]
            
            proj_cost = np.linalg.norm(Theta @ (W - W_ls)) / GW_ls
            overfit_cost = np.count_nonzero(W) / W.size
            lossvals = proj_cost + overfit_cost

            # Print the new initial start point
            # print("New Xi result")
            # print(Xi_num)
            
            # Determine which term should we focus on to optimize next
            Xi_act=tf.constant(Xi_act_dum,dtype=tf.dtypes.float32)
            Xi=tf.Variable(Xi_num,dtype=tf.dtypes.float32)
            
            # Calculate the performance
            Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_SINDy(x,dx,Noise,NoiseID,LibGPU,Xi,dataLen,dt)
            
            Epre_short=np.linalg.norm(x[1:preLen+1]-x_sim[0:preLen],'fro')**2/np.linalg.norm(x[1:preLen+1],'fro')**2       # by CL
            ParameterError = np.linalg.norm(Xi_base-Xi_num,2)/np.linalg.norm(Xi_base,2)                                    # by CL
            
            # Print the performance
            # print("\t\t The error between the true noise and estimated noise is:",Enoise_error,"\n")
            # print("\t\t The error between the true vector field and estimated vector field is:",Evector_field_error,"\n")
            # print("\t\t The error between the true trajector and simulted trajectory is:",Epre_error,"\n")
            
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
            Proj_cost[i,j,k]=np.mean(proj_cost)
            Overfit_cost[i,j,k]=overfit_cost
            Cost_fun[i,j,k]=lossvals
            # Xi_List.append(Xi.numpy())

#%%            
data_dict = {"Time": TrainTimeList, "E_noise": Enoise_error_List, "E_vector_field": Evector_field_error_list, 
             "E_traj": Epre_error_list,"E_short_traj": Epre_short_error_list,"E_parameter": Epar_error_list,
             "Rate_success": SuccessOrNot_list, "Cost_function": Cost_fun, "Cost_Proj": Proj_cost, "Cost_Overfit": Overfit_cost}
savemat("m_noAppNoise_CF.mat", data_dict)
