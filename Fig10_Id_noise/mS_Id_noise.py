# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:10:39 2020

@author: kahdi
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
from utils_NSS_SINDy import *
import time
from datetime import datetime
import os
import importlib
from scipy.io import savemat
#%% Create a path and folder to save the result
# FolderName="Result_NSS_SwipeNoiseLevel_HighSampling\\"
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
N_train= 5000

# Define the number of SINDy loop
Nloop = 8

# Define the parameters
p0 = np.array([0.5])

# Define the initial conditions
x0 = np.array([-2,1])

# Define the time points
T=10.0
dt=0.01

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x = odeint(VanderPol,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx = np.transpose(VanderPol(np.transpose(x), 0, p0))             # derivative of the trajectory 

# Get the data size info
stateVar,dataLen=np.transpose(x).shape

# Define the data type
dataType=tf.dtypes.float32

dh=tf.constant(dt)

# Define the SINDy parameters
# Test the SINDy
N_SINDy_Iter=15
disp=0
NormalizeLib=0
libOrder=3
lam=0.15

# Check the GPU status
CheckGPU()

# Define the prediction step
q = 2

# Define the optimizer to that will updates the Neural Network weights
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-09)
        
# Get the weights for the error
ro=0.9
weights=tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)

# Define the noise percent you would like to swipe
NoisePercentage=20
NoiseNum=1 

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
softstart=1

#%%
# Define the true parameters                                                   # By CL
Xi_base = np.zeros((9,2))

Xi_base[0,1]=-1
Xi_base[1,0]=1;Xi_base[1,1]=0.5
Xi_base[6,1]=-0.5
    
#%%
# =============================================================================
# Start the noise level swip from here! Good luck!
# =============================================================================
for i in range(N_run):
    print("This is the run:",str(i+1),"\n")

    # Define the random seed for the noise generation
    pin=pin+1
    np.random.seed(pin)
        
        # Generate the noise
    NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
    Noise=np.hstack([NoiseMag[i]*np.random.gamma(1,1,(dataLen,1)) for i in range(stateVar)])  # Changed to gamma noise

    # Add the noise and get the noisy data
    xn = x + Noise 
    
    # Define the mean of the noise
    NoiseMean=np.array([0,0])    
    
    for jj in range(3):
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
            
        dxes=CalDerivative(xes,dt,1)
    
        # Get the initial guess of noise, we make a random guess here. For beteer performance you could first smooth the data and guess the noise. 
        NoiseVar=tf.Variable(NoiseEs,dtype=tf.dtypes.float32)
    
        if jj==0:
        # Get the initial guess of the SINDy parameters
            Theta=Lib(xes,libOrder)
    
            Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)
        else:
            if softstart==1:
                Xi0=Xi.numpy()
            else:
            # Get the initial guess of the SINDy parameters
                Theta=Lib(xes,libOrder)
        
                Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)
            #Xi0=Xi.numpy()
            
        # print(Xi0)
    
        # Define the initial guess of the selection parameters
        Xi = tf.Variable(Xi0,dtype=dataType)
    
        # Set the initial active matrix
        Xi_act = tf.constant(np.ones(Xi0.shape),dtype=dataType)
        
        # Start training!
        # print("\t Start training...\n\n")
        # print(stop_flag) # Output: False
        for k in range(Nloop):
            print("Runing the loop ",str(k+1))
            # Denoise the signal
            # check_flag() # Output: Global flag is False

            NoiseID,totalTime=Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,N_train)
            
            # NoiseID,totalTime =Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,N_train, tolerance)
            
            # print("\t Current loop takes ",totalTime)
            # After the first iteration, minus the noise identified from the noisy measurement data
            xes=xn-NoiseID
            xes=xes[q+1:-q-1,:]
            
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
            
            # Regress
            for r in range(2):
                Xi_num[index_min[:,r],r]=solve_minnonzero(Theta[:,index_min[:,r]],dxes[:,r])
            
            # Print the new initial start point
            # print("New Xi result")
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
            
            preLen=len(t)
            Epre_short=np.linalg.norm(x[1:preLen+1]-x_sim[0:preLen],'fro')**2/np.linalg.norm(x[1:preLen+1],'fro')**2
            
            ParameterError = np.linalg.norm(Xi_base-Xi_num,2)/np.linalg.norm(Xi_base,2)                                    # By CL
            
            # Store the result of identified noise and training time                     # Commented, and values saved out of Nloop, by CL
            NoiseIDList.append(NoiseID)
            # x_sim_list.append(x_sim)
            # TrainTimeList[i,j,k]=totalTime
            Enoise_error_List[i,0,k]=Enoise_error
            Evector_field_error_list[i,0,k]=Evector_field_error
            Epre_error_list[i,0,k]=Epre_error
            Epre_short_error_list[i,0,k]=Epre_short
            Epar_error_list[i,0,k]=ParameterError
            if np.all((Xi_base != 0) == (Xi.numpy() != 0)):
                SuccessOrNot_list[i,0,k] = 1 
            # Xi_List.append(Xi.numpy())
            
#%%
data_dict = {"NoiseID": NoiseIDList, 
             "E_noise": Enoise_error_List, "E_vector_field": Evector_field_error_list, 
             "E_traj": Epre_error_list,"E_short_traj": Epre_short_error_list,"E_parameter": Epar_error_list,
             "Rate_success": SuccessOrNot_list}
savemat("m_Id_Noise_n20.mat", data_dict)