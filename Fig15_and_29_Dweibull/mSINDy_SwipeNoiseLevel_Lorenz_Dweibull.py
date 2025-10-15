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
from scipy.stats import dweibull
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
Nloop = 6

# Define the time for simulation
preLen=600

# Define the time points
T=25.0
dt=0.01  

# Define the prediction step
q=1

# Define the simulation parameters
p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])

# Define the initial conditions
#x0=np.array([-5.0,5.0,25.0])
x0=np.array([5.0,5.0,25.0])

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(Lorenz(np.transpose(x), 0, p0))

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
libOrder=2

# Check the GPU status
CheckGPU()

# Define the optimizer to that will updates the Neural Network weights
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-07)
        
# Get the weights for the error
ro=0.9
weights=tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)

# Define the noise percent you would like to swipe
NoisePercentageToSwipe=[0,5,10,15,20,25,30,35,40,45,50]
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

x_sim_list=[]
Xi_List=[]
# Xi0_List=[]

# Softstart?
Softstart=0

#%%
#%        Parameter error                                                      # By CL
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

lam=0.2

for i in range(N_run):
    print("This is the run:",str(i+1),"\n")
    
    for j in range(NoiseNum):
        
        # Recompute computational graph by calling tf.function one more time
        # LibGPU,RK45_F_SINDy,RK45_B_SINDy,SliceNoise,Prediction_SINDy,CalDerivativeMatrix,WeightMSE,OneStepLoss_NSS_SINDy,Train_NSS_SINDy,NSS_SINDy,ID_Accuracy_SINDy=ReloadFunction_SINDy()
        
        print("\t Setting the noise percentage as:",NoisePercentageToSwipe[j],"%\n")
        # First, let's set the noise for this run
        # Define the random seed for the noise generation
        pin=pin+1
        np.random.seed(pin)
        
        for jj in range(stateVar):
            # Generate the noise
            NoiseMag = [np.std(x[:,i])*NoisePercentageToSwipe[j]*0.01 for i in range(stateVar)]
                        
            # Center the distribution (subtract mean to get zero mean)
            centered_samples = NoiseMag[jj]*dweibull.rvs(2.1,size=dataLen)
            centered_samples = centered_samples.reshape(dataLen,1)
            
            # Stack the samples
            if jj == 0:
                Noise = centered_samples
            else:
                Noise = np.hstack((Noise, centered_samples))
        
        # Add the noise and get the noisy data
        xn=x+Noise
        
        # print("\t Getting initial guess...")
        
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
            Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)
            
            # Set up optimization parameter
            NoiseVar=tf.Variable(NoiseEs, dtype=tf.dtypes.float32)
        else:
            dxn=CalDerivative(xn,dt,1)
            Theta=Lib(xn,libOrder)
            Xi0=SINDy(Theta,dxn,lam,N_SINDy_Iter,disp,NormalizeLib)

            # NoiseVar=tf.Variable(tf.random.normal((dataLen,stateVar), mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None))  # Commented by CL
            
            NoiseEs = np.zeros((xn.shape[0],xn.shape[1]))
            # Get the initial guess of noise, we make a random guess here. For beteer performance you could first smooth the data and guess the noise. 
            NoiseVar = tf.Variable(NoiseEs,dtype=tf.dtypes.float32)
            
        # Store initial guess
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
            
            #------------ Compare Xi original with Xi calculated                       # by CL
            # J_c = OneStepLoss_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,LibGPU,optimizer)   
            # Xi_in = tf.Variable(Xi0,dtype=dataType)
            # J_in = OneStepLoss_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi_in,Xi_act,weights,dt,q,stateVar,dataLen,LibGPU,optimizer)  

            # if J_in < J_c:
            #     Xi = Xi_in
                
            # Do SINDy on the denoised data
            index_min=abs(Xi.numpy())>lam
            Xi_act_dum=Xi_act.numpy()*index_min.astype(int)
            Xi_num=Xi.numpy()
            Xi_num=Xi_num*Xi_act_dum
            index_min=Xi_act_dum.astype(bool)
            
            # Regress
            for r in range(3):
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
            
            Epre_short=np.linalg.norm(x[1:preLen+1]-x_sim[0:preLen],'fro')**2/np.linalg.norm(x[1:preLen+1],'fro')**2
            
            ParameterError = np.linalg.norm(Xi_base-Xi_num,2)/np.linalg.norm(Xi_base,2)                                    # By CL
            
            # Store the result of identified noise and training time                     
            NoiseIDList.append(NoiseID)
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

data_dict = {"NoiseID": NoiseIDList, "Time": TrainTimeList, 
             "E_noise": Enoise_error_List, "E_vector_field": Evector_field_error_list, 
             "E_traj": Epre_error_list,"E_short_traj": Epre_short_error_list,"E_parameter": Epar_error_list,
             "Rate_success": SuccessOrNot_list}
savemat("mLorenz_Dweibull.mat", data_dict)

