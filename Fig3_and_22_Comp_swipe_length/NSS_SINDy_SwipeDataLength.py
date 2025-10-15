# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:50:07 2020

@author: kahdi
"""
# =============================================================================
# The following code will swipe the effect of data length on using SINDy to perform noise signal speration 
# We set dt=0.01, q=10, x0=[-5.0,5.0,25.0], ro=0.9.
# The SINDy library has 9 nonlinear features. We also perform the loop multiple times to record the performance of SINDy.
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
import pandas as pd
import multiprocessing
from scipy.io import savemat

#%%
def calculo(Longitud):
    #%% Create a path and folder to save the result
    # FolderName="Result_NIC_SwipeDataLength\\"
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
    N_run=50
    
    # Define how many iterations you allow when training your model
    N_train=5000
    
    # Define the number of SINDy loop
    Nloop=8
    
    # Define the simulation parameters
    p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])
    
    # Define the initial conditions
    x0=np.array([-5.0,5.0,25.0])
    #x0=np.array([5.0,5.0,25.0])
    
    # Define the time points
    T=25.0
    dt=0.01
    
    t=np.linspace(0.0,T,int(T/dt))
    
    # Now simulate the system
    x_full_length=odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
    dx_full_length=np.transpose(Lorenz(np.transpose(x_full_length), 0, p0))
    
    # Get the data size info
    stateVar,dataLen=np.transpose(x_full_length).shape
    
    # Define the data type
    dataType=tf.dtypes.float32
    
    dh=tf.constant(dt)
    
    # Define the SINDy parameters
    N_SINDy_Iter=15
    disp=0
    NormalizeLib=0
    libOrder=2
    lam=0.2
    
    # Check the GPU status
    # CheckGPU()
    
    # Define the prediction step
    q=3
    
    # Define the optimizer to that will updates the Neural Network weights
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-07)
            
    # Get the weights for the error
    ro=0.9
    weights=tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)
    
    # Define the data length you would like to swipe
    # DataLengthToSwipe=(np.array([Longitud])/dt).astype(int)
    DataLengthToSwipe=(np.array([Longitud])).astype(int)
    # DataLengthToSwipe=(np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,7.5,10,12.5,15,17.5,20,22.5,25])/dt).astype(int)
    DataLengthNum=len(DataLengthToSwipe)
    
    # Set a pin to generate new noise every run
    pin=0
    
    # Define the noise level
    NoisePercentage=40
    
    #%% Set a list to store the noise value
    NoiseList=[]
    NoiseEsList=[]
    NoiseIDList=[]
    TrainTimeList=np.zeros((N_run,DataLengthNum,Nloop))
    Enoise_error_List=np.zeros((N_run,DataLengthNum,Nloop))
    Evector_field_error_list=np.zeros((N_run,DataLengthNum,Nloop))
    Epre_error_list=np.zeros((N_run,DataLengthNum,Nloop))
    Epre_short_error_list=np.zeros((N_run,DataLengthNum,Nloop))
    Epar_error_list=np.zeros((N_run,DataLengthNum,Nloop))
    SuccessOrNot_list=np.zeros((N_run,DataLengthNum,Nloop))
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
        # run_prints(i+1)
        # print("This is the run:",str(i+1),"\n")
        # print("\t Setting the noise percentage as:",NoisePercentage,"%\n")
        # First, let's set the noise for this run
        # Define the random seed for the noise generation
        pin=pin+1
        np.random.seed(pin)
            
        # Generate the noise
        NoiseMag=[np.std(x_full_length[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
        Noise_full_length=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])
            
        # Add the noise and get the noisy data
        xn_full_length=x_full_length+Noise_full_length
        
        # Do the smoothing
        # NoiseEs_full_length,xes_full_length=approximate_noise(np.transpose(xn_full_length), 20)
        # NoiseEs_full_length=np.transpose(NoiseEs_full_length)
        # xes_full_length=np.transpose(xes_full_length)
        
        #-------------------------- by CL replacing previous 3 lines----------------
        NoiseEs_full_length = np.zeros((xn_full_length.shape[0],xn_full_length.shape[1]))
        xes_full_length = xn_full_length
    
        # Swipe the data length    
        for j in range(DataLengthNum):
            print("\t Currently using: ",DataLengthToSwipe[j]," data points...")
            
            Trimed_dataLen=DataLengthToSwipe[j]
            
            # Recompute computational graph by calling tf.function one more time
            # LibGPU,RK45_F_SINDy,RK45_B_SINDy,SliceNoise,Prediction_SINDy,CalDerivativeMatrix,WeightMSE,OneStepLoss_NSS_SINDy,Train_NSS_SINDy,NSS_SINDy,ID_Accuracy_SINDy=ReloadFunction_SINDy()
            
            # Take a portion of the data 
            x=x_full_length[0:DataLengthToSwipe[j],:]
            dx=dx_full_length[0:DataLengthToSwipe[j],:]
            xn=xn_full_length[0:DataLengthToSwipe[j],:]
            xes=xes_full_length[0:DataLengthToSwipe[j],:]
            Noise=Noise_full_length[0:DataLengthToSwipe[j],:]
            NoiseEs=NoiseEs_full_length[0:DataLengthToSwipe[j],:]
            
            # Store the generated noise 
            NoiseList.append(Noise)
            NoiseEsList.append(Noise)
            
            print("\t Getting initial guess...")
            # Get the derivative of the noisy data, we directly take the derivative here without smoothing
            dxes=CalDerivative(xes,dt,1)
            
            # Get the initial guess of the SINDy parameters
            Theta=Lib(xes,libOrder)
            
            Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)
            Xi0_List.append(Xi0)
            
            print("\t The initial guess of the parameters are:")
            print(Xi0)
            
            # Define the initial guess of the selection parameters
            Xi=tf.Variable(Xi0,dtype=dataType)
            
            # Set the initial active matrix (all active)
            Xi_act=tf.constant(np.ones(Xi0.shape),dtype=dataType)
            
            print("\t Setting up the parameters...\n")
            # Get the middel part of the measurement data (it will be define as constant)
            Y=tf.constant(xn,dtype=dataType)
            Y0=tf.constant(GetInitialCondition(xn,q,Trimed_dataLen),dtype=dataType)
            
            # Ge the forward and backward measurement data (it is a constant that wouldn't change)
            Ypre_F,Ypre_B=SliceData(xn,q,Trimed_dataLen)
            Ypre_F=tf.constant(Ypre_F,dtype=dataType)
            Ypre_B=tf.constant(Ypre_B,dtype=dataType)
            
            # Get the initial guess of noise, we make a random guess here. For beteer performance you could first smooth the data and guess the noise. 
            NoiseVar=tf.Variable(NoiseEs, dtype=tf.dtypes.float32)
            
            # Satrt training!
            # print("\t Start training...\n\n")
            
            for k in range(Nloop):
                # print("Runing the loop ",str(k+1))
                # Denoise the signal
                NoiseID,totalTime=Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,Trimed_dataLen,optimizer,N_train)
                
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
                Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_SINDy(x,dx,Noise,NoiseID,LibGPU,Xi,Trimed_dataLen,dt)
                
                short_dataLen = int(0.24*Trimed_dataLen)
                Epre_short=np.linalg.norm(x[1:short_dataLen+1]-x_sim[0:short_dataLen],'fro')**2/np.linalg.norm(x[1:short_dataLen+1],'fro')**2
                
                ParameterError = np.linalg.norm(Xi_base-Xi_num,2)/np.linalg.norm(Xi_base,2)
                
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
                # Xi_List.append(Xi.numpy())
                
                
                # Save Time
                # df = pd.DataFrame(TrainTimeList.reshape(N_run,Nloop))
                # df.to_excel(f'mS_Time{Longitud}.xlsx', index=False)
                        
                # # Save Enoise_error_list
                # df_noise = pd.DataFrame(Enoise_error_List.reshape(N_run,Nloop))
                # df_noise.to_excel(f'mS_E_noise{Longitud}.xlsx', index=False)
        
                # # Save Evector_field_error_list
                # df_vector = pd.DataFrame(Evector_field_error_list.reshape(N_run,Nloop))
                # df_vector.to_excel(f'mS_E_vector_field{Longitud}.xlsx', index=False)
                        
                # # Save Evector_field_error_list
                # df_traj = pd.DataFrame(Epre_error_list.reshape(N_run,Nloop))
                # df_traj.to_excel(f'mS_E_pre_{Longitud}.xlsx', index=False)
                        
                # # Save Eshort_error_list
                # df_short = pd.DataFrame(Epre_short_error_list.reshape(N_run,Nloop))
                # df_short.to_excel(f'mS_E_short_{Longitud}.xlsx', index=False)
                        
                # # Save Eshort_error_list
                # df_parameter = pd.DataFrame(Epar_error_list.reshape(N_run,Nloop))
                # df_parameter.to_excel(f'mS_E_parameter_{Longitud}.xlsx', index=False)
                        
                # # Save ESuccess_list
                # df_success = pd.DataFrame(SuccessOrNot_list.reshape(N_run,Nloop))
                # df_success.to_excel(f'mS_Success_{Longitud}.xlsx', index=False)
                
                savemat(f'mS_noAppN_{Longitud}.mat', {
                                    'TrainTime': TrainTimeList.reshape(N_run, Nloop),
                                    'E_noise': Enoise_error_List.reshape(N_run, Nloop),
                                    'E_vector_field': Evector_field_error_list.reshape(N_run, Nloop),
                                    'E_pre': Epre_error_list.reshape(N_run, Nloop),
                                    'E_short': Epre_short_error_list.reshape(N_run, Nloop),
                                    'E_parameter': Epar_error_list.reshape(N_run, Nloop),
                                    'Success': SuccessOrNot_list.reshape(N_run, Nloop)
                                })

if __name__ == "__main__":
	arr= (np.array([5,7.5,10,12.5,15,17.5,20,22.5,25])/0.01).astype(int)
    # arr= (np.array([1,2])/0.01).astype(int)
	with multiprocessing.Pool(processes=len(arr)) as pool:
		pool.map(calculo, arr)            
#%%
# print("\n\n\n\n Training finished! Please save the file using the Spyder variable explorer!")


# #%%

# ind1=1

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(Epre_error_list[0,:,ind1])
# plt.title("Epre")
# plt.tight_layout
# plt.subplot(3,1,2)
# plt.plot(Evector_field_error_list[0,:,ind1])
# plt.title("Ef")
# plt.tight_layout
# plt.subplot(3,1,3)
# plt.plot(Enoise_error_List[0,:,ind1])
# plt.title("En")
# plt.tight_layout

# #%%
# ind2=1
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



