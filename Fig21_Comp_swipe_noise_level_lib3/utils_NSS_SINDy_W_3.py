# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:57:33 2025

@author: kahdi
        Modified by CL
"""
#%% Import packages
import numpy as np
import tensorflow as tf
from scipy.linalg import qr
import time

#%% Define functions here
# =============================================================================
# Define the ODE of Van der Pol
# =============================================================================
def  VanderPol(u,t,p):
    du1=u[1]
    du2=p[0]*(1-u[0]**2)*u[1]-u[0]
    
    du=np.array([du1,du2])
    
    #p0=0.5
    
    return du

# =============================================================================
# Define the ODE of Duffing
# =============================================================================
def  Duffing(u,t,p):
    du1=u[1]
    du2=-p[0]*u[1]-p[1]*u[0]-p[2]*u[0]**3
    
    du=np.array([du1,du2])
    
    #p0=[0.2,0.05,1]
    
    return du

# =============================================================================
# Define the ODE of Lotka
# =============================================================================
def  Lotka(u,t,p):
    du1=p[0]*u[0]-p[1]*u[0]*u[1]
    du2=p[1]*u[0]*u[1]-2*p[0]*u[1]
    
    du=np.array([du1,du2])
    
    #p0=[1,10]
    
    return du

# =============================================================================
# Define the ODE for cubic oscalator
# =============================================================================
def CubicOsc(u,t,p):
    du1=p[0]*u[0]**3+p[1]*u[1]**3
    du2=p[2]*u[0]**3+p[3]*u[1]**3
    
    du=np.array([du1,du2])
    
    #p0=[-0.1,2,-2,-0.1]
    
    return du 

# =============================================================================
# Define the ODE for the Lorenz system
# =============================================================================
def Lorenz(u, t, p):
   du1 = p[0]*u[0]+p[1]*u[1]
   du2 = p[2]*u[0]+p[3]*u[0]*u[2]+p[4]*u[1]
   du3 = p[5]*u[0]*u[1]+p[6]*u[2]
   
   du=np.array([du1,du2,du3])
   
   #p0=[-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3]
   
   return du

# =============================================================================
# Define the ODE for the Rössler attractor
# =============================================================================
def Rossler(u, t, p):
   du1 = -u[1]-u[2]
   du2 = u[0]+p[0]*u[1]
   du3 = p[1]+u[0]*u[2]-p[2]*u[2]
   
   du=np.array([du1,du2,du3])
   
   #p0=[0.1,0.0,0.0]
   
   return du

def Lorenz96(x, t, p):
    # Lorenz 96 model
    # Compute state derivatives
    N=p[0]
    d = np.zeros(np.shape(x))
    # First the 3 edge cases: i=1,2,N
    d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
    d[1] = (x[2] - x[N-1]) * x[0] - x[1]
    d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
    # Then the general case
    for i in range(2, N-1):
        d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
    # Add the forcing term
    d = d + p[1]
    
    #p0=[10,8]
    
    # Return the state derivatives
    return d

# =============================================================================
# Define the library function for the SINDy
# =============================================================================
def Lib(x,libOrder):
    # First get the dimension of the x
    n,m=x.shape
    
    # print("The first dimension of the input data:",n)
    # print("The second dimension of the input data:",m)

    # Lib order 0
    # Theta=np.ones((n,1),dtype=float)
    
    
    Theta=x[:,[0]]
    
    # Lib order 1
    if libOrder>=1:
        for i in range(1,m):
            Theta=np.concatenate((Theta,x[:,[i]]),axis=1)
    
    if libOrder>=2:
        for i in range(m):
            for j in range(i,m):
                NewTerm=x[:,[i]]*x[:,[j]]
                Theta=np.concatenate((Theta,NewTerm),axis=1)
    
    if libOrder>=3:
        for i in range(m):
            for j in range(i,m):
                for k in range(j,m):
                    NewTerm=x[:,[i]]*x[:,[j]]*x[:,[k]]
                    Theta=np.concatenate((Theta,NewTerm),axis=1)
    
    if libOrder>=4:
        for i in range(m):
            for j in range(i,m):
                for k in range(j,m):
                    for ij in range(k,m):
                        NewTerm=x[:,[i]]*x[:,[j]]*x[:,[k]]*x[:,[ij]]
                        Theta=np.concatenate((Theta,NewTerm),axis=1)
    if libOrder>=5:
            for i in range(m):
                for j in range(i,m):
                    for k in range(j,m):
                        for ij in range(k,m):
                            for ik in range(ij,m):
                                NewTerm=x[:,[i]]*x[:,[j]]*x[:,[k]]*x[:,[ij]]*x[:,[ik]]
                                Theta=np.concatenate((Theta,NewTerm),axis=1)                       

    return Theta

# =============================================================================
# Define the library function for the SINDy: GPU version
# =============================================================================
@tf.function
def LibGPU(x):
    # =============================================================================
    # Following is the 3nd order lib for 2 dimentional system   
    # =============================================================================
    # For VanderPol, Duffing, and Cubic Osc
    # z1=tf.gather(x,[0],axis=1)
    # z2=tf.gather(x,[1],axis=1)
    
    # Theta=tf.concat([z1,z2,z1**2,z1*z2,z2**2,z1**3,(z1**2)*z2,z1*(z2**2),z2**3],axis=1)  
    
    # =============================================================================
    # Following is the 2nd order lib for 3 dimentional system   
    # =============================================================================
    # For Lorentz attractor
    z1=tf.gather(x,[0],axis=1)
    z2=tf.gather(x,[1],axis=1)
    z3=tf.gather(x,[2],axis=1)
    
    
    # Theta=tf.concat([z1,z2,z3,z1**2,z1*z2,z1*z3,z2**2,z2*z3,z3**2],axis=1) 

    Theta=tf.concat([z1,z2,z3,z1**2,z1*z2,z1*z3,z2**2,z2*z3,z3**2,
                      z1**3,(z1**2)*z2,(z1**2)*z3,z1*(z2**2),(z1*z2)*z3,z1*(z3**2),
                      z2**3,(z2**2)*z3,z2*(z3**2),z3**3],axis=1)   
    
    # =============================================================================
    # Following is the 3nd order lib for 3 dimentional system   
    # =============================================================================
    # For Rossler attractor
    # z1=tf.gather(x,[0],axis=1)
    # z2=tf.gather(x,[1],axis=1)
    # z3=tf.gather(x,[2],axis=1)
    
    
    # Theta=tf.concat([tf.ones_like(z1),z1,z2,z3,z1**2,z1*z2,z1*z3,z2**2,z2*z3,z3**2,
    #                   z1**3,(z1**2)*z2,(z1**2)*z3,z1*(z2**2),(z1*z2)*z3,z1*(z3**2),
    #                   z2**3,(z2**2)*z3,z2*(z3**2),z3**3],axis=1)     


    #----------- For Lorenz 96
    # First get the dimension of the x
    # n,m=x.shape
    
    # # Lib order 0
    # Theta=tf.ones_like(tf.gather(x,[0],axis=1))
    
    # # Lib order 1
    # for i in range(m):
    #     Theta=tf.concat((Theta,tf.gather(x,[i],axis=1)),1)

    # # Lib order 2
    # for i in range(m):
    #     for j in range(i,m):
    #         NewTerm=tf.gather(x,[i],axis=1)*tf.gather(x,[j],axis=1)
    #         Theta=tf.concat((Theta,NewTerm),1)                
    
    return Theta            
    
# =============================================================================
# Define a function that solves the Matlab version of backslash
# Reference: https://pythonquestion.com/post/how-can-i-obtain-the-same-special-solutions-to-underdetermined-linear-systems-that-matlab-s-a-b-mldivide-operator-returns-using-numpy-scipy/
# =============================================================================
def solve_minnonzero(A, b):
    x1, res, rnk, s = np.linalg.lstsq(A, b,rcond=None)
    if rnk == A.shape[1]:
        return x1   # nothing more to do if A is full-rank
    Q, R, P = qr(A.T, mode='full', pivoting=True)
    Z = Q[:, rnk:].conj()
    C = np.linalg.solve(Z[rnk:], -x1[rnk:])
    return x1 + Z.dot(C)


# =============================================================================
# Check whether the GPU is available
# =============================================================================
def CheckGPU():
    if tf.test.is_gpu_available():
        print("\n\n\n\n\n")
        print("The GPU is available")
        print("\n\n\n\n\n")
    else:
        print("\n\n\n\n\n")
        print("The GPU is not available")
        print("\n\n\n\n\n")
    return None
 
# =============================================================================
# Define the RK45 for the SINDy method
# =============================================================================
@tf.function
def RK45_F_SINDy(xin,LibGPU,Xi,dt):
    K1=tf.linalg.matmul(LibGPU(xin),Xi)*dt
        
    K2=tf.linalg.matmul(LibGPU(tf.math.add(xin,tf.constant(0.5)*K1)),Xi)*dt
        
    K3=tf.linalg.matmul(LibGPU(tf.math.add(xin,tf.constant(0.5)*K2)),Xi)*dt                        
        
    K4=tf.linalg.matmul(LibGPU(tf.math.add(xin,K3)),Xi)*dt

    return tf.math.add_n([xin,tf.constant(1/6)*K1,tf.constant(1/3)*K2,tf.constant(1/3)*K3,tf.constant(1/6)*K4])

# =============================================================================
# Define the RK45 for the SINDy method        
# =============================================================================
@tf.function
def RK45_B_SINDy(xin,LibGPU,Xi,dt):
    K1=-tf.linalg.matmul(LibGPU(xin),Xi)*dt
        
    K2=-tf.linalg.matmul(LibGPU(tf.math.add(xin,tf.constant(0.5)*K1)),Xi)*dt
        
    K3=-tf.linalg.matmul(LibGPU(tf.math.add(xin,tf.constant(0.5)*K2)),Xi)*dt                        
        
    K4=-tf.linalg.matmul(LibGPU(tf.math.add(xin,K3)),Xi)*dt

    return tf.math.add_n([xin,tf.constant(1/6)*K1,tf.constant(1/3)*K2,tf.constant(1/3)*K3,tf.constant(1/6)*K4])

# =============================================================================
# Define a function that will generate multiple initial condtion for the forward and backward simulation.
# The input tensor should be n x m where the n is the number of states and m is the time horizon.
# =============================================================================
def GetInitialCondition(X,q,n):
    if q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else:
        X0=X[q:n-q,:]

    return X0

# =============================================================================
# Define a function that will slice the data provided
# =============================================================================
def SliceData(Y,q,dataLen):
    if q==0:
        Ypre_F=Y
        Ypre_B=Y
    elif q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else: 
        Ypre_F=[]
        Ypre_B=[]
        for j in range(1,q+1):
            if j==1:
                Ypre_F=Y[q+j:dataLen-q+j,:]
                Ypre_B=Y[q-j:dataLen-q-j,:]
            else:
                Ypre_F=np.append(Ypre_F,Y[q+j:dataLen-q+j,:],axis=1)
                Ypre_B=np.append(Ypre_B,Y[q-j:dataLen-q-j,:],axis=1)      

    return Ypre_F,Ypre_B

# =============================================================================
# Define a function that will slice the noise variable into future and previous state
# =============================================================================
@tf.function
def SliceNoise(NoiseVar,q,dataLen,stateVar):
    if q==0:
        NoiseVar_F=Noise
        NoiseVar_B=Noise
    elif q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else: 
        NoiseVar_F=tf.slice(NoiseVar,[q+1,0],[dataLen-2*q,stateVar])
        NoiseVar_B=tf.slice(NoiseVar,[q-1,0],[dataLen-2*q,stateVar])

        for i in range(1,q):
            NoiseVar_F=tf.concat([NoiseVar_F,tf.slice(NoiseVar,[q+1+i,0],[dataLen-2*q,stateVar])],axis=1)
            NoiseVar_B=tf.concat([NoiseVar_B,tf.slice(NoiseVar,[q-1-i,0],[dataLen-2*q,stateVar])],axis=1)
            
    return NoiseVar_F,NoiseVar_B

# =============================================================================
# Define a function will calculate the prediction result given initial condition matrix and prediction step.
# This is the SINDy version
# =============================================================================
@tf.function
def Prediction_SINDy(X0,LibGPU,dt,q,stateVar,dataLen,Xi):
    if q==0:
        Xpre_F=X0
        Xpre_B=X0
    elif q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else:
        Xpre_F=RK45_F_SINDy(X0,LibGPU,Xi,dt)
        Xpre_B=RK45_B_SINDy(X0,LibGPU,Xi,dt)
        
        for i in range(q-1):
            Xpre_F=tf.concat([Xpre_F,RK45_F_SINDy(tf.slice(Xpre_F,[0,stateVar*i],[dataLen-2*q,stateVar]),LibGPU,Xi,dt)],axis=1)
            Xpre_B=tf.concat([Xpre_B,RK45_B_SINDy(tf.slice(Xpre_B,[0,stateVar*i],[dataLen-2*q,stateVar]),LibGPU,Xi,dt)],axis=1)
        
    return Xpre_F,Xpre_B

# =============================================================================
# Define a function that calculate the decay factor
# =============================================================================
def DecayFactor(ro,stateVar,q):
    if q==0:
        weights=1
    elif q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else:
        weights=[]
        for j in range(q):
            for i in range(stateVar):
                weights=np.append(weights,ro**(j))
    
    return weights

# # =============================================================================
# # Define a function that calculate the weighted mean suqare error
# # =============================================================================
@tf.function
def WeightMSE(Yt_F,Xp_F,Yt_B,Xp_B,NoiseVar_F,NoiseVar_B,weights): # weights, V
    # Calculate the prediction of noisy data
    Yp_F = Xp_F + NoiseVar_F
    Yp_B = Xp_B + NoiseVar_B
    
    # # Calculate the loss
    Jwmse_F=tf.reduce_mean(tf.math.multiply(tf.math.squared_difference(Yt_F,Yp_F),weights))
    Jwmse_B=tf.reduce_mean(tf.math.multiply(tf.math.squared_difference(Yt_B,Yp_B),weights))
    
    Jwmse=tf.math.add(Jwmse_F,Jwmse_B)
    
    return Jwmse

# =============================================================================
# Define the one step loss function for the noise signal separation: WmSINDy approach
# =============================================================================
@tf.function                                                                  
def OneStepLoss_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,LibGPU,optimizer,Tags,V, Vp, Grid): 
    with tf.GradientTape() as g:
        
        # First get the initial condition guess based on the measurement data and noise variable
        X0 = tf.math.subtract(Y0,tf.slice(NoiseVar,[q,0],[dataLen-2*q,stateVar]))
        
        # Then use the constraint
        Xi_var = tf.math.multiply(Xi,Xi_act)
        
        # Next, simulate the system forward and backward
        Xpre_F,Xpre_B = Prediction_SINDy(X0,LibGPU,dt,q,stateVar,dataLen,Xi_var)
        
        # Similarly get the forward and backward noise
        NoiseVar_F,NoiseVar_B = SliceNoise(NoiseVar,q,dataLen,stateVar)
        
        # Next calculate the weighted loss
        Jw = WeightMSE(Ypre_F,Xpre_F,Ypre_B,Xpre_B,NoiseVar_F,NoiseVar_B,weights)
        
        # First obtain the denoised data
        Xes = tf.math.subtract(Y,NoiseVar)
        ### Added by CL
        Jr = 1.0*Weight_Residual(Xes,Xi_var,stateVar,Tags,V,Vp,Grid)

        # Finally, add all the noise together
        J = tf.add(Jw,Jr)
        
    # Calculate the gradient with respect to the variables
    gard = g.gradient(J,[Xi,NoiseVar])
    optimizer.apply_gradients(zip(gard,[Xi,NoiseVar]))
        
    return J

@tf.function
def Weight_Residual(xin,Xi_var,stateVar,Tags,V,Vp,Grid):
        
    # wTheta = LibGPU(xin) 
    Theta_transposed = tf.transpose(LibGPU(xin))          
    
    b_Gw = []
    for nn in range(xin.shape[1]):
        y = []; v_n = []; vp_n = []; b = []; 
        
        grid_i = Grid.variables[nn]
        grid_i = tf.cast(grid_i, tf.int32)
        
        y = tf.reshape(xin[:, nn], [1, -1, 1])
        vp_n = tf.convert_to_tensor(Vp.variables[nn])
        vp_n = tf.reshape(vp_n, [len(vp_n), 1, 1])
        
        b = tf.nn.conv1d(y, vp_n, stride=1, padding='VALID')
        b = -tf.squeeze(b, axis=0)
        # b = tf.gather(b, grid_i)
        
        v_n = tf.convert_to_tensor(V.variables[nn])    
        v_n = tf.reshape(v_n, [len(v_n), 1, 1])

        # Reshape Theta to a 4D tensor
        Theta_4d = tf.reshape(Theta_transposed, [1] + Theta_transposed.shape.as_list() + [1])
        # Perform 2D convolution
        G_p = tf.nn.conv2d(Theta_4d, tf.reshape(v_n, [1, len(v_n), 1, 1]), strides=[1, 1, 1, 1], padding='VALID')
        # Reshape the result back to a 2D matrix
        G = tf.transpose(tf.reshape(G_p, G_p.shape[1:3]), perm=[1, 0])
        # G = tf.gather(G, grid_i)
    
        Gw = tf.linalg.matmul(G, tf.reshape(Xi_var[:,nn], (LibGPU(xin).shape[1], 1)))

        b_Gw.append(tf.reduce_mean(tf.math.squared_difference(b,Gw)))                                     
        
    Jresidual = tf.reduce_mean(b_Gw)                                           
    
    return Jresidual

# =============================================================================
# Now define the training function for the noise signal speration: SINDy approach
# =============================================================================
def Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,update_each,Tags, V, Vp, Grid): 
    
    start = time.time() 
    for i in range(update_each):   # Before: N_train
        # Calculate the cost and updte the gradient        
        J = OneStepLoss_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,LibGPU,optimizer,Tags,V, Vp, Grid) 
        
        # if i%1000==0:
        #     tf.print(J)
        # # tf.print(J)
        #     tf.print(i)
    
    totoalTime = time.time()-start
    
    return NoiseVar.numpy(),totoalTime

# =============================================================================
# Define a function that calculates the noise signal speration accuracy
# =============================================================================
def ID_Accuracy_SINDy(x,dx,Noise,NoiseID,LibGPU,Xi,dataLen,dt):
    Enoise_error=np.linalg.norm(Noise-NoiseID,'fro')**2/dataLen
    Evector_field_error=np.linalg.norm(dx-tf.linalg.matmul(LibGPU(tf.constant(x,dtype='float32')),Xi),'fro')**2/np.linalg.norm(dx,'fro')**2
    
    xpre=[]
    xpre=RK45_F_SINDy(tf.constant([x[0,:]],dtype="float32"),LibGPU,Xi,dt).numpy()
    
    try:
        for i in range(1,dataLen-1):
            dummy=RK45_F_SINDy(tf.constant([xpre[-1]],dtype="float32"),LibGPU,Xi,dt).numpy()
            xpre=np.append(xpre,[dummy[0]],axis=0)
                
        Epre_error=np.linalg.norm(x[1:]-xpre,'fro')**2/np.linalg.norm(x,'fro')**2
    except:
        print("The simulation blows up...Current Neural Network is not stable...")
        Epre_error=float('nan')

    return Enoise_error,Evector_field_error,Epre_error,xpre

# =============================================================================
# Define a function that will generate gasiian 
# =============================================================================
def Gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# =============================================================================
# This code is used for approximate the noise that we added into the signal.
# Reference: https://github.com/snagcliffs/RKNN
# =============================================================================
def approximate_noise(Y, lam):
	n,m = Y.shape

	D = np.zeros((m,m))
	D[0,:4] = [2,-5,4,-1]
	D[m-1,m-4:] = [-1,4,-5,2]

	for i in range(1,m-1):
	    D[i,i] = -2
	    D[i,i+1] = 1
	    D[i,i-1] = 1
	    
	D = D.dot(D)

	X_smooth = np.vstack([np.linalg.solve(np.eye(m) + lam*D.T.dot(D), Y[j,:].reshape(m,1)).reshape(1,m) for j in range(n)])

	N_hat = Y-X_smooth

	return N_hat, X_smooth
