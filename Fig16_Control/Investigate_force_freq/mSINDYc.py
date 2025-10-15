# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:42:36 2025

@author: crist
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
import tensorflow as tf
import matplotlib.pyplot as plt
from utils_NSS_SINDYc import * # Import the fixed utility functions
import time
from datetime import datetime

#%% Define how many percent of noise you need
NoisePercentage = 0

#%% Simulate
# Define the random seed for the noise generation
np.random.seed(4)

def sphs(Pf, K, t):
    i = np.arange(1, K + 1)
    theta = 2 * np.pi / K * np.cumsum(i)
    u = np.sqrt(2 / K) * np.cos((2 * np.pi * i[:, np.newaxis] * t / Pf + theta[:, np.newaxis])/10)    # reduce freq: 10
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
    x0 = np.array([-8, 8, 27])                     # From Kaiser: x0 
    dt = 0.002
    tspan = np.arange(0, 10 + dt, dt)
    
    # Forcing parameters
    Pf = 0.5
    K = 8
    A = 10
    
    # Calculate forcing term for all time points
    u = A * sphs(Pf, K, tspan)
    
    # Create wrapper for solve_ivp that includes forcing term
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
        rtol=1e-10,
        atol=1e-10 * np.ones(3),
        method='RK45',
        dense_output=True
    )
    
    # Extract solution
    t = sol.t
    x = sol.y.T
    
    # Take first half of data for training
    # Ntrain = (len(t) - 1) // 2 + 1
    # x = x[:Ntrain]
    # u = u[:Ntrain]
    # t = t[:Ntrain]
    
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

#%% Prepare data for SINDYc

# Get data dimensions
stateVar, dataLen = np.transpose(x).shape

# Generate noise if desired
NoiseMag = [np.std(x[:,i]) * NoisePercentage * 0.01 for i in range(stateVar)]
Noise = np.hstack([NoiseMag[i] * np.random.randn(dataLen, 1) for i in range(stateVar)])

# Add noise to the data
xn = x + Noise

# SINDy algorithm parameters
N_SINDy_Iter = 15
disp = 0
NormalizeLib = 0
libOrder = 2
lam = 0.2

# Use hard start (no initial smoothing)
softstart = 0

#%% Visualize the data and noise

# Plot state trajectories with noise
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(t, x[:, 0], linewidth=0.5, color='k')
plt.scatter(t, xn[:, 0], s=0.5, color='b')
plt.ylabel('x')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, x[:, 1], linewidth=0.5, color='k')
plt.scatter(t, xn[:, 1], s=0.5, color='b')
plt.ylabel('y')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, x[:, 2], linewidth=0.5, color='k')
plt.scatter(t, xn[:, 2], s=0.5, color='b')
plt.ylabel('z')
plt.xlabel('t')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3D phase plot
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(x[:, 0], x[:, 1], x[:, 2], color='black', linewidth=1.5, label='True')
ax.plot3D(xn[:, 0], xn[:, 1], xn[:, 2], color='red', linestyle='--', linewidth=1, label='Noisy')
ax.view_init(30, -60)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.title('Lorenz Attractor')
plt.show()

#%% Set up TensorFlow for SINDYc

# Check GPU availability
# CheckGPU()

# Define data type
dataType = tf.dtypes.float32

# Define the prediction step
q = 1

# Define measurement data as tensors
Y = tf.constant(xn, dtype=dataType)
Y0 = tf.constant(GetInitialCondition(xn, q, dataLen), dtype=dataType)

u = u.reshape(-1, 1)
U = tf.constant(u,dtype=dataType)                                              # Added by CL
U0 = tf.constant(u[q:dataLen-q],dtype=dataType)                                # Added by CL

# Get forward and backward measurement data
Ypre_F, Ypre_B = SliceData(xn, q, dataLen)
Ypre_F = tf.constant(Ypre_F, dtype=dataType)
Ypre_B = tf.constant(Ypre_B, dtype=dataType)

# Define weights for the error
ro = 0.9
weights = tf.constant(DecayFactor(ro, stateVar, q), dtype=dataType)

# Initialize noise estimate
if softstart == 1:
    # Soft Start: Use approximate noise from smoothing
    NoiseEs, xes = approximate_noise(np.transpose(xn), 20)
    NoiseEs = np.transpose(NoiseEs)
    xes = np.transpose(xes)
else:
    # Hard Start: Initialize with zeros
    NoiseEs = np.zeros((xn.shape[0], xn.shape[1]))
    xes = xn - NoiseEs

# Calculate derivatives of denoised data
dxes = CalDerivative(xes, dt, 1)

# Create noise variable for optimization
NoiseVar = tf.Variable(NoiseEs, dtype=dataType)

# Construct library with state and control inputs
xaug = np.concatenate((xes, u), axis=1)
Theta = Lib(xaug, libOrder)

# Apply SINDy to find sparse coefficients
Xi0 = SINDy(Theta, dxes, lam, N_SINDy_Iter, disp, NormalizeLib)

print("Initial SINDy coefficients:")
print(Xi0)

# Initialize variable for optimization
Xi = tf.Variable(Xi0, dtype=dataType)

# Set active terms matrix (all terms initially active)
Xi_act = tf.constant(np.ones(Xi0.shape), dtype=dataType)

#%% Plot initial noise estimate (all zeros at start)
StartIndex = 0  # Starting index for visualization
EndIndex = 200  # Ending index for visualization

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.title("Initial Noise Estimate")
plt.plot(t[StartIndex:EndIndex], Noise[StartIndex:EndIndex, 0], linewidth=1.5, color='b')
plt.plot(t[StartIndex:EndIndex], NoiseVar.numpy()[StartIndex:EndIndex, 0], linewidth=1.5, color='k', linestyle='--')
plt.ylabel('Noise in x')
plt.legend(['True Noise', 'Estimated Noise'])
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t[StartIndex:EndIndex], Noise[StartIndex:EndIndex, 1], linewidth=1.5, color='b')
plt.plot(t[StartIndex:EndIndex], NoiseVar.numpy()[StartIndex:EndIndex, 1], linewidth=1.5, color='k', linestyle='--')
plt.ylabel('Noise in y')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t[StartIndex:EndIndex], Noise[StartIndex:EndIndex, 2], linewidth=1.5, color='b')
plt.plot(t[StartIndex:EndIndex], NoiseVar.numpy()[StartIndex:EndIndex, 2], linewidth=1.5, color='k', linestyle='--')
plt.ylabel('Noise in z')
plt.xlabel('Time')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Define optimizer and run SINDYc algorithm
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07)

# Number of iterations
Nloop = 2#5  # Outer loop iterations (SINDy updates)
N_train = 5#5000  # Inner loop iterations (optimization)

# Lists to store results
NoiseIDList = []
TrainTimeList = np.zeros((Nloop, 1))
Enoise_error_List = np.zeros((Nloop, 1))
Evector_field_error_list = np.zeros((Nloop, 1))
Epre_error_list = np.zeros((Nloop, 1))
x_sim_list = []
Xi_List = []

# Define the true parameter matrix for comparison
Xi_base = np.zeros((14, 3))  # 14 library terms (no constant)
# Set true Lorenz parameters with control 
Xi_base[0, 0] = -10;  Xi_base[1, 0] = 10; Xi_base[3, 0] = 1       
Xi_base[0, 1] = 28;   Xi_base[1, 1] = -1; Xi_base[2, 1] = 0; Xi_base[6, 1] = -1      
Xi_base[2, 2] = -8/3; Xi_base[5, 2] = 1 

#%% Run the SINDYc algorithm with iterative updates
for k in range(Nloop):
    print(f"\nRunning iteration {k+1}/{Nloop}")
    
    # Train to identify noise
    NoiseID, totalTime = Train_NSS_SINDy( Y, Y0, Ypre_F, Ypre_B, NoiseVar, Xi, Xi_act, 
                                          weights, dt, q, stateVar, dataLen, optimizer, N_train, U, U0)
    
    print(f"\tTraining completed in {totalTime:.2f} seconds")
    
    # Process the denoised data
    xes = xn - NoiseID
    xes = xes[q+1:-q-1, :]  # Remove boundaries for derivative calculation
    dxes = CalDerivative(xes, dt, 1)
    
    # Extract control inputs for the same time window
    u = u.reshape(-1, 1)
    ues = u[q+1:-q-1]
    
    print("Current Xi result:")
    print(Xi.numpy())
    
    # Construct library with denoised states and control
    xaug = np.concatenate((xes, ues.reshape(-1, 1)), axis=1)
    Theta = Lib(xaug, libOrder)
    
    # Apply sparsity-promoting optimization
    index_min = abs(Xi.numpy()) > lam
    Xi_act_dum = Xi_act.numpy() * index_min.astype(int)
    Xi_num = Xi.numpy() * Xi_act_dum
    index_min = Xi_act_dum.astype(bool)
    
    # Perform regression on active terms
    for r in range(xes.shape[1]):
        if np.any(index_min[:, r]):
            Xi_num[index_min[:, r], r] = solve_minnonzero(
                Theta[:, index_min[:, r]], dxes[:, r]
            )
    
    print("Updated Xi result:")
    print(Xi_num)
    
    # Update active terms and coefficients
    Xi_act = tf.constant(Xi_act_dum, dtype=dataType)
    Xi = tf.Variable(Xi_num, dtype=dataType)
    
    # Evaluate identification accuracy
    Enoise_error, Evector_field_error, Epre_error, x_sim = ID_Accuracy_SINDy(
                                                         x, dx, Noise, NoiseID, LibGPU, Xi, dataLen, dt, u)
    
    # Print performance metrics
    print(f"\tNoise identification error: {Enoise_error:.6f}")
    print(f"\tVector field approximation error: {Evector_field_error:.6f}")
    print(f"\tTrajectory prediction error: {Epre_error:.6f}")
    
    # Store results
    NoiseIDList.append(NoiseID)
    x_sim_list.append(x_sim)
    TrainTimeList[k] = totalTime
    Enoise_error_List[k] = Enoise_error
    Evector_field_error_list[k] = Evector_field_error
    Epre_error_list[k] = Epre_error
    Xi_List.append(Xi.numpy())

#%% Visualize final results

# Use the final results
NoiseID = NoiseIDList[-1]
x_sim = x_sim_list[-1]
Xi_final = Xi_List[-1]

# 1. Plot identified noise vs true noise
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.title("Noise Identification Results")
plt.plot(t[StartIndex:EndIndex], Noise[StartIndex:EndIndex, 0], linewidth=1.5, color='b')
plt.plot(t[StartIndex:EndIndex], NoiseID[StartIndex:EndIndex, 0], linewidth=1.5, color='k', linestyle='--')
plt.ylabel('Noise in x')
plt.legend(['True Noise', 'Identified Noise'])
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t[StartIndex:EndIndex], Noise[StartIndex:EndIndex, 1], linewidth=1.5, color='b')
plt.plot(t[StartIndex:EndIndex], NoiseID[StartIndex:EndIndex, 1], linewidth=1.5, color='k', linestyle='--')
plt.ylabel('Noise in y')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t[StartIndex:EndIndex], Noise[StartIndex:EndIndex, 2], linewidth=1.5, color='b')
plt.plot(t[StartIndex:EndIndex], NoiseID[StartIndex:EndIndex, 2], linewidth=1.5, color='k', linestyle='--')
plt.ylabel('Noise in z')
plt.xlabel('Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. 3D Phase plot of original and denoised trajectories
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(x[:, 0], x[:, 1], x[:, 2], color='black', linewidth=2, label='True')
ax.plot3D((xn-NoiseID)[:, 0], (xn-NoiseID)[:, 1], (xn-NoiseID)[:, 2], 
          color='orange', linestyle='--', linewidth=1.5, label='Denoised')
ax.view_init(30, -60)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.title('Original vs Denoised Trajectories')
plt.show()

# 3. Plot predictions vs true trajectories
preLen = 200  # Length for prediction visualization
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.title('State Prediction')
plt.plot(t[1:preLen+1], x[1:preLen+1, 0], color='blue', linewidth=1.5, label='True')
plt.plot(t[1:preLen+1], x_sim[0:preLen, 0], color='red', linestyle='--', linewidth=1.5, label='Predicted')
plt.ylabel('x')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t[1:preLen+1], x[1:preLen+1, 1], color='blue', linewidth=1.5, label='True')
plt.plot(t[1:preLen+1], x_sim[0:preLen, 1], color='red', linestyle='--', linewidth=1.5, label='Predicted')
plt.ylabel('y')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t[1:preLen+1], x[1:preLen+1, 2], color='blue', linewidth=1.5, label='True')
plt.plot(t[1:preLen+1], x_sim[0:preLen, 2], color='red', linestyle='--', linewidth=1.5, label='Predicted')
plt.ylabel('z')
plt.xlabel('Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Plot prediction in 3D
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(x[1:preLen+1, 0], x[1:preLen+1, 1], x[1:preLen+1, 2], 
          color='blue', linewidth=2, label='True')
ax.plot3D(x_sim[0:preLen, 0], x_sim[0:preLen, 1], x_sim[0:preLen, 2], 
          color='red', linestyle='--', linewidth=1.5, label='Predicted')
ax.scatter([x[0, 0]], [x[0, 1]], [x[0, 2]], 
           color='green', marker='o', s=100, label='Initial Point')
ax.view_init(30, -60)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.title('True vs Predicted Trajectory')
plt.show()

# 5. Plot error metrics over iterations
plt.figure(figsize=(10, 8))
iterations = np.arange(1, Nloop+1)

plt.subplot(3, 1, 1)
plt.plot(iterations, Enoise_error_List, 'o-', linewidth=2)
plt.ylabel('Noise Error')
plt.grid(True)
plt.title('Convergence of Error Metrics')

plt.subplot(3, 1, 2)
plt.plot(iterations, Evector_field_error_list, 'o-', linewidth=2)
plt.ylabel('Vector Field Error')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(iterations, Epre_error_list, 'o-', linewidth=2)
plt.ylabel('Prediction Error')
plt.xlabel('Iteration')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Compare identified parameters with true parameters
# Calculate parameter error
ParameterError = np.linalg.norm(Xi_base - Xi_final, 2) / np.linalg.norm(Xi_base, 2)
print(f"Parameter identification error: {ParameterError:.6f}")
