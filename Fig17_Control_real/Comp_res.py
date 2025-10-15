# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:25:31 2025

@author: crist
"""

#%% Import packages
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import interp1d
from utils_NSS_SINDy_Duffing import *
from scipy.integrate import solve_ivp


#%% Import Data
# https://github.com/Samuel-Ayankoso/Neural-Networks-and-ML--Geared-DC-Motor-Case-Study/blob/main/DC%20Motor%20PySINDy_model.ipynb

data = scipy.io.loadmat('exp_data.mat')

hop = 1
t_data   = data['exp_data'][:,0][::hop]
theta_tr  = data['exp_data'][:,2][::hop]
U_data      = data['exp_data'][:,1][::hop]
x = np.expand_dims(theta_tr, axis=1)  

dt = float(t_data[1] - t_data[0])
 
#%% Plot the system

lw=5

plt.figure()
pp1=plt.plot(x,linewidth=lw,color='k',linestyle='-')
plt.ylabel('x')
plt.grid(False)
plt.axis('off')
 
#%% Test the SINDy
N_SINDy_Iter=15
disp=0
NormalizeLib=0
libOrder=5
lam=0.001

dxes=CalDerivative(x,dt,1)   # Dimension added

x_both = np.column_stack((theta_tr, U_data))                           #-------------
# Get the initial guess of the SINDy parameters
Theta=Lib(x_both,libOrder)

Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)
print(Xi0)

#%%

# Interpolate U(t) for arbitrary time input
U_func = interp1d(t_data, U_data, kind='linear', fill_value='extrapolate')
psi0  = theta_tr[0]

# Define ODE function (from mSINDy)
def odefun_mS(t, psi_os):
    psi = psi_os[0]  # Extract scalar from array
    U_t = U_func(t)
    
    dpsi_dt = (-58.7 - 0.700 * psi + 3.5 * U_t + 0.00353 * psi**2 
               - 0.0157 * psi * U_t + 0.00402 * U_t**2)
    
    return [dpsi_dt]

def odefun_Wm(t, psi_os):
    psi = psi_os[0]  # Extract scalar from array
    U_t = U_func(t)
    
    dpsi_dt = (-53.9 - 0.59 * psi + 3.1 * U_t + 0.00295 * psi**2 
               - 0.0136 * psi * U_t + 0.00343 * U_t**2)
    
    return [dpsi_dt]

# Solve ODEs
sol_mS = solve_ivp(odefun_mS, [t_data[0], t_data[-1]], [psi0], 
                rtol=1e-12, atol=1e-12, dense_output=True)
sol_Wm = solve_ivp(odefun_Wm, [t_data[0], t_data[-1]], [psi0], 
                rtol=1e-12, atol=1e-12, dense_output=True)

# Extract solution
t_sol_mS = sol_mS.t
t_sol_Wm = sol_Wm.t

psi_sol_mS = sol_mS.y[0]
psi_sol_Wm = sol_Wm.y[0]

# Interpolate simulated psi onto original time
psi_interp_func_mS = interp1d(t_sol_mS, psi_sol_mS, kind='linear', fill_value='extrapolate')
psi_interp_func_Wm = interp1d(t_sol_Wm, psi_sol_Wm, kind='linear', fill_value='extrapolate')

psi_interp_mS = psi_interp_func_mS(t_data)
psi_interp_Wm = psi_interp_func_Wm(t_data)

# Plot results
plt.figure()
plt.plot(t_data, theta_tr, 'k', label='real')
plt.plot(t_data, psi_interp_mS, 'b', label='mSINDy')
plt.plot(t_data, psi_interp_Wm, 'r', label='WmSINDy')
plt.ylabel(r'$\psi_{os}$')
plt.legend()
plt.show()

#%%
res_mS = scipy.io.loadmat('results_mS.mat')
NoiseID_mS = np.squeeze(res_mS['NoiseID'])
signal_mS = psi_interp_mS + NoiseID_mS

res_Wm = scipy.io.loadmat('results_Wm.mat')
NoiseID_Wm = np.squeeze(res_mS['NoiseID'])
signal_Wm = psi_interp_Wm + NoiseID_Wm

Err_mS = np.mean((theta_tr-signal_mS)**2)
Err_Wm = np.mean((theta_tr-signal_Wm)**2)

print("\t Error using mSINDy ",Err_mS)
print("\t Error using WmSINDy ",Err_Wm)
