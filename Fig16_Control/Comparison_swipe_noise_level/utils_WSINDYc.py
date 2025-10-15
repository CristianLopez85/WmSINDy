# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:42:20 2023

@author: crist
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.io
from scipy.signal import convolve2d

####### From: utils_NSS_SINDy_Wc

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
#%%
def LibGPU_WSINDy(xnoisy, u, libOrder):
    
    xaug = np.concatenate((xnoisy, u), axis=1)                                       # Augmented by CL
    
    Theta = Lib(xaug,libOrder)    
    
    return Theta

#%%
def findcorners(xn,t,tau,tauhat,phi_class):
    t = t.reshape(-1, 1)
    T = len(t)
    n = xn.shape[1]

    mts0 = np.zeros(n)
    pts0 = np.zeros(n)
    
    def l(m, k, N):
        return np.log((2*m - 1) / m**2)*(4*np.pi**2*k**2*m**2 - 3*N**2*tauhat**2) - 2*N**2*tauhat**2*np.log(tau)
               
    from scipy.optimize import root_scalar

    def root_function(m):
        return l(m, k, T)
    
    for nn in range(n):
        corner = findcornerpts(xn[:, nn], t)
        k = corner[1]
        
        # if phi_class == 1:
            # Find the root of the function using the specified interval
        result = root_scalar(root_function, method='brentq', bracket=[1, 2 / np.sqrt(tau)])
        mnew = result.root

        if mnew > T/2 - 1:
            mnew = T / (2 * k)

        mts0[nn] = min(np.floor((T - 1) / 2), np.ceil(mnew))
        pts0[nn] = max(2, np.floor(np.log(tau) / np.log(1 - (1 - 1 / mts0[nn]) ** 2)))
            
    return mts0,pts0

#%%
def findcornerpts(xn,t):
    t = t.reshape(-1, 1)
    T = len(t)
    
    wn = (np.arange(T) - np.floor(T/2)) * (2 * np.pi) / np.ptp(t)
    xx = wn[:int(np.ceil(len(wn) / 2))]
    NN = len(xx)
    Ufft = np.abs(np.fft.fftshift(np.fft.fft(xn))) / np.sqrt(2 * NN)            ## Removed mean by CL

    Ufft = Ufft[:int(np.ceil(T / 2))]
    Umax = np.argmax(Ufft)

    xx1 = xx[:Umax+1]
    Ufft1 = np.cumsum(np.abs(Ufft[:Umax+1]))                                       # +1 to match 

    tstarind1 = getcorner(Ufft1, xx1)
    tstarind2 = getcorner(np.log(np.abs(Ufft[:Umax+1])), xx1)
    tstarind = int(np.floor((tstarind1 + tstarind2) / 2))

    tstar = -xx[tstarind]
    corner = [tstar, max(Umax+1 - tstarind, 1)]                                
    
    return corner

def getcorner(Ufft2, xx1):
    NN = len(Ufft2)
    Ufft11 = Ufft2 / np.max(np.abs(Ufft2)) * NN
    errs = np.zeros(NN)
    
    for k in range(2, NN-1):
        try:
            L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2 = build_lines(Ufft11, xx1, k)
            errs[k-1] = np.sqrt(np.sum(((L1 - Ufft_av1) / Ufft_av1) ** 2) + np.sum(((L2 - Ufft_av2) / Ufft_av2) ** 2))
        except IndexError:
            continue  # Skip to the next iteration if IndexError occurs
    max_err = np.max(errs)                                                 ## To avoid 0 values, by CL
    errs[errs == 0] = max_err
    tstarind = np.argmin(errs)
    
    return tstarind

def build_lines(Ufft11, xx1, k):
    NN3 = len(Ufft11)
    subinds1 = range(k)
    subinds2 = range(k-1, NN3)
    Ufft_av1 = Ufft11[subinds1]
    Ufft_av2 = Ufft11[subinds2]
    
    xx_sb1 = xx1[subinds1]
    xx_sb2 = xx1[subinds2]
    
    m1, b1, L1 = lin_regress(Ufft_av1, xx_sb1)
    m2, b2, L2 = lin_regress(Ufft_av2, xx_sb2)

    return L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2

# def lin_regress(U, x):
#     if x[-1] - x[0] != 0:
#         m = (U[-1] - U[0]) / (x[-1] - x[0])
#     else:
#         m = np.nan
#     b = U[0] - m * x[0]
#     L = U[0] + m * (x - x[0])
#     return m, b, L

def lin_regress(U, x):
    x_diff = x[-1] - x[0]
    if x_diff != 0:
        m = (U[-1] - U[0]) / x_diff
    else:
        m = tf.constant(float('nan'))
    b = U[0] - m * x[0]
    L = U[0] + m * (x - x[0])
    return m, b, L

#%%
def gen_data(ode_num, ode_params, tspan, x0, tol_ode):
    # Simulate ODE
    ode_names = ['Linear', 'Lorenz', 'Duffing']
    ode_name = ode_names[ode_num]
    
    if ode_name == 'Linear':
        # ll-dim
        ll = 2
        x = np.zeros(ll)
        y = np.zeros(ll)
        x[[2, -1]] = [1, -1]
        y[[2, -1]] = [-1, 1]
        A = np.flipud(np.fliplr(np.outer(y, x)))
        np.fill_diagonal(A, -0.2)
        
        if ode_params is None:
            ode_params = [A]
        
        if x0 is None:
            x0 = np.zeros(ll)
            x0[0] = 10
        
        if tspan is None:
            tspan = np.arange(0, 40, 0.025)
            
    elif ode_name == 'Lorenz':
        if ode_params is None:
            ode_params = [10, 8/3, 28]
        
        if x0 is None:
            x0 = np.array([-8, 7, 27])
        
        if tspan is None:
            tspan = np.arange(0, 10.001, 0.001)           ## Added .001 to 10, to get 10k+1 pointsby CL
            
    elif ode_name == 'Duffing':
        if ode_params is None:
            ode_params = [0.2, 0.2, 1]
        
        if x0 is None:
            x0 = np.array([0, 2])
        
        if tspan is None:
            tspan = np.arange(0, 30.01, 0.01)           ## Added .001 to 10, to get 10k+1 pointsby CL
    
    weights, t, x, rhs = sim_ode(x0, tspan, tol_ode, ode_name, ode_params)
    
    return weights, t, x, x0, ode_name, ode_params, rhs


#%%
def gen_noise(U_exact, sigma_NR, noise_dist, noise_alg):
    if noise_alg == 0:  # additive
        stdv = (np.sqrt(np.mean(U_exact**2)))**2
    elif noise_alg == 1:  # multiplicative
        stdv = 1
    
    dims = U_exact.shape
    
    if noise_dist == 0:  # white noise
        if sigma_NR > 0:
            sigma = sigma_NR * np.sqrt(stdv)
        else:
            sigma = -sigma_NR
        noise = np.random.normal(0, sigma, dims)
        
    elif noise_dist == 1:  # uniform noise
        if sigma_NR > 0:
            sigma = np.sqrt(3 * sigma_NR**2 * stdv)
        else:
            sigma = -sigma_NR
        noise = sigma * (2 * np.random.random(dims) - 1)
    
    if noise_alg == 0:  # additive
        U = U_exact + noise
    elif noise_alg == 1:  # multiplicative
        U = U_exact * (1 + noise)
    
    noise_ratio_obs = np.linalg.norm(U - U_exact) / np.linalg.norm(U_exact)
    
    return U, noise, noise_ratio_obs, sigma

#%%
def get_optimal_SMAF(tobs, fx_obs, max_points, init_m_fac, max_filter_fac, expand_fac, maxits, deriv_tol, verbose):

    if max_points is None:
        max_points = 10**5
    if init_m_fac is None:
        init_m_fac = 200
    if max_filter_fac is None:
        max_filter_fac = 8
    if expand_fac is None:
        expand_fac = 2
    if maxits is None:
        maxits = 100
    if deriv_tol is None:
        deriv_tol = 10**-6
    if verbose is None:
        verbose = 0

    def estimate_sigma(f):

        Ih = (f[3:-1] - f[1:-3]) / 2
        I2h = (f[4:] - f[:-4]) / 4
        sig = np.sqrt(8/5) * np.sqrt(np.mean((Ih - I2h)**2))

        if sig < 0.01:
            I4h = (f[8:] - f[:-8]) / 8
            Ih_1 = 4/3 * (Ih - 1/4 * I2h)
            I2h_1 = 4/3 * (I2h[2:-2] - 1/4 * I4h)
            sig = np.sqrt(576/714) * np.sqrt(np.mean((Ih_1[2:-2] - I2h_1)**2))
        
        return sig

    # fx_obs = xobs[:,0] 
    sigma_est = estimate_sigma(fx_obs);
    subsamp = max(int(np.floor(len(tobs) / max_points)), 1)

    import math

    fx_subsamp = fx_obs[::subsamp]
    M = fx_subsamp.shape[0]
    dx_subsamp = np.mean(np.diff(tobs[::subsamp]))
    m = math.ceil(M / init_m_fac)
    max_filter_width = math.floor(M / max_filter_fac)

    its = 1; check = 1
    m = min(m, max_filter_width)

    from scipy.signal import convolve
    from scipy.optimize import root_scalar
    options = {'xtol': 1e-6, 'maxiter': 1000}  # Adjust the options as needed


    from scipy.linalg import pinv

    def build_poly_kernel(deg, k, n, dx, max_dx):
        x = np.arange(-n, n+1) * dx
        X = np.power(x.reshape(-1, 1), np.arange(deg+1))
        K = k(x / (n*dx))
        K = K / np.linalg.norm(K, ord=1)
        A = pinv(np.sqrt(K)[:, np.newaxis] * X) * np.sqrt(K)[:, np.newaxis].T
        M = np.concatenate((np.diag(np.arange(max_dx+1)), np.zeros((max_dx+1, deg-max_dx))), axis=1)
        f = np.dot(M, A)
        return f, A

    while check > 0 and its < maxits:
        # if verbose:
            #     tic = time.time()
    
        _, A = build_poly_kernel(2, lambda x: x*0+1, min(max(math.floor(m*expand_fac), 3), math.floor((M-1)/2)), dx_subsamp, 0)
    
        d = 2 * np.mean(np.abs(convolve(fx_subsamp, A[2, :], 'valid')))
        ## Missing:  d = 2*mean(reshape(abs(conv2(A(3,:),1,fx_subsamp,'valid')),[],1));
    
        C = sigma_est**2 / ((d + deriv_tol)**2 * dx_subsamp**4 / 144)
    
        def f(n):
            return n**5 - n**3 - C

        eq_f_solve = root_scalar(f, method='brentq', bracket=[0.1, max_filter_width], options=options).root    # CL: fsolve seems not working
        mnew = min(math.floor((eq_f_solve - 1) / 2), max_filter_width)
    
        check = abs(m - mnew)
        m = mnew
        its += 1
    
        # if verbose:
            #     print(time.time() - tic, m, d)

    m = m * subsamp
    W = 1 / (2 * m + 1) * np.ones((2 * m + 1, 1))

    return W


#%%
def get_tags_SINDy(n):
    
    if n ==2:
        tags = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [2,0,0,0], [1,1,0,0], [1,0,1,0], [1,0,0,1] ,
                         [0,2,0,0], [0,1,1,0], [0,1,0,1], [0,0,2,0], [0,0,1,1], [0,0,0,2]])
        
        # tags = np.array([[0,0,0] , [1,0,0] , [0, 1, 0] ,[0, 0, 1] , [2, 0, 0] , [1, 1, 0] ,[1, 0 ,1] ,[0 ,2 ,0] ,
        #  [0, 1, 1] , [0, 0, 2] , [3, 0, 0] ,[2, 1, 0] , [2, 0, 1] , [1, 2, 0] ,[1, 1, 1] ,[1, 0, 2] ,
        #  [0, 3, 0] , [0, 2, 1] , [0, 1, 2] ,[0, 0, 3]])
    if n ==3:
        tags = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [2,0,0,0], [1,1,0,0], [1,0,1,0], [1,0,0,1] ,
                         [0,2,0,0], [0,1,1,0], [0,1,0,1], [0,0,2,0], [0,0,1,1], [0,0,0,2], [3,0,0,0], [2,1,0,0] ,
                         [2,0,1,0], [2,0,0,1], [1,2,0,0], [1,1,1,0], [1,1,0,1], [1,0,2,0], [1,0,1,1], [1,0,0,2] ,
                         [0,3,0,0], [0,2,1,0], [0,2,0,1], [0,1,2,0], [0,1,1,1], [0,1,0,2], [0,0,3,0], [0,0,2,1] ,
                         [0,0,1,2], [0,0,0,3]])

    elif n == 5:
        tags = np.array([[0,0,0] , [1,0,0] , [0, 1, 0] ,[0, 0, 1] , [2, 0, 0] , [1, 1, 0] ,[1, 0 ,1] ,[0 ,2 ,0] ,
         [0, 1, 1] , [0, 0, 2] , [3, 0, 0] ,[2, 1, 0] , [2, 0, 1] , [1, 2, 0] ,[1, 1, 1] ,[1, 0, 2] ,
         [0, 3, 0] , [0, 2, 1] , [0, 1, 2] ,[0, 0, 3] , [4, 0, 0] , [3, 1, 0] ,[3, 0, 1] ,[2, 2, 0],
         [2, 1, 1] , [2, 0, 2] , [1, 3, 0] ,[1, 2, 1] , [1, 1, 2] , [1, 0, 3] ,[0, 4, 0] ,[0, 3, 1] ,
         [0, 2, 2] , [0, 1, 3] , [0, 0, 4] ,[5, 0, 0] , [4, 1, 0] , [4, 0, 1] ,[3, 2, 0] ,[3, 1, 1],
         [3, 0, 2] , [2, 3, 0] , [2, 2, 1] ,[2, 1, 2] , [2, 0, 3] , [1, 4, 0] ,[1, 3, 1] ,[1, 2, 2] ,
         [1, 1, 3] , [1, 0, 4] , [0, 5, 0] ,[0, 4, 1] , [0, 3, 2] , [0, 2, 3] ,[0, 1, 4] ,[0, 0, 5]])
    return tags

#%%
def get_true_weights(weights, tags, n):

  if not weights:
    return []

  true_nz_weights = np.zeros((len(tags), n))
  for i in range(len(weights)):
    weights_i = weights[i]
    l1, l2 = weights_i.shape
    for j in range(l1):
      true_nz_weights[np.all(weights_i[j, :l2 - 1] == tags, axis=1), i] = weights_i[j, l2-1]

  return true_nz_weights


#%%
def phi_int_weights(m, d, tol):
    m = int(m)
    d = int(d)

    if tol < 0:
        p = -tol
    else:
        p = np.ceil(max(np.log(tol) / np.log((2 * m - 1) / m ** 2), d + 1))

    t = np.linspace(0, 1, m + 1)
    t_L = np.zeros((d + 1, m + 1))
    t_R = np.zeros((d + 1, m + 1))

    for j in range(m):
        t_L[:, j] = (1 + t[j]) ** np.flip(np.arange(p - d, p + 1))
        t_R[:, j] = (1 - t[j]) ** np.flip(np.arange(p - d, p + 1))

    ps = np.ones((d + 1,))
    for q in range(1, d + 1):
        ps[q] = (p - q + 1) * ps[q - 1]

    t_L = ps[:, np.newaxis] * t_L
    t_R = ((-1) ** np.arange(d + 1)[:, np.newaxis] * ps[:, np.newaxis]) * t_R

    Cfs = np.zeros((d + 1, 2 * m + 1))
    Cfs[0, :] = np.concatenate((np.flip(t_L[0, :]) * np.flip(t_R[0, :]), t_L[0, 1:] * t_R[0, 1:]))
   
    P = np.fliplr(scipy.linalg.pascal(d+1, kind='symmetric', exact=True))

    for k in range(1, d+1):
        binoms = np.diag(P, d-k)
        Cfs_temp = np.zeros(int(m) + 1)
        
        for j in range(k+1):
            Cfs_temp = Cfs_temp + binoms[j] * t_L[k-j,:] * t_R[j,:]
    
        Cfs_temp = Cfs_temp[np.newaxis, :]
        Cfs[k, :] = np.hstack(((-1)**k * np.fliplr(Cfs_temp), Cfs_temp[:, 1:]))
    
    return Cfs
#%%
def sim_ode(x0, tspan, tol_ode, ode_name, params):
    def rhs(x, t):
        if ode_name == 'Linear':
            A = params[0]
            return np.dot(A, x)
        elif ode_name == 'Lorenz':
            sigma, beta, rho = params
            return lorenz(x, sigma, beta, rho)
        elif ode_name == 'Duffing':
            mu, alpha, beta = params
            return duff(x, mu, alpha, beta)
    
    weights = []
    
    if ode_name == 'Linear':
        A = params[0]
        for dim in range(A.shape[0]):
            weights.append(np.concatenate((np.eye(A.shape[0]), A[dim, :].reshape(-1, 1)), axis=1))
    
    elif ode_name == 'Lorenz':
        sigma, beta, rho = params
        weights = [
            np.array([[0, 1, 0, sigma], [1, 0, 0, -sigma]]),
            np.array([[1, 0, 0, rho], [1, 0, 1, -1], [0, 1, 0, -1]]),
            np.array([[1, 1, 0, 1], [0, 0, 1, -beta]])
        ]
    elif ode_name == 'Duffing':
        mu, alpha, beta = params
        weights = [
            np.array([[0, 1, 1], [1, 0, -alpha]]),
            np.array([[0, 1, -mu], [3, 0, -beta]]),
        ]
    
    options = {'rtol': tol_ode, 'atol': tol_ode}
    t = np.array(tspan)
    x = odeint(rhs, x0, t, args=(), **options)
    
    return weights, t, x, rhs


def lorenz(x, sigma, beta, rho):
    dx = np.zeros(3)
    dx[0] = sigma * (x[1] - x[0])
    dx[1] = x[0] * (rho - x[2]) - x[1]
    dx[2] = x[0] * x[1] - beta * x[2]
    
    return dx

def duff(x, mu, alpha, beta):
    dx = np.zeros(2)
    dx[0] = x[1]
    dx[1] = -mu*x[1] - alpha*x[0] - beta * (x[0]**3)
    
    return dx

#%%
def sparsifyDynamics(Theta,dXdt,lambda1,n1,gamma,M):
    
    # n1 = 1
    Theta_G = Theta
    # dXdt = b[:,k]    
    _, nn = Theta_G.shape
    
    if gamma != 0:
        identity = gamma * np.eye(nn)
        Theta = np.vstack((Theta, identity))
        zeros = np.zeros((nn, n1))
        dXdt = np.vstack((dXdt, zeros))
    
    Xi = np.linalg.lstsq(Theta_G, dXdt, rcond=-1)[0]       # initial guess: Least-squares

    if M.size != 0:
        Xi = M * Xi
        bnds = np.linalg.norm(dXdt) / np.linalg.norm(Theta_G, axis=0) * M
        LBs = lambda1 * np.maximum(1, bnds)
        UBs = 1 / lambda1 * np.minimum(1, bnds)
        thrs_EL = np.stack((LBs, bnds, UBs)).T
    else:
        thrs_EL = np.array([])
    
    smallinds = 0*Xi;    

    for j in range(nn):
        smallinds_new = np.logical_or(np.abs(Xi) < LBs, np.abs(Xi) > UBs)
        if np.all(smallinds_new == smallinds):
            its = j
            break
        else:
            smallinds = smallinds_new
            Xi[smallinds] = 0
            for ind in range(n1):
                Xi_lsq = M[~smallinds] * np.linalg.lstsq(Theta_G[:, ~smallinds], dXdt, rcond=-1)[0]     # modified dXdt[:, ind]
                Xi[~smallinds] = Xi_lsq
    
    return Xi   

#%%
def wsindy_ode_fun(xobs,u,tobs,
                polyorder,    custom_tags,custom_fcns,
                phi_class,max_d,tau,tauhat,K_frac,overlap_frac,relax_AG,
                scale_Theta,useGLS,lambdas,gamma,alpha_loss,
                                                            smoothing_window):
    
    ### get true weight vector
    n = xobs.shape[1]      
    m = len(tobs)

    if custom_tags.size > 0 and np.any(custom_tags):                           # Added by CL
        tags = custom_tags
        J = tags.shape[0]
    else:
        # tags = get_tags(polys, n)
        tags = get_tags_SINDy(polyorder)                                                      # Manually assigned
        # tags = np.unique(np.vstack((tags, custom_tags)), axis=0)
        J = tags.shape[0]

    # true_nz_weights = get_true_weights(weights, tags, n)

    ### get scales
    if scale_Theta == 0:
        # scale_x = []                                                         Originally!
        scale_x = np.ones(n+1)                                                 # For Control?
    elif scale_Theta < 0:
        scale_x = np.sqrt(np.mean(np.square(xobs))) * (-scale_Theta)
    else:
        if len(tags.real) > 0:
            max_real_tags = np.max(tags.real)
            scale_x = np.power(np.linalg.norm(np.power(xobs, max_real_tags), axis=0), 1.0 / max_real_tags)
        else:
            scale_x = []
        
    ### set number of test functions. *** decreases when overlapfrac<1

    import math

    if K_frac > 1:
        K = math.ceil(K_frac)
    elif K_frac > 0:
        K = math.floor(m * K_frac)
    elif K_frac < 0:
        K = J * math.ceil(-K_frac)

    ### set mts, pts
    
    if np.any(tau > 1):
        if len(tau) == n:
            mts = tau
        else:
            mts = tau * np.ones(n)
        
        if len(tauhat) == n:
            pts = tauhat
        else:
            pts = tauhat * np.ones(n)
    else:
        if tauhat > 0:
            pts, mts = test_fcn_param_whm(tauhat, tobs, tau)
            pts = np.tile(pts, (n, 1))
            mts = np.tile(mts, (n, 1))
        elif tauhat < 0:
            mts, pts = findcorners(xobs, tobs, tau, -tauhat, phi_class)
        else:
            print('Error: tauhat cannot be 0')
    # mts = np.mean(mts) * np.ones(len(mts))                                      # Modified by CL because of tf.Variable
    # pts = np.mean(pts) * np.ones(len(pts))                                      # Modified by CL because of tf.Variable
    ############## smooth data  ##########################################
    from scipy.signal import convolve

    if smoothing_window > 0:
        filter_weights = []
        for j in range(n):
            filter_weights.append(get_optimal_SMAF(tobs, xobs[:, j], None, smoothing_window, None, None, None, None, None))
            w = (len(filter_weights[j]) - 1) / 2
            xobsj_sym = np.concatenate((np.flipud(xobs[1:int(w)+1, j]), xobs[:, j], np.flipud(xobs[-int(w)-1:-1, j])))
            xobsj_sym = np.reshape(xobsj_sym, (-1, 1))
            filter_weights_j = np.reshape(filter_weights[j], (-1, 1))

            xobs[:, j] = convolve(xobsj_sym, filter_weights_j, mode='valid').flatten()
    else:
        filter_weights = []
        
    ############## missing      ##########################################

    ### set integration line element

    dt = np.mean(np.diff(tobs))
    dv = np.full_like(mts, dt) 

    ### get theta matrix
    # xaug = np.column_stack((xobs,u.reshape((-1, 1))))                           # For control
    # Theta = build_theta(xaug,tags,custom_fcns,scale_x)
    ###########################################################################
    Theta = LibGPU_WSINDy(xobs, u, polyorder)                                                      # For control
    ###########################################################################

    ### get column scaling 

    if scale_Theta == 0:
        M_diag = np.ones((Theta.shape[1]))
    else:
        M_diag = 1. / np.prod(scale_x**np.real(tags), axis=1)
        cfun_offset = Theta.shape[1] - len(M_diag)
        if cfun_offset:
            M_diag = np.concatenate([M_diag, np.ones((cfun_offset, 1))])

    w_sparse = np.zeros((Theta.shape[1], n))
    # grids = [None] * n
    # Gs = [None] * n
    # RTs = Gs
    # bs = Gs
    # vs = [None] * n
    # bweaks = [None] * n
    loss_wsindy = [None] * n
    
    V = [None] * n
    Vp = [None] * n
    Grid_i = [None] * n
    #%%
    for nn in range(n):

    ### get test function weights
        mt = mts[nn]
        pt = pts[nn]
        diffthresh = max(np.ceil((2 * mt + 1) * (1 - overlap_frac)), 1)
        Cfs = phi_int_weights(mt,max_d,-pt)
        v = Cfs[-2, :] * ((mt * dt) ** (-max_d + 1)) * dv[nn]
        vp = Cfs[-1, :] * ((mt * dt) ** (-max_d)) * dv[nn]

    ########## get_tf_centers to obtain grid_i, just relax_AG==0 ##################
        grid_i = np.arange(0, m - 2 * mt, max(diffthresh, np.ceil((m - 2 * mt) / K))).astype(int)
        
    ### get linear system
        b = np.convolve(xobs[:, nn], vp.ravel(), mode='valid')
        # b = b[grid_i-1]
        G = convolve2d(Theta, np.outer(v, [1]), mode='valid')
        # G = G[grid_i-1]

    ########## RT is not calculated, just useGLS = 0 ##################

        M_scale_b = np.concatenate((np.array([1]), M_diag))

        Theta_pdx = np.concatenate((np.reshape(b, (-1, 1)), G), axis=1)
        w_sparse0,lossvals = wsindy_pde_RGLS_seq(lambdas,gamma,Theta_pdx,1,M_scale_b,alpha_loss)
        w_sparse[:,nn] = w_sparse0.ravel()
        
    #### save outputs  
        # vs[nn] = [v,vp]
        V[nn] = v
        Vp[nn] = vp
        loss_wsindy[nn] = lossvals
        Grid_i[nn] = grid_i
        
    return w_sparse, V, Vp, mts,pts, Grid_i, loss_wsindy, tags

#%%
def wsindy_pde_RGLS_seq(lambdas,gamma,Theta_pdx,lhs_ind,M_scale_b,alpha_loss):

    
    num_eq = lhs_ind
    K, m = Theta_pdx.shape

    G = Theta_pdx[:, np.logical_not(np.isin(np.arange(1, m+1), lhs_ind))]
    b = np.zeros((K, num_eq))

    for k in range(num_eq):
        b[:, k] = Theta_pdx[:, 0]                          ########### 0 by CL
    
    W_ls = np.linalg.lstsq(G, b, rcond=-1)[0]
    GW_ls = np.linalg.norm(np.dot(G, W_ls))

    proj_cost = []
    overfit_cost = []
    lossvals = []

    W = np.zeros((m-num_eq, num_eq))

    for l in range(len(lambdas)):
        lambda1 = lambdas[l]                          
        M = []

        for k in range(num_eq):
            M        = M_scale_b[~np.isin(range(m), 0)] / M_scale_b[k]
            W[:, k]  = sparsifyDynamics(G, b[:, k], lambda1, 1, gamma, M)
            W[:, k]  = W[:, k] / M

        proj_cost.append(2 * alpha_loss * np.linalg.norm(G.dot(W - W_ls)) / GW_ls)
        overfit_cost.append(2 * (1 - alpha_loss) * len(np.nonzero(W)[0]) / len(W))
        lossvals.append(proj_cost[-1] + overfit_cost[-1])

    l = lossvals.index(min(lossvals))

    lambda1 = lambdas[l]

    M = []
    for k in range(num_eq):
        M        = M_scale_b[~np.isin(range(m), 0)] / M_scale_b[k]
        W[:, k]  = sparsifyDynamics(G, b[:, k], lambda1, 1, gamma, M)

    lossvals1 = np.vstack((lossvals, lambdas,np.hstack((np.vstack((lossvals[:l+1], lambdas[:l+1])), np.zeros((2, len(lambdas)-(l+1))))),proj_cost, overfit_cost))
    
    return W,lossvals1
