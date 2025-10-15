# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:42:20 2025

@author: crist
"""

import matplotlib.pyplot as plt

#%%
def plot_success_rates(mS_no, mS_Ap, WmS_no, WmS_Ap, filename):
    """
    Plot horizontal bar chart of success rates.
    
    Parameters:
    -----------
    mS_no : float
        mSINDy success rate without approximation
    mS_Ap : float
        mSINDy success rate with approximation
    WmS_no : float
        WmSINDy success rate without approximation
    WmS_Ap : float
        WmSINDy success rate with approximation
    filename : str
        Name of the file to save the plot (should end with .pdf)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8,2))
    
    # Labels for the bars
    labels = ['Wm (App)', 'Wm (No App)', 'm (App)', 'm (No App)']
    
    # Adding labels and title
    plt.xlabel('Success rate %', fontsize=20)
    
    # Heights of the bars
    heights = [WmS_Ap, WmS_no, mS_Ap, mS_no]
    
    # Add grid
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    
    # Define colors if not already defined globally
    color_mSINDy_noApp = 'c'
    color_WmSINDy_noApp = [0.58, 0.44, 0.86]
    color_mSINDy_App = 'orange'
    color_WmSINDy_App = 'pink'
    bar_height = 0.5
    
    # Plotting horizontal bars
    plt.grid()
    plt.barh(labels, heights, 
            color=[color_WmSINDy_App,
                  color_WmSINDy_noApp,
                  color_mSINDy_App,
                  color_mSINDy_noApp],
            height=bar_height, 
            edgecolor='black')
    
    plt.xticks([10,20,30,40,50,60,70,80,90,100])
    plt.xlim(0,100)
    
    # Save and close
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.show()

#%% Lorenz 63      m_no  m_ap  Wm_no  Wm_ap
plot_success_rates(30.0, 80.0, 88.0, 58.0, "L63.pdf")  

#%% Rossler
plot_success_rates(0.0, 2.0, 50.0, 20.0, "Rossler.pdf")

#%%  Lorenz 96
plot_success_rates(76.0, 94.0, 86.0, 98.0, "L96.pdf")

#%%  Duffing
plot_success_rates(68.0, 88.0, 82.0, 96.0, "Duffing.pdf")

#%%  Osc. Cubic
plot_success_rates(2.0, 20.0, 8.0, 28.0, "CubicOsc.pdf")

#%%  Van der Pol
plot_success_rates(72.0, 70.0, 88.0, 68.0, "VdP.pdf")

#%%  Lotka
plot_success_rates(48.0, 78.0, 94.0, 92.0, "Lotka.pdf")
