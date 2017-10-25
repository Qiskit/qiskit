"""

Wigner Function Visualization Module

Description:
    This module contains functions for calculating and plotting the Wigner
    function. The ways the Wigner function can be calculated includes:
    - The equal angle slice bloch sphere plot of an arbitrary state
      described by a density matrix or state vector.
    - An array of points in phase space to be plotted as a curve
    - A single point in phase space
    - An NxN array plotted as a plaquette

References:
    [1] T. Tilma, M. J. Everitt, J. H. Samson, W. J. Munro,
and K. Nemoto,
        Phys. Rev. Lett. 117, 180401 (2016).
    [2] R. P. Rundle, P. W. Mills, T. Tilma, J. H. Samson, and
M. J. Everitt,
        Phys. Rev. A 96, 022117 (2017).
"""

import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def wigner_function(rho, res=100):

    """
    Plot the equal angle slice spin Wigner function of an arbitrary
    quantum state

    Args:
        rho (np.matrix[[complex]]): - Matrix of 2**n x 2**n complex
                                        numbers
                                    - State Vector of 2**n x 1 complex
                                        numbers
        res (int) : number of theta and phi values in meshgrid
                    on sphere (creates a res x res grid of points)
    Returns:
        none: plot is shown with matplotlib on the screen and expression for
            Wigner function printed
        
    """

    if np.amin(rho.shape) == 1:
        rho = np.outer(rho,rho) # turns state vector to a density matrix
    rho = np.matrix(rho)
    num = int(np.log2(len(rho))) # number of qubits
    phi_vals = np.linspace(0,math.pi,num=res,
                             dtype = np.complex_)
    theta_vals = np.linspace(0,0.5*math.pi,num=res,
                             dtype = np.complex_) # phi and theta values for WF
    W = np.empty([res,res])
    harr = np.sqrt(3)
    Delta_su2 = np.zeros((2,2),dtype = np.complex_)

    #create the spin Wigner function
    
    for theta in range(res):
        costheta  = harr*np.cos(2*theta_vals[theta])
        sintheta  = harr*np.sin(2*theta_vals[theta])
        
        for phi in range(res):                  
            Delta_su2[0,0] =  0.5*(1+costheta)
            Delta_su2[0,1] = -0.5*(np.exp(2j*phi_vals[phi])*sintheta)
            Delta_su2[1,0] = -0.5*(np.exp(-2j*phi_vals[phi])*sintheta)
            Delta_su2[1,1] =  0.5*(1-costheta)            
            kernel = 1
            for i in range(num):
                kernel = np.kron(kernel,Delta_su2) # creates phase point kernel              
            
            W[phi,theta] = np.real(np.trace(rho*kernel)) # The Wigner function
                
    # Plot a sphere (x,y,z) with Wigner function facecolor data stored in Wc
    fig = plt.figure(figsize=(11,9))
    ax = fig.gca(projection = '3d')
    Wmax = np.amax(W)
    #color data for plotting
    Wc = cm.seismic_r((W+Wmax)/(2*Wmax)) # color data for sphere
    Wc2 = cm.seismic_r((W[0:res, int(res/2):res]+Wmax)/(2*Wmax)) # bottom
    Wc3 = cm.seismic_r((W[int(res/4):int(3*res/4), 0:res]+Wmax)/(2*Wmax)) #side
    Wc4 = cm.seismic_r((W[int(res/2):res, 0:res]+Wmax)/(2*Wmax)) #  back
    
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v)) # creates a sphere mesh

    ax.plot_surface(x,y,z, facecolors=Wc, 
                    vmin=-Wmax, vmax=Wmax,
                    rcount=res, ccount=res,
                    linewidth=0, zorder=0.5,
                    antialiased=False) # plots Wigner Bloch sphere
    
    ax.plot_surface(x[0:res, int(res/2):res],
                    y[0:res, int(res/2):res],
                    -1.5*np.ones((res,int(res/2))),
                    facecolors=Wc2,
                    vmin=-Wmax, vmax=Wmax,
                    rcount=res/2, ccount=res/2,
                    linewidth=0, zorder=0.5,
                    antialiased=False) # plots bottom reflection

    ax.plot_surface(-1.5*np.ones((int(res/2), res)),
                    y[int(res/4):int(3*res/4), 0:res],
                    z[int(res/4):int(3*res/4), 0:res],
                    facecolors=Wc3,
                    vmin=-Wmax, vmax=Wmax,
                    rcount=res/2, ccount=res/2,
                    linewidth=0, zorder=0.5,
                    antialiased=False) # plots side reflection

    ax.plot_surface(x[int(res/2):res, 0:res],
                    1.5*np.ones((int(res/2), res)),
                    z[int(res/2):res, 0:res],
                    facecolors=Wc4,
                    vmin=-Wmax, vmax=Wmax,
                    rcount=res/2, ccount=res/2,
                    linewidth=0, zorder=0.5,
                    antialiased=False) # plots back reflection
    
    ax.w_xaxis.set_pane_color((0.4, 0.4, 0.4, 1.0))
    ax.w_yaxis.set_pane_color((0.4, 0.4, 0.4, 1.0))
    ax.w_zaxis.set_pane_color((0.4, 0.4, 0.4, 1.0))
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_zticks([], [])
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    m = cm.ScalarMappable(cmap=cm.seismic_r)
    m.set_array([-Wmax, Wmax])
    plt.colorbar(m, shrink=0.5, aspect=10)
    
    plt.show()


def plot_wigner_curve(wigner_results, xaxis=None):
    """
    Plots a curve for points in phase space of the spin Wigner function
    
    Args:
        wigner_results(np.array): an array of points to plot as a 2d curve
        xaxis (np.array):  the range of the x axis

    Returns:
        none: plot is shown with matplotlib to the screen
    
    """
    if xaxis is None:
        xaxis = np.linspace(0,len(wigner_results)-1,num=len(wigner_results))
    
    plt.plot(xaxis,wigner_results)
    plt.show()


def plot_wigner_plaquette(wigner_results, maxWigner='local'):
    """
    Plots plaquette of wigner function data, the plaquette will consist
    of cicles each colored to match the value of the Wigner function at
    the given point in phase space.

    Args:
        wigner_results (matrix): array of Wigner function data where the
                                 rows are plotted along the x axis and the
                                 columns are plotted along the y axis
        maxWigner: - 'local' puts the maximum value to maximum of the points
                   - 'unit' sets maximum to 1
                   - float for a custom maximum.

    Returns:
        none: plot is shown with matplotlib to the screen

    """
    wigner_results = np.matrix(wigner_results)
    dim = wigner_results.shape
    
    if maxWigner == 'local':
        Wmax = np.amax(wigner_results)
    elif maxWigner == 'unit':
        Wmax = 1
    else:
        Wmax = maxWigner #For a float input
        
    cmap = matplotlib.cm.get_cmap('seismic_r')
    
    xax = dim[1]-0.5
    yax = dim[0]-0.5
    norm = np.amax(dim)
    
    fig = plt.figure(figsize=((xax+0.5)*6/norm,(yax+0.5)*6/norm))
    ax = fig.gca()

    for x in range(int(dim[1])):
        for y in range(int(dim[0])):
            
            circle = plt.Circle((x,y),0.49,
                                color =
                                cmap((wigner_results[y,x]+Wmax)/(2*Wmax)))
            ax.add_artist(circle)

    ax.set_xlim(-1, xax+0.5)
    ax.set_ylim(-1, yax+0.5)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    m = cm.ScalarMappable(cmap=cm.seismic_r)
    m.set_array([-Wmax, Wmax])
    plt.colorbar(m, shrink=0.5, aspect=10)
    plt.show()


def plot_wigner_function(wigner_results, phis=None,
                         thetas=None, method=None,
                         text_out=None):
    """
    Plots Wigner results in appropriate format

    Args:
        wigner_results:
        phis: Values of phi
        thetas: Values of theta
        method: how the data is to be plotted,
            methods are:
                point: a single point in phase space
                curve: a two dimensional curve
                plaquette: points plotted as circles

    Returns:
        none: plot is shown with matplotlib to the screen
    """
    
    if method is None:
        wigDim = len(np.shape(wigner_results))
        if wigDim == 1:
            if np.shape(wigner_results) == 1:
                method = 'point'
            else:
                method = 'curve'
        elif wigDim ==2:
            method ='plaquette'
            
    if method == 'curve':
        plot_wigner_curve(wigner_results, xaxis=phis)
    elif method == 'plaquette':
        plot_wigner_plaquette(wigner_results)
    elif method == 'state':
        wigner_function(wigner_results, text_out)
    elif method == 'point':
        plot_wigner_plaquette(wigner_results)
        print('point in phase space is '+str(wigner_results))
    else:
        print("No method given")
        

