#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 09:48:56 2022

@author: anton
"""

import numpy as np
np.set_printoptions(precision=9)
np.seterr(divide='ignore', invalid='ignore', over='ignore')                #dont print warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import mathieu_functions_OG as mf
import timeit
from datetime import timedelta
import multiprocessing as mp

#Parameter
element_type = 'circle'     # can be either 'line' or 'circle'
r = 1
alpha_l = 2
alpha_t = 0.05
beta = 1/(2*alpha_l)
C0 = 10
Ca = 8
gamma = 3.5

d = np.sqrt((r*np.sqrt(alpha_l/alpha_t))**2-r**2)           #semi-focal distance: c**2 = a**2 - b**2 --> c = +/-SQRT(a**2 - b**2)
q = (d**2*beta**2)/4
# print(d,q)
n = 7             #Number of terms in mathieu series -1
M = 100          #Number of Control Points, 5x overspecification

#wrapper xy to eta psi
def uv(x, y):
    """
    Convert Cartesian (x, y) to elliptic coordinates (eta, psi),
    with foci at (±d, 0) and optional anisotropy.

    Parameters:
        x, y     : Cartesian coordinates
        d        : half-distance between foci

    Returns:
        eta, psi : elliptic coordinates
    """
    Y = np.sqrt(alpha_l / alpha_t) * y
    B = x**2 + Y**2 - d**2
    discriminant = B**2 + 4 * d**2 * x**2

    # Safety against numerical issues
    discriminant = max(discriminant, 0.0)
    sqrt_disc = np.sqrt(discriminant)

    p = (-B + sqrt_disc) / (2 * d**2)
    q = (-B - sqrt_disc) / (2 * d**2)

    # Clip p to [0, 1] for arcsin safety
    p_clipped = np.clip(p, 0, 1)
    psi_0 = np.arcsin(np.sqrt(p_clipped))

    # Determine correct quadrant
    if Y >= 0 and x >= 0:
        psi = psi_0
    elif Y < 0 and x >= 0:
        psi = np.pi - psi_0
    elif Y <= 0 and x < 0:
        psi = np.pi + psi_0
    else:  # Y > 0 and x < 0
        psi = 2 * np.pi - psi_0

    # Compute eta
    inner = 1 - 2*q + 2 * np.sqrt(max(q**2 - q, 0.0))
    eta = 0.5 * np.log(inner)

    return eta, psi

#polar coordinates
phi = np.linspace(0, 2*np.pi, M)
x1 = r*np.cos(phi)
y1 = r*np.sin(phi)

#elliptic coordinates
uv_vec = np.vectorize(uv)
psi1 = uv_vec(x1, y1)[1]
if element_type == 'circle':
    eta1 = uv_vec(x1, y1)[0]
if element_type == 'line':
    eta1 = np.zeros(M)

#Mathieu Functions
m = mf.mathieu(q)

def Se(order, psi):                    #even angular first kind
    return m.ce(order, psi).real
def So(order, psi):                    #odd angular first kind
    return m.se(order, psi).real
def Ye(order, eta):                    #even radial second Kind
    return m.Ke(order, eta).real
def Yo(order, eta):                    #odd radial second Kind
    return m.Ko(order, eta).real

#Target Function
def F1(x1):
    return (C0*gamma+Ca)*np.exp(-beta*x1)

#System of Equations to calculate coefficients
lst = []                                                                        #empty array

for i in range(0, M):                                                            #filling array with all terms of MF for the source
    for j in range(0, 1):
        lst.append(Se(j, psi1[i])*Ye(j, eta1[i]))
    for j in range(1, n):
        lst.append(So(j, psi1[i])*Yo(j, eta1[i]))
        lst.append(Se(j, psi1[i])*Ye(j, eta1[i]))

F_M = []
s = 2*n-1
for k in range(0, len(lst), s):                                                   #appending each line (s elements) as arrays (in brackets) -> achieve right array structure (list of arrays)
    F_M.append(lst[k:k+s])

F = []

for u in range(0, M):                                                            #target function vector
    F.append(F1(x1[u])) #

Coeff = np.linalg.lstsq(F_M, F, rcond=None)
# print(Coeff[0])

#comprehensive solution
def c(x, y):
    if element_type == 'circle':
        if (x**2+y**2)<=r**2:
            return C0
    if element_type == 'line':
        if x == 0 and -r < y < r:
            return C0

    eta = uv(x, y)[0]
    psi = uv(x, y)[1]

    F = Coeff[0][0]*Se(0, psi)*Ye(0, eta)
    for w in range(1, n):
        F += Coeff[0][2*w-1]*So(w, psi)*Yo(w, eta) \
            + Coeff[0][2*w]*Se(w, psi)*Ye(w, eta)

    if ((F*np.exp(beta*x)))> Ca:
        return ((((F*np.exp(beta*x)))-Ca)/gamma).round(9)
    else:
        return ((F*np.exp(beta*x))-Ca).round(9)
#%%

inc = 0.1
# Define a helper function for `Pool.map`
def compute_conc(point):
    x, y = point
    return c(x, y)

def Conc_array(x_min, x_max, y_min, y_max, inc):
    xaxis = np.arange(x_min, x_max, inc)
    yaxis = np.arange(y_min, y_max, inc)
    X, Y = np.meshgrid(xaxis, yaxis)

    # Flatten the grid for parallel processing
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Prepare the inputs as tuples of (x, y) for the function `c`
    points = list(zip(X_flat, Y_flat))

    # Use multiprocessing.Pool to parallelize the computation of `c(x, y)` over all grid points
    with mp.Pool(mp.cpu_count()) as pool:
        Conc_flat = pool.map(compute_conc, points)

    # Reshape the flat result back to the original grid shape
    Conc = np.array(Conc_flat).reshape(X.shape)

    return xaxis, yaxis, Conc


# Run the function
def run():
    start = timeit.default_timer()

    result = Conc_array(0, 400+inc, -5, 5+inc, inc)

    stop = timeit.default_timer()
    sec = int(stop - start)
    cpu_time = timedelta(seconds = sec)
    print('Computation time [hh:mm:ss]:', cpu_time)

    plt.figure(figsize=(16, 9), dpi = 300, layout="constrained")
    mpl.rcParams.update({'font.size': 22})
    # plt.axis('scaled')
    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')
    plt.xticks(range(len(result[0]))[::int(50/inc)], result[0][::int(50/inc)].round(0))
    plt.yticks(range(len(result[1]))[::int(10/inc)], result[1][::int(10/inc)].round(0))
    Plume_cd = plt.contourf(result[2], levels=np.linspace(0, C0+0.01, 11), cmap='Reds') #np.linspace(Ca, 43, 10)
    Plume_ca = plt.contourf(result[2], levels=np.linspace(-Ca, 0, 9), cmap='Blues_r')
    Plume_max = plt.contour(result[2], levels=[0], linewidths=2, colors='k')

    #Colorbar
    cbar_cd = plt.colorbar(Plume_cd, ticks=Plume_cd.levels, label='Electron donor concentration [mg/l]', location='bottom', aspect=75)
    cbar_ca = plt.colorbar(Plume_ca, ticks=Plume_ca.levels, label='Electron acceptor concentration [mg/l]', location='bottom', aspect=75)
    cbar_ca.set_ticks(Plume_ca.levels)  # Ensure it uses the same tick positions
    cbar_ca.set_ticklabels([f"{abs(level):.0f}" for level in Plume_ca.levels])


    # Get one of the original colorbar positions to reuse width/height
    bar_height = 0.01
    bar_width = 0.8
    bar_x = 0.1

    # Set tighter vertical positions
    cbar_ca.ax.set_position([bar_x, 0, bar_width, bar_height])  # Acceptor (top one)
    cbar_cd.ax.set_position([bar_x, 1, bar_width, bar_height])  # Donor (bottom one)

    Lmax = Plume_max.get_paths()[0]
    print('Lmax =',int(np.max(Lmax.vertices[:,int((result[1][0]+result[1][-1])/2)])*inc-np.abs(result[0][0])))
    plt.show()
#%%

#absolut error [mg/l]
    phi2 = np.linspace(0, 2*np.pi, 360)
    if element_type == 'circle':
        x_test = (r + 1e-9) * np.cos(phi2)
        y_test = (r + 1e-9) * np.sin(phi2)
    if element_type == 'line':
        x_test = np.linspace(0, 0, 360)
        y_test = (r + 1e-9) * np.sin(phi2) #np.linspace(-r, r, 360)


    Err = []
    for i in range(0, 360, 1):
        Err.append((c(x_test[i], y_test[i])))
    #print(Err)
    print('Min =', np.min(Err).round(9), 'mg/l')
    print('Max =', np.max(Err).round(9), 'mg/l')
    print('Mean =', np.mean(Err).round(9), 'mg/l')
    print('Standard Deviation =', np.std(Err).round(15), 'mg/l')

    plt.figure(figsize=(16,9), dpi=300)
    mpl.rcParams.update({'font.size': 22})
    plt.plot(phi2, Err, color='k')
    plt.xlabel('Angle (°)')
    plt.ylabel('Concentration (mg/l)')
    plt.ticklabel_format(axis='both', style='scientific', useMathText=True, useOffset=True, scilimits=(0,2))
    plt.xticks(np.linspace(0, 2*np.pi, 7), np.linspace(0, 360, 7))
    plt.xlim([0, 2*np.pi])
    return result

if __name__ ==  '__main__':
    run()