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


#Parameter
alpha_l = 10
alpha_t = 1
beta = 1/(2*alpha_l)
C0 = 10
Ca = 8
gamma = 3.5
r = 1
d = np.sqrt((r*np.sqrt(alpha_l/alpha_t))**2-r**2)           #focal distance: c**2 = a**2 - b**2 --> c = +/-SQRT(a**2 - b**2)
q = (d**2*beta**2)/4
# 2q = (d**2*beta**2)/2 --> q = (see above)

n = 7             #Number of terms in mathieu series -1
M = 100           #Number of Control Points

#wrapper xy to eta psi
def uv(x, y):
    Y = np.sqrt(alpha_l/alpha_t)*y
    B = x**2+Y**2-d**2
    p = (-B+np.sqrt(B**2+4*d**2*x**2))/(2*d**2)
    q = (-B-np.sqrt(B**2+4*d**2*x**2))/(2*d**2)

    psi_0 = np.arcsin(np.sqrt(p))

    if Y >= 0 and x >= 0:
        psi = psi_0
    if Y < 0 and x >= 0:
        psi = np.pi-psi_0
    if Y <= 0 and x < 0:
        psi = np.pi+psi_0
    if Y > 0 and x < 0:
        psi = 2*np.pi-psi_0

    eta = 0.5*np.log(1-2*q+2*np.sqrt(q**2-q))
    return eta, psi

#polar coordinates
phi = np.linspace(0, 2*np.pi, M)
x1 = r*np.cos(phi)
y1 = r*np.sin(phi)

#elliptic coordinates
uv_vec = np.vectorize(uv)
psi1 = uv_vec(x1, y1)[1]
eta1 = uv_vec(x1, y1)[0]

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

Coeff = np.linalg.lstsq(F_M, F, rcond=None)                                       #calculated coefficients and residual
print(Coeff[0])

#comprehensive solution
def c(x, y):
    if (x**2+y**2)<=r**2:
        return C0
    eta = uv(x, y)[0]
    psi = uv(x, y)[1]

    F = Coeff[0][0]*Se(0, psi)*Ye(0, eta)
    for w in range(1, n):
        F += Coeff[0][2*w-1]*So(w, psi)*Yo(w, eta) \
            + Coeff[0][2*w]*Se(w, psi)*Ye(w, eta)

    # return (F*np.exp(beta*x)).round(9)

    if ((F*np.exp(beta*x)))> Ca:
        return ((((F*np.exp(beta*x)))-Ca)/gamma).round(9)
    else:
        return ((((F*np.exp(beta*x)))-Ca)).round(9)
#%% Fourier Approx.
# theta = np.linspace(0, 2*np.pi, M)
# def Fourier(theta):
#     Fou = 0.5 * Coeff[0][0]
#     for i in range(1, n+1):
#         Fou += Coeff[0][i] * np.cos((i)*theta)
#     return Fou.round(9)

# bc = (C0*gamma+Ca) * np.exp(-r*np.cos(theta)/2/alpha_l) #*gamma+Ca
# #print(Fourier(theta))
# plt.plot(Fourier(theta), label = 'fou')
# plt.plot(bc, label = 'bc')
# plt.legend()
#%%
# concentration array for plotting purpose
inc = 0.1

def Conc_array(x_min, x_max, y_min, y_max, inc):
    xaxis = np.arange(x_min, x_max, inc)
    yaxis = np.arange(y_min, y_max, inc)
    X, Y = np.meshgrid(xaxis, yaxis)
    v = np.vectorize(c)
    Conc = v(X, Y)
    return xaxis, yaxis, Conc

# single-core processing
start = timeit.default_timer()
result = Conc_array(-5, 10.1, -3, 3.1, inc)
stop = timeit.default_timer()
sec = int(stop - start)
cpu_time = timedelta(seconds = sec)
print('Computation time [hh:mm:ss]:', cpu_time)

plt.figure(figsize=(16, 16*(len(result[1])/len(result[0]))), dpi = 300)
mpl.rcParams.update({'font.size': 22})
plt.axis('scaled')
#plt.grid()
#plt.xlim([0,150])
plt.xlabel('$x$ (m)')
plt.ylabel('$y$ (m)')
plt.xticks(range(len(result[0]))[::int(1/inc)], result[0][::int(1/inc)].round())
plt.yticks(range(len(result[1]))[::int(1/inc)], result[1][::int(1/inc)].round())
#Plume = plt.contourf(result[2], levels=np.linspace(Ca, 43, 5), cmap='binary') #np.linspace(-8,35,10) [0], levels=[0], , colors='k'
Plume_max = plt.contour(result[2], levels=np.linspace(0, C0, 5), linewidths=1, colors='k') #colors='k'
Plume_max.clabel(inline=True, colors = 'k')
plt.text(80,5,r'$C_D$ concentration in mg/l')
plt.text(43,28.5,r'$C_D^s = 10$')
Source = plt.Circle((50, 30), 10, edgecolor='red', facecolor='mistyrose', linewidth = 6, fill=True)
plt.gca().add_patch(Source)


#Colorbar
# norm= mpl.colors.Normalize(vmin=Plume.cvalues.min(), vmax=Plume.cvalues.max())
# sm = plt.cm.ScalarMappable(norm=norm, cmap = Plume.cmap)
# sm.set_array([])
# plt.colorbar(Plume, ticks=Plume.levels, label='Concentration (mg/l)', location='bottom', shrink=0.8)

# Label = '$C_{D}=C_{A}=0$'
# Lmax = Plume.get_paths()[0]
# plt.clabel(Plume, fmt=Label, manual = [(50,-(2*np.max(result[1])))])
# print('Lmax =',int(np.max(Lmax.vertices[:,int((result[1][0]+result[1][-1])/2)])*inc-np.abs(result[0][0])))
# textbox = r'$L_{max} = $' + str(int(np.max(Lmax.vertices[:,int((result[1][0]+result[1][-1])/2)])*inc-np.abs(result[0][0]))) + ' m'
# plt.text(20,2*np.max(result[1])-10,textbox)
# #plt.savefig('onesource.pdf')
#%%

#absolut error [mg/l]
phi2 = np.linspace(0, 2*np.pi, 360)
x_test = (r) * np.cos(phi2)
y_test = (r) * np.sin(phi2)

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
# min_text = 'Min = ' + str(np.min(Err).round(9)) + ' mg/l'
# max_text = 'Max = ' + str(np.max(Err).round(9)) + ' mg/l'
# mean_text = 'Mean = ' + str(np.mean(Err).round(9)) + ' mg/l'
# std_text = 'Standard Deviation = '+ str(np.std(Err).round(9)) + ' mg/l'
# plt.text(0.8, 43+1e-8, min_text)
# plt.text(0.8, 43+9e-9, max_text)
# plt.text(0.8, 43+8e-9, mean_text)
# plt.text(0.8, 43+7e-9, std_text)
# rectangle = mpl.patches.Rectangle((0.75, 43+6.5e-9), 2.75, 4.75e-9, color='lightgrey', linewidth=3)
# plt.gca().add_patch(rectangle)
# #plt.savefig('erroralongboundary.pdf')
