# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:15:27 2021

@author: celin
"""

import numpy as np;
import matplotlib.pyplot as plt

# Constants
simulationTime = 200;   # ms
deltaT = 0.01;          # ms

g_Na = 120;             # mS/cm^2
E_Na = 55;              # mV
g_K = 20;               # mS/cm^2
E_K = -72;              # mV
g_leak = 0.3            # mS/cm^2
E_leak = -17;           # mV
g_A = 47.7;             # mS/cm^2
E_A = -75;              # mV
Cm = 1;                 # uF/cm^2
Vrest = -65;            # mV

# Time grid
t = np.arange(0, simulationTime + deltaT, deltaT) 

# Initial conditions
V0 = -67.976;             # mV
n0 = 0.1558;              # admensional
m0 = 0.01;                # admensional
h0 = 0.965;               # admensional
a0 = 0.5404;              # admensional
b0 = 0.2885;              # admensional

# Arrays Initializations
V = np.zeros(len(t))
n = np.zeros(len(t))
m = np.zeros(len(t))
h = np.zeros(len(t))
a = np.zeros(len(t))
b = np.zeros(len(t))

V[0] = V0
n[0] = n0
m[0] = m0
h[0] = h0
a[0] = a0
b[0] = b0

## J applied
J = np.zeros(len(t))

# Equations
def e(x):
    return np.exp(x)

## n
def alpha_n_eq(Vm):
    return .01*(Vm + 45.7)/(1 - e(-.1*(Vm + 45.7)))

def beta_n_eq(Vm):
    return .125*e(-.0125*(Vm + 55.7))

def n_f_eq(Vm):
    return alpha_n_eq(Vm)/(alpha_n_eq(Vm) + beta_n_eq(Vm))

def tau_n_eq(Vm):
    return 2/(3.8*(alpha_n_eq(Vm) + beta_n_eq(Vm)))

## m
def alpha_m_eq(Vm):
    return .1*(Vm + 29.7)/(1 - e(-.1*(Vm + 29.7)))

def beta_m_eq(Vm):
    return 4*e(-.0556*(Vm + 54.7))

def m_f_eq(Vm):
    return alpha_m_eq(Vm)/(alpha_m_eq(Vm) + beta_m_eq(Vm))

def tau_m_eq(Vm):
    return 1/(3.8*(alpha_m_eq(Vm) + beta_m_eq(Vm)))

## h
def alpha_h_eq(Vm):
    return .07*e(-.05*(Vm + 48))

def beta_h_eq(Vm):
    return 1/(1 + e(-.1*(Vm + 18)))

def h_f_eq(Vm):
    return alpha_h_eq(Vm)/(alpha_h_eq(Vm) + beta_h_eq(Vm))

def tau_h_eq(Vm):
    return 1/(3.8*(alpha_h_eq(Vm) + beta_h_eq(Vm)))

## a
def a_f_eq(Vm):
    return ((.0761*e(.0314*(Vm + 94.22)))/(1 + e(.0346*(Vm + 1.17))))**(1/3)

def tau_a_eq(Vm):
    return .3632 + (1.158/(1 + e(.0497*(Vm + 55.96))))

## b
def b_f_eq(Vm):
    return (1/(1 + e(.0688*(Vm + 53.3))))**(4)

def tau_b_eq(Vm):
    return 1.24 + (2.678/(1 + e(.0624*(Vm + 50))))

## Currents
def I_Na_eq(m,h,V):
    return g_Na*(m**3)*h*(V - E_Na)

def I_K_eq(n,V):
    return g_K*(n**4)*(V - E_K)

def I_leak_eq(V):
    return g_leak*(V - E_leak)

def I_A_eq(a,b,V):
    return g_A*(a**3)*b*(V - E_A)

# J
J1 = np.zeros(len(t))
ti = 6000;
duration = 14000;
tf = ti + duration;
currentValue = 20;           # uA/cm^2
J1[ti:tf] = currentValue;

# FJ curve
J_0 = 8                 # Initial current value
J_f = 10                # Final current value
dJ = .2                 # current step
J_array = np.arange(J_0, J_f, dJ)
f_array_1 = np.zeros(len(J_array))
f_array_2 = np.zeros(len(J_array))

t_L = 0
f1 = 0
f2 = 0
currentValue = J_0
# Method: Runge-Kutta 4th order
for k in range (len(J_array)):
    J1[ti:tf] = J_array[k]
    for i in range(len(t) - 1):
        n1 = deltaT*((n_f_eq(V[i]) - n[i])/tau_n_eq(V[i]))
        n2 = deltaT*((n_f_eq(V[i] + .5*n1) - n[i])/tau_n_eq(V[i] + .5*n1))
        n3 = deltaT*((n_f_eq(V[i] + .5*n2) - n[i])/tau_n_eq(V[i] + .5*n2))
        n4 = deltaT*((n_f_eq(V[i] + deltaT*n3) - n[i])/tau_n_eq(V[i] + deltaT*n3))
        n[i + 1] = n[i] + (n1 + 2*(n2 + n3) + n4)/6 
        
        m1 = deltaT*((m_f_eq(V[i]) - m[i])/tau_m_eq(V[i]))
        m2 = deltaT*((m_f_eq(V[i] + .5*m1) - m[i])/tau_m_eq(V[i] + .5*m1))
        m3 = deltaT*((m_f_eq(V[i] + .5*m2) - m[i])/tau_m_eq(V[i] + .5*m2))
        m4 = deltaT*((m_f_eq(V[i] + deltaT*m3) - m[i])/tau_m_eq(V[i] + deltaT*m3))
        m[i + 1] = m[i] + (m1 + 2*(m2 + m3) + m4)/6 
        
        h1 = deltaT*((h_f_eq(V[i]) - h[i])/tau_h_eq(V[i]))
        h2 = deltaT*((h_f_eq(V[i] + .5*h1) - h[i])/tau_h_eq(V[i] + .5*h1))
        h3 = deltaT*((h_f_eq(V[i] + .5*h2) - h[i])/tau_h_eq(V[i] + .5*h2))
        h4 = deltaT*((h_f_eq(V[i] + deltaT*h3) - h[i])/tau_h_eq(V[i] + deltaT*h3))
        h[i + 1] = h[i] + (h1 + 2*(h2 + h3) + h4)/6 
        
        a1 = deltaT*((a_f_eq(V[i]) - a[i])/tau_a_eq(V[i]))
        a2 = deltaT*((a_f_eq(V[i] + .5*a1) - a[i])/tau_a_eq(V[i] + .5*a1))
        a3 = deltaT*((a_f_eq(V[i] + .5*a2) - a[i])/tau_a_eq(V[i] + .5*a2))
        a4 = deltaT*((a_f_eq(V[i] + deltaT*a3) - a[i])/tau_a_eq(V[i] + deltaT*a3))
        a[i + 1] = a[i] + (a1 + 2*(a2 + a3) + a4)/6 
        
        b1 = deltaT*((b_f_eq(V[i]) - b[i])/tau_b_eq(V[i]))*0.25
        b2 = deltaT*((b_f_eq(V[i] + .5*b1) - b[i])/tau_b_eq(V[i] + .5*b1))*0.25
        b3 = deltaT*((b_f_eq(V[i] + .5*b2) - b[i])/tau_b_eq(V[i] + .5*b2))*0.25
        b4 = deltaT*((b_f_eq(V[i] + deltaT*b3) - b[i])/tau_b_eq(V[i] + deltaT*b3))*0.25
        
        b[i + 1] = b[i] + (b1 + 2*(b2 + b3) + b4)/6 
        
        I_Na = I_Na_eq(m[i], h[i], V[i])
        I_K = I_K_eq(n[i], V[i]) 
        I_leak = I_leak_eq(V[i])  
        I_A = I_A_eq(a[i], b[i], V[i])
        
        J[i] = J1[i]
        Iion = J[i] - I_K - I_Na - I_leak - I_A;
        
        V1 = deltaT*(Iion/Cm)
        V2 = deltaT*((Iion + .5*V1)/Cm)
        V3 = deltaT*((Iion + .5*V2)/Cm)
        V4 = deltaT*((Iion + V3)/Cm)
        V[i + 1] = V[i] + (V1 + 2*(V2 + V3) + V4)/6 
    
    # Action Potential Counter
    AP_Counter = 0
    position = 0
    for j in range (0, len(t) - 1):
        if (V[j] > 0 and V[j + 1] < 0):
            AP_Counter += 1
        
        # First AP Latency 
        if (AP_Counter == 1):
            t_L = t[j]
            # Frequency
            f1 = (1/(t_L))*1000
            position = j
            
        f2 = (AP_Counter/(duration*deltaT))*1000
    
    f_array_1[k] = f1
    f_array_2[k] = f2

# Plots

fig2, axs2 = plt.subplots(4, figsize=(14,20))
fig2.suptitle("Questão 1 - letra d - latência do primeiro disparo")
axs2[0].plot(np.arange(J_0, J_f, dJ),J_array)
axs2[0].set_title('J x t')
axs2[0].set(ylabel='J(uA/cm^2)')
axs2[0].set(xlabel='t (ms)')
axs2[0].legend(shadow=True, fancybox=True)
axs2[0].grid()

axs2[1].plot(t,V)
axs2[1].set_title('V')
axs2[1].set(ylabel='V (mV)')
axs2[1].set(xlabel='t (ms)')
axs2[1].legend(shadow=True, fancybox=True)
axs2[1].grid()

axs2[2].set_title('f-J - método letra c')
axs2[2].plot(J_array,f_array_2)
axs2[2].set(ylabel='f (Hz)')
axs2[2].set(xlabel='J (uA/cm^2)')
axs2[2].grid()
axs2[2].set_xticks(np.arange(J_0, J_f + dJ, step=dJ))

axs2[3].set_title('f-J - método letra d')
axs2[3].plot(J_array,f_array_1)
axs2[3].set(ylabel='f (Hz)')
axs2[3].set(xlabel='J (uA/cm^2)')
axs2[3].grid()
axs2[3].set_xticks(np.arange(J_0, J_f + dJ, step=dJ))
