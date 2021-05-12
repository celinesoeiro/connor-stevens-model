# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:13:48 2021

@author: celin
"""

import numpy as np;
import matplotlib.pyplot as plt
import scipy.integrate as scp

# Constants (SI)
mili = 10**(-3)
micro = 10**(-6)
pico = 10**(-9)
nano = 10**(-12)

simulationTime = 750*mili   # s
deltaT = 0.01*mili          # s

A = 0.01*mili           # m^2

E_leak = -0.070         # V
E_Na = 0.055            # V
E_K = -0.090            # V
E_Ca = 0.120            # V

G_leak = 10*nano        # Siemens
G_Na = 3.6*micro        # Siemens
G_K = 1.6*micro         # Siemens
G_CaT = 0.22*micro      # Siemens

Cm = 100*pico           # Farads

# Time grid
t = np.arange(0, simulationTime + deltaT, deltaT) 

## J applied
Jbase_value = 10*micro;           # A/cm^2
Jstep_value = 10*micro;
J = Jbase_value*np.ones(len(t))
step_on = 2500
step_off = 5000
J[step_on: step_off] = Jbase_value + Jstep_value

# Equations
def e(x):
    return np.exp(x)

## m
def alpha_m_eq(Vm):
    if (Vm == -0.035):                         # to prevent division by zero
        return 10**3;
    else:
        return ((10**5)*(Vm + .035))/(1 - e(-100*(Vm + .035)))

def beta_m_eq(Vm):
    return 4000*e((-(Vm + .06))/(.018))

def m_f_eq(Vm):
    return alpha_m_eq(Vm)/(alpha_m_eq(Vm) + beta_m_eq(Vm))

def tau_m_eq(Vm):
    return 1/(alpha_m_eq(Vm) + beta_m_eq(Vm))

## h
def alpha_h_eq(Vm):
    return 350*e(-50*(Vm + .058))

def beta_h_eq(Vm):
    return 5000/(1 + e(-100*(Vm + .028)))

def h_f_eq(Vm):
    return alpha_h_eq(Vm)/(alpha_h_eq(Vm) + beta_h_eq(Vm)) 

def tau_h_eq(Vm):
    return 1/(alpha_h_eq(Vm) + beta_h_eq(Vm))

## n
def alpha_n_eq(Vm):
    if (Vm == -0.034):                       # prevent division by zero
        return 500;                          # potassium activation rate constant
    else:
        return (5*(10**4)*(Vm + .034))/(1 - e(-100*(Vm + .034)))

def beta_n_eq(Vm):
    return 625*e(-12.5*(Vm + .044))

def n_f_eq(Vm):
    return alpha_n_eq(Vm)/(alpha_n_eq(Vm) + beta_n_eq(Vm))

def tau_n_eq(Vm):
    return 1/(alpha_n_eq(Vm) + beta_n_eq(Vm))

## mT
def mCaT_f_eq(Vm):
    return 1/(1 + e((-(Vm + .052))/(.0074)))

## hT
def hCaT_f_eq(Vm):
    return 1/(1 + e(500*(Vm + .076)))

def tau_hCaT_eq(Vm):
    if (Vm < -.080):
        return .001*e(15*(Vm + .467))
    else:
        return .028 + .001*e((-(Vm + .022))/(.0105))

## Currents
def I_leak_eq(Vm):
    return G_leak*(Vm - E_leak)

def I_Na_eq(m,h,Vm):
    return G_Na*(m**3)*h*(Vm - E_Na)

def I_K_eq(n,Vm):
    return G_K*(n**4)*(Vm - E_Na)

def I_CaT_eq(mT,hT,Vm):
    return G_CaT*(mT**2)*hT*(Vm - E_Ca)

# Initial conditions
V0 = E_leak
n0 = n_f_eq(E_leak)
h0 = h_f_eq(E_leak)
h_CaT0 = hCaT_f_eq(E_leak)

m_f = m_f_eq(E_leak)
m_CaT_f = mCaT_f_eq(E_leak)

# Method: Runge-Kutta 4th order
def calcium_Current_T_Type(t,y):
    V, n, h, h_CaT = y
    
    I_leak = I_leak_eq(V)  
    I_Na = I_Na_eq(m_f, h, V)
    I_K = I_K_eq(n, V) 
    I_CaT = I_CaT_eq(m_CaT_f, h_CaT, V)
    
    if (t < step_on):
        Jin = Jbase_value
    elif (step_on < t < step_off):
        Jin = Jbase_value + Jstep_value
    else:
        Jin = Jbase_value
        
    Iion = Jin - I_K - I_Na - I_leak - I_CaT;
    
    dVdt = Iion/Cm
    dndt = alpha_n_eq(V)*(1 - n) - beta_n_eq(V)*n
    dhdt = alpha_h_eq(V)*(1 - h) - beta_h_eq(V)*h
    dh_CaTdt = (hCaT_f_eq(V) - h_CaT)/tau_hCaT_eq(V)
    
    return dVdt, dndt, dhdt, dh_CaTdt


solution = scp.solve_ivp(calcium_Current_T_Type, 
                               [step_on, step_off],
                               [V0, n0, h0, h_CaT0],
                               t_eval=t)

'''
# Plots
fig2, axs2 = plt.subplots(5, sharex=True, figsize=(14,13))
fig2.suptitle("Questão 2 - letra a")
axs2[0].set_title('n')
axs2[0].plot(t,n)
axs2[0].set(ylabel='n')
axs2[0].grid()

axs2[1].set_title('h')
axs2[1].plot(t,h)
axs2[1].set(ylabel='h')
axs2[1].grid()

axs2[2].set_title('h_CaT')
axs2[2].plot(t,h_CaT)
axs2[2].set(ylabel='h_CaT')
axs2[2].grid()

axs2[3].set_title("Tensão - V")
axs2[3].set(ylabel='V')
#axs2[3].set_yticks(np.arange(-85, 55, step = 10))
axs2[3].plot(t,V)
axs2[3].grid()

axs2[4].set_title("J - Densidade de corrente")
axs2[4].plot(t,J)
axs2[4].set(ylabel='J (uA/cm^2)')
axs2[4].set_xticks(np.arange(0, simulationTime + 1, step = 50))
axs2[4].set(xlabel='t (ms)')
axs2[4].grid()
'''