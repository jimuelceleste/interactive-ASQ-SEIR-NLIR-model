#!/usr/bin/env python
# coding: utf-8

# # Interactive ASQ-SEIR-NLIR Covid-19 Model

# ## Age-Stratified Quarantine-Modified SEIR with Non-Linear Incidence Rates (ASQ-SEIR-NLIR) COVID-19 Model (Bongolan et al. 2021; Rayo et al. 2020; Minoza et al. 2020)
# 
# The ASQ-SEIR-NLIR model is a deterministic epidemic (COVID-19) model. It features quarantine, age, and behavioral (e.g., mask-wearing, physical distancing, etc.) and disease-resistance (e.g., healthy living, natural immunity, etc.) factors. It is described by a system of differential equations:
# 
# $\S' = \frac{-\beta Q(t) S I/N}{(1 + \alpha S/N)(1 + \epsilon I/N)}$
# 
# $E' = \frac{\beta Q(t) S I/N}{(1 + \alpha S/N)(1 + \epsilon I/N)} - \sigma U E$
# 
# $I' = \sigma U E - \gamma I$
# 
# $R' = \gamma I$
# 
# where $\beta$, $\sigma$, $\gamma$ are the transmission, incubation, and removal rates; $Q(t)$ is the time-varying quarantine variable (Bongolan et al. 2021); $U$ is the age-stratified infection expectation (Rayo et al. 2020); $\alpha$ and $\epsilon$ are the non-linear incidence rates (Minoza et al. 2020). 
# 
# 

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import ipywidgets as widgets

def get_model(params, n):
    def asq_seir_nlir(seir, t):
        beta = params[0]
        sigma = params[1]
        gamma = params[2]
        q = params[3]
        u = params[4]
        alpha = params[5]
        epsilon = params[6]

        s = seir[0]
        e = seir[1]
        i = seir[2]
        r = seir[3]

        # Non-linear Incidence Rate 
        nlir = 1 / ((1 + alpha * s/n) * (1 + epsilon * i/n))

        # Differential Equations
        dsdt = -beta * q * s * i/n * nlir
        dedt = beta * q * s * i/n * nlir  - sigma * u * e
        didt = sigma * u * e - gamma * i
        drdt = gamma * i

        return [dsdt, dedt, didt, drdt]
    return asq_seir_nlir

def solve_system(t_max, n, s0, e0, i0, r0, sigma, beta, gamma, q, u, alpha, epsilon):
    params = [sigma, beta, gamma, q, u, alpha, epsilon]
    seir0 = [s0, e0, i0, r0]
    t = np.linspace(0, t_max)
    
    seir = odeint(get_model(params, n), seir0, t)    
    s = seir[:,0]
    e = seir[:,1]
    i = seir[:,2]
    r = seir[:,3]
    
    plt.plot(t, s, label="S")
    plt.plot(t, e, label="E")
    plt.plot(t, i, label="I")
    plt.plot(t, r, label="R")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("individuals")
    plt.show()
    
    return seir

t_max = widgets.IntText(
    description="Max Time", 
    value=100)
s0 = widgets.FloatText(
    description=r"$S_0$", 
    value=80)
e0 = widgets.FloatText(
    description=r"$E_0$", 
    value=10)
i0 = widgets.FloatText(
    description=r"$I_0$", 
    value=10)
r0 = widgets.FloatText(
    description=r"$R_0$", 
    value=0)
n = widgets.FloatText(
    description=r"$N$", 
    value=s0.value + e0.value + i0.value + r0.value)
beta = widgets.FloatText(
    description=r"$\beta$", 
    value=3.2, 
    min=0, 
    step=0.01)
sigma = widgets.FloatSlider(
    description=r"$\sigma$", 
    value=1/7, 
    min=0,
    max=1,
    step=0.01)
gamma = widgets.FloatSlider(
    description=r"$\gamma$", 
    value=0.3, 
    min=0,
    max=1,
    step=0.01)
q = widgets.FloatSlider(
    description=r"$Q(t)$", 
    value=0.2, 
    min=0,
    max=1,
    step=0.01)
u = widgets.FloatSlider(
    description=r"$U$", 
    value=0.5, 
    min=0,
    max=1,
    step=0.01)
alpha = widgets.FloatSlider(
    description=r"$\alpha$", 
    value=0.1, 
    min=0,
    max=1,
    step=0.01)
epsilon = widgets.FloatSlider(
    description=r"$\epsilon$", 
    value=0.1, 
    min=0,
    max=1,
    step=0.01)

ui = widgets.VBox([t_max, s0, e0, i0, r0, n, beta, sigma, gamma, q, u, alpha, epsilon])
out = widgets.interactive_output(
    solve_system, 
    {
        "t_max": t_max, 
        "n": n, 
        "s0": s0, 
        "e0": e0, 
        "i0": i0, 
        "r0": r0, 
        "sigma": sigma, 
        "beta": beta, 
        "gamma": gamma, 
        "q": q, 
        "u": u, 
        "alpha": alpha, 
        "epsilon": epsilon
    })

display(ui, out)


# # References 
# 
# Bongolan, V.P., Minoza, J.M.A., de Castro, R., and Sevilleja, J.E., 2021, Age-Stratified Infection Probabilities Combined With a Quarantine-Modified Model for COVID-19 Needs Assessments: Model Development Study. J Med Internet Res 2021; 23(5): e19544. DOI: 10.2196/19544
# 
# Minoza, J. M., Sevilleja, J. E., de Castro, R., Caoili S.E., Bongolan, V. P., 2020, Protection after Quarantine: Insights from a Q-SEIR Model with Nonlinear Incidence Rates Applied to COVID-19. medRxiv [pre-print]. doi.org/10.1101/2020.06.06.2012
# 
# Rayo, J.F., de Castro, R., Sevilleja, J.E., and Bongolan, V.P., 2020, Modeling the dynamics of COVID-19 using Q-SEIR model with age-stratified infection probability. medRxiv [preprint]. DOI: 10.1101/2020.05.20.20095406

# In[ ]:




