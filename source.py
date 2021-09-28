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
