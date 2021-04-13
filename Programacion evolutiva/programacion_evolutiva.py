# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import os
import math
import numpy as np

from typing import Tuple, List

# %% [markdown]
# Usando programacion evolutiva, resolver el siguente probelma:
# 
# $$
# \min f(\vec{x}) = -20exp \left( -0.2 \sqrt{\frac{1}{n} \sum_{i=1}^n x^2_i} \right) - exp \left(\frac{1}{n}\sum_{i=1}^n cos(2\pi x_i) \right) + 20 + e
# $$

# %%
MAX_DOMAIN = 32
MIN_DOMAIN = -32


# %%
def read_input(filename : str = "input.txt") -> Tuple[int, int, float, float]:
    try:
        with open(filename) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"El archivo {filename} no fue encontrado")
        return
    except Exception as e:
        print(f"ha ocurrido un problema {e}")
        return
    
    n = int(lines.pop(0))
    mu, g = map(int, lines.pop(0).split())
    alpha, epsilon = map(float, lines.pop(0).split())
    return n,mu,g,alpha,epsilon


# %%
def std(M_list: list, mean: float) -> float:
    return np.sqrt(sum([(sol_i[-1] - mean)**2 for sol_i in M_list])/len(M_list))


# %%
def f(x : List[float], n : int) -> float:
    coso1 = -0.2*np.sqrt(sum([xi**2 for xi in x[:n]])/n)
    coso2 = (sum([math.cos(2*math.pi*xi) for xi in x[:n]])/n)
    return -20*math.e**(coso1) -math.e**(coso2) + 20 + math.e


# %%
def initial_population(mu : int, n : int) -> List[List[float]]:
    padres = []
    for i in range(mu):
        p = np.concatenate([np.random.uniform(MIN_DOMAIN, MAX_DOMAIN, n) ,np.random.uniform(0,1, n)])
        p = np.concatenate([p, [f(p, n)]])
        padres.append(p)
    return padres


# %%
parents = initial_population(100,2); parents


# %%
parents[0][2:-1] * (100 *np.random.normal(0,1,2))


# %%
def create_individual(parent : List[float], n: int, alpha : float, epsilon : float):
    # print(parent.shape)
    offspring_sigma = parent[n:-1]*(1 + (alpha *np.random.normal(0,1,n)))
    offspring_sigma[offspring_sigma < epsilon] = epsilon

    offspring_x = parent[:n] + (offspring_sigma* np.random.normal(0,1, n))
    offspring_x[offspring_x < MIN_DOMAIN] = MIN_DOMAIN
    offspring_x[offspring_x > MAX_DOMAIN] = MAX_DOMAIN

    return np.concatenate([offspring_sigma, offspring_x])


# %%
np.concatenate ([np.random.uniform(-1, 1, 3), np.random.uniform(0,1, 3)])


# %%
def programacion_evolutiva(n : int,mu : int,g : int ,alpha : float,epsilon : float):
    padres = initial_population(mu, n)
    t = 1
    while t <= g:
        current_gen = padres.copy()
        for p in padres:
            offspring = create_individual(p, n, alpha, epsilon)
            current_gen.append(np.concatenate([offspring, [f(offspring, n)]]))
        
        padres = sorted(current_gen, key = lambda x : x[-1])[:mu]
        t+=1
    return padres[0]    #la mejor sol


# %%
programacion_evolutiva(*read_input())


# %%
f([1.00000000e-04,  1.00000000e-04,  2.24971042e+01, -5.69032933e+00], 2)


# %%
def statistical(M : int = 10, filename : str = 'input.txt'):
    # n,mu,g,alpha,epsilon
    args = read_input(filename)
    sols = []
    for _ in range(M):
        sols.append(programacion_evolutiva(*args))
    
    sols = sorted(sols, key = lambda x : x[-1], reverse=True)
    print(f"Besto sol : {sols[0]}")
    print(f"Worsto sol: {sols[-1]}")
    print(f"Mediana : {sols[M//2]}")
    mean = sum([_[-1] for _ in sols])/M
    desviacion = std(sols, mean)
    print(f"Media {mean}")
    print(f"STD : {desviacion}")


# %%
print(f"2 variables")
statistical()
print(f"10 variables")
statistical(filename="input10.txt") # 10 variables
print(f"20 variables")
statistical(filename="input20.txt") # 20 variables