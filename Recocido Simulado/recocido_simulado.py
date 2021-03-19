import copy
import random
import math
import numpy as np

def f(mochila, p):
    return sum([mi * pi for (mi,pi) in zip(mochila, p)])

def g(mochila, w):
    return sum([mi * wi for (mi,wi) in zip(mochila, w)])

def T(t):
    return 0.99*t

def read_input(filename):
    with open(filename) as f:
        lines = f.readlines()
    ti,tf = map(float,lines.pop(0).split())
    N = int(lines.pop(0))
    c = float(lines.pop(0))
    items = []
    for i in range(N):
        p,w = map(float,lines.pop(0).split())
        items.append((p,w))

    return ti,tf,N,c,items

def initial_sol(N,items,c):
    initial_sol = [random.randint(0,1) for _ in range(N)]
    w = [_[1] for _ in items]
    cost = g(initial_sol, w)
    while cost >= c:
        initial_sol = [random.randint(0,1) for _ in range(N)]
        cost = g(initial_sol, w)
    return initial_sol

def generate_neighborhood(mochila, c, items):
    neighborhood = []
    w = [_[1] for _ in items]
    mochila_aux = copy.copy(mochila)
    for i in range(len(mochila)):
        if mochila_aux[i] == 0:
            mochila_aux[i] = 1
        else:
            mochila_aux[i] = 0
        cost = g(mochila_aux, w)
        if cost <= c:
            neighborhood.append(mochila_aux)
        
        mochila_aux = copy.copy(mochila)
         
    return neighborhood

ti,tf,N,c,items = read_input('input.txt')


# evaluar valor de la mochila en lugar de los pesos porque esos ya los tengo
p = [_[0] for _ in items]
x0 = initial_sol(N,items, c)
x = x0
fx = f(x0, p)
t = ti
x_besto = x.copy()
f_best = fx
print("Initial sol: ", x, fx)
while t >= tf:
    new_x = random.choice(generate_neighborhood(x, c, items))
    new_f = f(new_x, p)
    
    if new_f >= f_best:
        x_besto = new_x.copy()
        f_best = new_f

    else:
        if new_f < fx or np.random.random() < math.exp(-1.*(new_f - fx)/t):
            x = new_x
            fx = new_f
    
    t = T(t)
    
print(x_besto, f_best)

