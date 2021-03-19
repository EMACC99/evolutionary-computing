import copy
import random
import math
import numpy as np

from typing import Tuple
def mediana(M_sols) -> list:
    if len(M_sols)%2==0:
        indexes = (int(len(M_sols)/2), (int(len(M_sols)/2) +1))
        return M_sols[indexes[0]], M_sols[indexes[1]]
    else:
        indexes = len(M_sols)//2
        return M_sols[indexes]

def std(M_list: list, mean: float) -> float:
    return np.sqrt(sum([(sol_i - mean)**2 for sol_i in M_list])/len(M_list))


def f(mochila: list, p: list) -> float:
    """
    Calulates the value of the backpack

    Paramenters
    --------------
    mochila : `list` the backpack to evaluate
    p       : `list` the list of values of the items


    Returns
    ------------
    `float` the value of the backpack
    """
    return sum([mi * pi for (mi,pi) in zip(mochila, p)])

def g(mochila : list, w : list) -> float:
    """
    Calculates the weight of the backpack
    
    Parameters
    --------------
    mochila : `list` the backpack to evaluate
    w       : `list` the list of weights of the items

    Returns
    -------------
    `float` the weight of the backpack
    """
    return sum([mi * wi for (mi,wi) in zip(mochila, w)])

def T(t : float) ->  float:
    """
    Calculates the temperature

    Parameters
    ------------
    t : `float`

    Returns
    ----------
    `float` the next value of t
    """
    # return 0.99*t # con esto siempre encuentra el optimo por alguna razon quien sabe, supongo que es porque como disminuye menos, hace mas iteraciones y provoca que encuetre el optimo
    return 0.9*t # esta ya le varia a la soluicion que encuentra

def read_input(filename) -> Tuple[float, float, int, int, list]:
    """
    Function to read the input from a file provided

    Parameters
    ---------------
    filename : `str` the absolute or relative path to the file

    Returns
    ---------------

    A tuple with the following elements in order:

    ti    : `float` the initial temperature
    tf    : `float` the target temperature
    N     : `int` the number of items
    c     : `int` the capacity of the backpack
    items : `list` the list of size `N` containing the value and weigth of every item
    """
    try:
        with open(filename) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Archivo no encontrado")
        return
    
    ti,tf = map(float,lines.pop(0).split())
    N = int(lines.pop(0))
    c = float(lines.pop(0))
    items = []
    for i in range(N):
        p,w = map(float,lines.pop(0).split())
        items.append((p,w))

    return ti,tf,N,c,items



def initial_sol(N : int,items : list,c : float) -> list:
    """
    Generates the initial solution

    Parameters
    -------------
    N     : `int` The number of items
    items : `list` The list of the items
    c     : `float` The max capacity of the backpack

    Returns
    ----------
    initial_sol : `list` a list of 0's and 1's indicating which item is present in the current backpack
    """
    initial_sol = [random.randint(0,1) for _ in range(N)]
    w = [_[1] for _ in items]
    cost = g(initial_sol, w)
    while cost >= c:
        initial_sol = [random.randint(0,1) for _ in range(N)]
        cost = g(initial_sol, w)
    return initial_sol

def generate_neighborhood(mochila : list, c : float, items : list) -> list:
    """
    Generates the neighborhood for simulated annealing

    Parameters
    -----------
    mochila : `list` the backpack
    c       : `float` the capacity of the backpack
    items   : `list` the list of weigths and values of the items


    Returns
    ---------
    `list` A list containing the neighborhood
    """
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

def get_items(sol : list) -> list:
    """
    Returns the index of the items that are in the backpack

    Parameters
    -----------
    sol : `list` the backpack to get the indexes from

    Returns
    -----------
    `list` A list containing the indexes
    """
    return [i for i in range(len(sol)) if sol[i] != 0]

def recocido_simulado(ti : float, tf : float, N : int, c :float, items : list) -> Tuple[list, list, float, float]:
    """
    perfomrs simulated annealing

    Parameters
    ------------
    ti    : `float` the initial temperature
    tf    : `float` the target temperature
    N     : `int` The number of items
    c     : `float` the capacity of the backpack
    items : `list` the list of wights and values of the items

    Returns
    -------------
    Tuple with the following objects 
    `list` a list with the indexes of the items included in the backpack
    x_besto: `list` The best solution found by simulated annealing
    f_besto: `float` The value of the best solution found
    g_besto: `float` The weight of the best solution found
    """
    p = [_[0] for _ in items]
    x = initial_sol(N,items, c)
    fx = f(x, p)
    t = ti
    x_besto = x.copy()
    f_besto = fx
    print("Initial sol: ", x, fx)
    while t >= tf:
        x_current = random.choice(generate_neighborhood(x, c, items))
        f_current = f(x_current, p)
    
        if f_current >= f_besto:
            x_besto = x_current.copy()
            f_besto = f_current

        if f_current < fx or np.random.random() < math.exp(-1.*(f_current - fx)/t):
            x = x_current
            fx = f_current
    
        t = T(t)
    return (get_items(x_besto), x_besto, f_besto, g(x_besto, [_[1] for _ in items]))



def satatistical(M: int, filename : str = 'input.txt'):
    """
    Function to make an statistical analysis of simulated annealing and prints 

    Parameters
    ------------
    M        : `int` The number of iterations to run the simulated annealing
    filename : `str` The absolute path or relative path to the file with the input

    Returns
    ------------
    `None`
    """
    args = read_input(filename)
    
    sols = []
    besto_sol = [[],[], 0, 0]
    worst_sol = [[], np.inf ,np.inf]
    f_values = []
    g_values = []
    for _ in range(M):
        mochila, sol, f_sol, g_sol = recocido_simulado(*args)
        if f_sol < worst_sol[2]:
            worst_sol = [mochila, sol, f_sol, g_sol]
        elif f_sol > besto_sol[2]:
            besto_sol = [mochila,sol, f_sol, g_sol]

        sols.append([mochila, sol, f_sol, g_sol])
        f_values.append(f_sol)
        g_values.append(g_sol)
    
    print(f"Mejor solucion: {besto_sol}")
    print(f"Peor solucion: {worst_sol}")
    promedio_f = sum(f_values)/len(f_values)
    promedio_g = sum(g_values)/len(g_values)
    print(f"Promedio del valor de la mochila: {promedio_f}")
    print(f"Promedio del peso de la mochila:  {promedio_g}")
    print(f"Desviacion estandar de f: {std(f_values, promedio_f)}")
    print(f"Desviacion estandar de g: {std(g_values, promedio_g)}")
    # return besto_sol, worst_sol, sols


ti,tf,N,c,items = read_input('input.txt')
satatistical(10)