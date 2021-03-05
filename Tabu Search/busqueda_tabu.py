import numpy as np
import copy
import random
from typing import Tuple

def std(M_sols : list, mean : float) -> float:
    """
    Population Standard Deviation

    Paramenteres
    ------------
    M_sols: `list`.
    The list of solutions

    mean: `float`.
    The population mean

    Returns
    ------------
    `float`.
    The value of the population standad Deviation

    """
    return np.sqrt(sum([(sol_i[1] - mean)**2 for sol_i in M_sols])/len(M_sols))

def get_median(M_sols: list):
    """
    Gets the midvalue of the solutions.

    Parameters
    ------------
    M_sols: `list`.
    The list of solutions to get the median from

    Returns
    ------------
    `list`.
    The value or values 
    """
    if len(M_sols)%2==0:
        indexes = (int(len(M_sols)/2), (int(len(M_sols)/2) +1))
        return M_sols[indexes[0]], M_sols[indexes[1]]
    else:
        indexes = len(M_sols)//2
        return M_sols[indexes]

def read_input(filename: str) -> Tuple[int, int, list]:
    """
    Read input from file

    Parameters
    ------------
    filename: `str`.
    The relative or absolute path of the file to read

    Returns
    ------------
    n: `int`.
    Number of cities
    i_max: `int`.
    Maximum number of iterations
    costos: `list`.
    A tringular matrix of the costs. This is the one used to generate the adjacency matrix

    Raises
    ------------
    `FileNotFoundError`
    The specified file is not found
    """
    try:
        with open(filename) as f:
            contents = f.readlines()
    except:
        raise FileNotFoundError

    n = int(contents.pop(0))
    i_max = int(contents.pop(0))
    costos = []
    for city in contents:
        costos_ni = list(map(int,city.split()))
        costos.append(costos_ni)
    costos.append([])
    return n,i_max,costos

def get_cost(costo_ciudades : np.ndarray, camino : list) -> int:
    """ 
    Returns cost of following the path provided

    Parameters
    ------------
    costo_ciudades: `numpy.ndarray`
    The adjacency matrix of cost.

    camino: `list`
    The path to follow.

    Returns
    ------------
    costo: `int`
    The cost of following the path given the adjacency matrix.
    """

    costo = 0
    camino = camino[1:] + [camino[0]]
    for i in range(len(camino)):
        costo += costo_ciudades[camino[i-1], camino[i]]
    return costo

def greedy_sol(costo_ciudades : np.ndarray) -> list:
    """
    Generates the initial solution based on a geedy solution

    Parameters
    ------------
    costo_ciudades: `numpy.ndarray`
    The adjacency matrix of cost.

    Returns
    ------------
    initial_sol: `list`
    The greedy solution to be used as the initial solution
    """
    costo = 0
    initial_sol = [0] #empezamos en 0
    costo_ciudades[:, 0] = np.inf # lo tachamos porque ya lo usamos
    while(len(initial_sol) < len(costo_ciudades)):
        min_travel_index = costo_ciudades[initial_sol[-1]].argmin()
        initial_sol.append(min_travel_index)
        costo_ciudades[:,min_travel_index] = np.inf

    return initial_sol

def generate_neighborhood(permutation : list, tabu_list : dict) -> Tuple[int, list]:
    """
    Generates the neighborhood based on a given permutation

    Parameters
    ------------
    permutation: `list`
    The path to follow

    tabu_list: `dict`
    The tabu list

    Returns
    ------------
    permutation[random_index] (city): `int`
    neighborhood: `list`
    """

    ### hay que quitar el 0!!!, porque es tu ciudad inicial, por eso son n-2
    neighborhood = []
    permutation = permutation[1:]
    random_index = random.randint(0, len(permutation)-1)
    while len(tabu_list) > 0 and permutation[random_index] in tabu_list:
        random_index = random.randint(0, len(permutation)-1)

    for i in range(len(permutation)):
        permutation_aux = copy.copy(permutation)
        if i == random_index:
            continue
        # print(permutation_aux)
        permutation_aux[i], permutation_aux[random_index] = permutation_aux[random_index], permutation_aux[i]
        # print(permutation_aux)
        permutation_aux = [0] + permutation_aux
        neighborhood.append(permutation_aux)

    return permutation[random_index], neighborhood

def update_tabu_list(tabu_list: dict, N: int, item = None) -> dict:
    """
    Updates time value, add items to the list and remove them if necessary

    Parameters
    ------------
    tabu_list: `dict`
    The tabu list

    N: `int`
    The number of cities, used to calculate the time using `N//2`

    item:
    the item to introduce to the tabu list.

    Returns
    ------------
    tabu_list: `dict`.
    The updated tabu list with new times and values
    """
    keys_to_pop = []
    if len(tabu_list)  > 0:
        for k in tabu_list.keys():
            if tabu_list[k] == 0:
                keys_to_pop.append(k)
                continue
            tabu_list[k] -= 1
    if len(keys_to_pop) > 0:
        for k in keys_to_pop:
            tabu_list.pop(k)
        
    if item is not None:
        time = N//2
        tabu_list[item] = time

    return tabu_list

def tabu_search(matrix : np.ndarray, i_max : int = 100, return_candidate_sols = False) -> tuple:
    """
    The function to perfom tha tebu search. It begins with a greedy solution and then moves forward

    Parameters
    ------------
    i_max: `int` default = 100.
    the maximum number of iterations for the tabu search

    return_candidate_sols: `bool` default = False.
    Flag to return `candidate_sols`

    matrix: `numpy.ndarray`.
    The adjacency matrix with costs of the cities.


    Returns
    ------------
    besto_sol: `tuple`.
    The best solution found by the search

    k_besto: `int`.
    Iteration where `best_sol` was founded

    candidate_sols: `list`
    The list of the best solution of each neighborhood, only returns when `return_candidate_sols = True`
    """
    tabu_list = {}
    besto_sol = (init_sol,get_cost(matrix,init_sol))
    candidate_sols = [besto_sol]
    listas_tabu = [tabu_list]
    besto_local_sol = besto_sol

    k = 0
    k_besto = k
    while k < i_max:
        k += 1
        neighborhood_w_cost = []
        city, neighborhood = generate_neighborhood(besto_local_sol[0], tabu_list)
        for n in neighborhood:
            neighborhood_w_cost.append((n, get_cost(matrix, n)))

        besto_local_sol = sorted(neighborhood_w_cost, key = lambda x : x[1])[0]
        candidate_sols.append(besto_local_sol)

        if besto_local_sol[1] < besto_sol[1]:
            besto_sol = besto_local_sol
            k_besto = k        
        tabu_list = update_tabu_list(tabu_list, N = 10, item = city)
        listas_tabu.append(tabu_list)
    
    if return_candidate_sols:
        return besto_sol, k_besto, candidate_sols
    else:
        return besto_sol, k_besto

def statistical_analysis(tabu_max_iters : int, M:int = 20, return_values = False, return_M_sols = False) -> tuple:
    """
    Function to perform analysis on the results of the tabu search.

    Parameters
    ------------

    tabu_max_iters: `int`
    The max number of iterations fot the tabu search
    
    M: `int`
    Max number of iteretations to run the tabu search

    return_values: `boolean` default: `False`.
    Flag to return the values, if set to `True`, it will not print nor calculate the mean,std and median

    return_M_sols: `boolean` default:`False`.
    Flag to return all the solutions founded in the `M` iterations

    Returns
    ------------
    Prints the mean, std and the median as well as the best solution and worst solution of the `M` iters

    if return_values is set to `True` a `tuple` that contains:

    best_m_sol: `list`.
    The best solution founded in the `M` iters

    worst_m_sol: `list`.
    The worst solution founded in the `M` iters

    if return_M_sols is `True`, it will also return

    M_sols: `list`.

    The list of all the solutions founded

    """
    worst_m_sol = [[],0]
    best_m_sol = [[],np.inf]
    M_sols = []
    for _ in range(M):
        besto_sol, k_besto = tabu_search(matrix, i_max = tabu_max_iters ,return_candidate_sols = False) 
        M_sols.append(besto_sol)
        if besto_sol[1] > worst_m_sol[1]:
            worst_m_sol = besto_sol
        elif besto_sol[1] < best_m_sol[1]:
            best_m_sol = besto_sol


    values_to_return = tuple((best_m_sol, worst_m_sol))
    if return_M_sols:
        if not return_values:
            return_values = True
        values_to_return.append(M_sols)

    if return_values:
        return return_values

    mean = sum([M_i[1] for M_i in M_sols])/M 
    median = get_median(M_sols) 
    sigma = std(M_sols, mean)
    print(f"{M} iterations for {i_max} iteratios of tabu search: ")
    print(f"Best solution: {best_m_sol}")
    print(f"Worst solution: {worst_m_sol}")
    print(f"Mean value of the objective function: {mean}")
    print(f"Median solution: {median}")
    print(f"The standard deviation: {sigma}")

if __name__ == '__main__':

    n,i_max,costos = read_input('input.txt')

    matrix = np.diag([np.inf] * n)
    for i in range(n):
        matrix[i,i+1:] = costos[i]

    matrix = np.where(matrix,matrix,matrix.T); print(matrix)

    init_sol = greedy_sol(copy.copy(matrix)); print(init_sol)


    besto_sol, k_besto = tabu_search(matrix,return_candidate_sols = False) 

    print(besto_sol)
    print(k_besto)

    statistical_analysis(tabu_max_iters=i_max)