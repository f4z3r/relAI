#!/usr/bin/env python3

from gurobipy import *

"""Module containing helper functions use to solve linear optimisations."""

def linear_solver_layerwise(weights, biases, l_bounds, u_bounds, neurons_next_l):

    #TODO: to complete (objective)

    """
    Params: see bounds_linear_solver_layerwise

    Return: gurobi LP model, objective expression
    """

    #Create gurobipy linear solver
    m = Model("layerwise_linear_solver")
    n_bounds = l_bounds.shape()[0]

    #Create variables and constraints of linear solver
    for i in range(n_bounds):

        x_i="x"+str(i)
        #xi >= lower bound && xi <= upper bound
        m.addVar(lb=l_bounds[i], ub=u_bounds[i], vtype=GRB.CONTINUOUS, name=x_i )

    #TODO: sum of all z_i objective ? Maybe wrong objective
    z_i_sum=LinExpr()

    for i in range(neurons_next_l):

        zi="z"+str(i)

        m.addVar(vtype=GRB.CONTINUOUS,name=z_i)

        zi_expr=LinExpr()
        #z_i = sum(Wij*xi) --> for both lower and upper bounds
        for j in range(n_bounds):
            zi_expr += weights[i][j] * m.getVarByName("x"+str(j))

        m.addConstr(m.getVarByName(zi),GRB.EQUAL,zi_expr,"c"+i)

    return m, obj

def linear_solver_neuronwise(weights, bias, xi_lbounds, xi_ubounds):

    #TODO: to be tested

    """
    Params: see bounds_linear_solver_neuronwise

    Return: gurobi LP model, objective expression
    """

    #Create gurobipy linear solver
    m = Model("neuronwise_linear_solver")
    n_bounds = l_bounds.shape()[0]

    #Create variables and constraints of linear solver
    for i in range(n_bounds):
        x_i="x"+str(i)
        #xi >= lower bound && xi <= upper bound
        m.addVar(lb=l_bounds[i], ub=u_bounds[i], vtype=GRB.CONTINUOUS, name=x_i )

    #z next layer neuron output
    z = LinExpr()

    #z = sum(wi*xi)
    for i in range(n_bounds):
        z += weights[i] * m.getVarByName("x"+str(i))
    z += bias

    return m, z

def bounds_linear_solver_neuronwise(weights, bias, xi_lbounds, xi_ubounds):

    #TODO: to be tested

    """
    Params:
    -weights: n x 1 vector    | representing the weights coming from the previous layer neurons to the next layer neuron z
    -bias: scalar             | representing the bias to the next layer neuron z value
    -xi_lbounds: n x 1 vector | representing the lower bounds of the previous layer neurons
    -xi_ubounds: n x 1 vector | representing the upper bounds of the previous layer neurons

    where n is the number of the previous layer neurons

    Return:
    - neuron_lb: scalar  | representing the scalar lower bound of the next layer neuron z
    - neuron_ub: scalar  | representing the scalar upper bound of the next layer neuron z
    """

    model, z = linear_solver_neuronwise(weights, bias, xi_lbounds, xi_ubounds)

    #Find upper bound of the neuron z
    model.SetObjective(z, GRB.MAXIMIZE)
    model.optimize()
    #Applying ReLU on neuron_ub
    neuron_ub = z.X if z.X > 0 else 0

    model.reset(0)

    #Find lower bound of the neuron z
    model.SetObjective(z, GRB.MINIMIZE)
    model.optimize()
    #Applying ReLU on neuron_lb
    neuron_lb = z.X if z.X > 0 else 0

    return neuron_lb,neuron_ub


def bounds_linear_solver_layerwise(weights, bias, xi_lbounds, xi_ubounds):

    #TODO: to be completely correctly defined along with linear_solver_layerwise

    """
    Params:
    -weights: n x m vector    | representing the n weights coming from the previous layer n neurons to the m next layer neurons z_i
    -bias: m x 1 vector       | representing the bias to the m next layer neurons z_i values
    -xi_lbounds: n x 1 vector | representing the lower bounds of the previous layer neurons
    -xi_ubounds: n x 1 vector | representing the upper bounds of the previous layer neurons

    where n is the number of the previous layer neurons

    Return:
    - neurons_lb: m x 1 vector | representing the scalar values of the next layer z_i neurons' lower bounds
    - neurons_ub: m x 1 vector | representing the scalar values of the next layer z_i neurons' upper bounds
    """

    model, obj = linear_solver_layerwise(weights, bias, xi_lbounds, xi_ubounds)
    return neurons_lbs,neurons_ubs
