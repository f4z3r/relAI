#!/usr/bin/env python3

import numpy as np
from gurobipy import *

# This needs to be placed in the analyzer file as it uses the interval solver
# to find the overapproximations for the ReLU

def linear_solver_layerwise(nn, man, lbounds, ubounds, layer_start,
                            layer_stop, timeout):
    """Builds a linear model of the contraints for a specified number of layers
    for a neural network.

    Args:
        - nn: the neural network to perform linear programming on
        - man: the elina manager that manages the abstract domains
        - lbounds: the lower bounds to inject in the first layer on which the
          linear programming is performed
        - ubounds: the upper bounds to inject in the first layer on which the
          linear programming is performed
        - layer_start: the first layer to perform the linear programming on
        - layer_stop: the last layer to perform the linear programming on
        - timeout: the timeout in seconds for each solving; note that the
          solver is actually called many times to solve the layers

    Returns:
        The dimensions of the bounds of the last propagated layer as well as
        the bounds themselves.

    Notes:
        This currently solves the ReLU in the last layer as well using linear
        programming. This is not necessary, as only the affine computation
        can be used and the ReLU can be computed manually in order to make the
        solving slightly faster. (I am not sure if this will actually have a
        significant impact on performance, but it will surely not be a bad
        idea).
    """
    assert len(lbounds) == len(ubounds), "bounds should have the same length"
    assert layer_start < layer_stop, "start layer must be before stop layer"

    # create gurobi linear model
    m = Model("layerwise_linear_solver")
    m.setParam("TimeLimit", timeout)

    # create box constraints on initial layer
    for neuron, (lb, ub) in enumerate(zip(lbounds, ubounds)):
        neuron_name = f"lay{layer_start-1}_{neuron}"
        neuron_var = m.addVar(lb=lb, ub=ub, name=neuron_name)

    m.update()   # process pending modifications

    # for each layer, add the constraints
    for layer in range(layer_start, layer_stop):
        weights = nn.weights[layer] # weights for each nueron in next layer
        biases = nn.biases[layer]   # biases of each neuron in next layer

        # compute interval bounds on layer
        bound_size, bounds = interval_propagation(nn, man, lbounds, ubounds,
                                                  layer, layer+1)
        lbounds, ubounds = el_bounds_to_list(bounds, bound_size)

        # create variables for neurons in next layer, doing this early ensures
        # one needs to call update only once per layer
        for neuron in range(len(weights)):
            neuron_name = f"lay{layer}_{neuron}"
            neuron_var = m.addVar(lb=lbounds[neuron], ub=ubounds[neuron],
                                  name=neuron_name)

        m.update()   # make neuron vars available for contraint building

        # loop over neuron in next layer to set contraints
        for neuron, neuron_weights in enumerate(weights):
            neuron_name = f"lay{layer}_{neuron}"

            affine_sum = LinExpr()
            # build affine sum
            for prev_neuron in range(len(neuron_weights)):
                prev_neuron_name = f"lay{layer-1}_{prev_neuron}"
                affine_sum += neuron_weights[prev_neuron] *\
                              m.getVarByName(prev_neuron_name)

            # build relu contraints
            # ReLU(z) >= z
            constr_name = f"lay{layer}_{neuron}_id"
            m.addConstr(m.getVarByName(neuron_name), GRB.GREATER_EQUAL,
                        affine_sum, constr_name)
            # ReLU(z) <= (ub_z / ub_z - lb_z) * z + (ub_z * lb_z / lb_z - ub_z)
            constr_name = f"lay{layer}_{neuron}_over"
            ub = ubounds[neuron]
            lb = lbounds[neuron]
            lhs = LinExpr((ub * lb) / (lb - ub))
            lhs += (ub / (ub - lb)) * affine_sum
            m.addConstr(lhs, GRB.GREATER_EQUAL, m.getVarByName(neuron_name),
                        constr_name)
            # ReLU(z) >= 0
            constr_name = f"lay{layer}_{neuron}_gt0"
            m.addConstr(m.getVarByName(neuron_name), GRB.GREATER_EQUAL, 0,
                        contr_name)

    # all layers built, compute value of layer_stop - 1
    m.update()
    lbounds = []
    ubounds = []
    for neuron in range(len(nn.weights[layer_stop-1])):
        neuron_name = f"lay{layer_stop-1}_{neuron}"
        m.setObjective(m.getVarByName(neuron_name), GRB.MAXIMIZE)
        m.optimise()
        ubounds.append(m.getVarByName(neuron_name))
        m.reset(0)
        m.setObjective(m.getVarByName(neuron_name), GRB.MINIMIZE)
        m.optimise()
        lbounds.append(m.getVarByName(neuron_name))

    return lbounds, ubounds
