import sys
sys.path.insert(0, '../ELINA/python_interface/')

import numpy as np
import re
import csv
from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time


libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0


def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v


def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v.reshape((v.size,1))


def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i+1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result


def parse_matrix(text):
    i = 0
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    return np.array([*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))])


def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i+1])
            b = parse_bias(lines[i+2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer+= 1
            i += 3
        else:
            raise Exception('parse error: '+lines[i])
    return res


def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open('dummy', 'w') as my_file:
        my_file.write(text)
    data = np.genfromtxt('dummy', delimiter=',',dtype=np.double)
    low = np.copy(data[:,0])
    high = np.copy(data[:,1])
    return low,high


def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon

    for i in range(num_pixels):
        if(LB_N0[i] < 0):
            LB_N0[i] = 0
        if(UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0


def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)
    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0,i,weights[i])
    return linexpr0


def el_bounds_to_list(bounds, bounds_size):
    """"Converts ELINA bounds to two list of bounds, containing the lower and
    upper bounds respectively.

    Args:
        - bounds: the ELINA bounds to convert
        - bounds_size: the size of the ELINA bounds.

    Returns:
        The lower and upper bounds of the corresponding ELINA bounds.
    """
    lbounds = []
    ubounds = []
    for i in range(bounds_size):
        lbounds.append(bounds[i].contents.inf.contents.val.dbl)
        ubounds.append(bounds[i].contents.sup.contents.val.dbl)
    elina_interval_array_free(bounds, bounds_size)
    return lbounds, ubounds


def interval_propagation_el_bounds(nn, man, bounds_size, bounds, layer_start,
                                   layer_stop):
    """Performs an interval propagation similar to `interval_propagation` but
    allows to pass elina bounds and their size instead of two lists of upper
    and lower bounds.

    Args:
        - nn: the neural network to perform the interval propagation on
        - man: the elina manager used to manage abstract domains
        - bounds_size: the size of the input bounds passed
        - bounds: the elina bounds given an input
        - layer_start: the first layer on which to apply interval propagation
        - layer_stop: the last layer on which to apply interval propagation

    Returns:
        The dimensions of the bounds of the last propagated layer as well as
        the bounds themselves.
    """
    lbounds, ubounds = el_bounds_to_list(bounds, bounds_size)
    bounds_size, bounds = interval_propagation(nn, man, lbounds, ubounds,
                                               layer_start, layer_stop)
    return bounds_size, bounds


def interval_propagation(nn, man, lbounds, ubounds, layer_start, layer_stop):
    """Perform linear interval propagation on the neural network.

    Args:
        - nn: the neural network to propagate the intervals on
        - man: the elina manager that manages the abstract domains
        - lbounds: the lower bounds to inject in the first layer on which the
          interval propagation is performed
        - ubounds: the upper bounds to inject in the first layer on which the
          interval propagation is performed
        - layer_start: the first layer to perform the interval propagation on
        - layer_stop: the last layer to perform the interval propagation on

    Returns:
        The dimensions of the bounds of the last propagated layer as well as
        the bounds themselves.
    """
    assert len(lbounds) == len(ubounds), "bounds must have same length"
    assert layer_start < layer_stop, "start layer must be before stop layer"

    # inject the bounds into the current layer
    itv = elina_interval_array_alloc(len(lbounds))
    for i in range(len(lbounds)):
        elina_interval_set_double(itv[i], lbounds[i], ubounds[i])

    # inject bounds
    element = elina_abstract0_of_box(man, 0, len(lbounds), itv)
    elina_interval_array_free(itv, len(lbounds))

    # compute the interval propagation
    for layerno in range(layer_start, layer_stop):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
            weights = nn.weights[layerno]
            biases = nn.biases[layerno]
            # get the domain's dimensions
            dims = elina_abstract0_dimension(man,element)
            num_in_pixels = dims.intdim + dims.realdim
            num_out_pixels = len(weights)

            # allocate space for a change in dimension
            dimadd = elina_dimchange_alloc(0,num_out_pixels)
            for i in range(num_out_pixels):
                dimadd.contents.dim[i] = num_in_pixels
            # add dimensions based on the changed dimensions dimadd
            elina_abstract0_add_dimensions(man, True, element, dimadd, False)
            # free dimadd dimension object as no longer needed
            elina_dimchange_free(dimadd)
            # make weights a continuous np array
            np.ascontiguousarray(weights, dtype=np.double)
            # make biases a continuous np array
            np.ascontiguousarray(biases, dtype=np.double)
            var = num_in_pixels
            # handle affine layer
            for i in range(num_out_pixels):
                tdim= ElinaDim(var)
                # apply affine transformation ??
                linexpr0 = generate_linexpr0(weights[i], biases[i],
                                             num_in_pixels)
                element = elina_abstract0_assign_linexpr_array(
                    man, True, element, tdim, linexpr0, 1, None)
                var+=1
            dimrem = elina_dimchange_alloc(0, num_in_pixels)

            for i in range(num_in_pixels):
                dimrem.contents.dim[i] = i
            elina_abstract0_remove_dimensions(man, True, element, dimrem)
            elina_dimchange_free(dimrem)
            # handle ReLU layer
            if(nn.layertypes[layerno]=='ReLU'):
                element = relu_box_layerwise(man, True, element, 0,
                                             num_out_pixels)
            nn.ffn_counter+=1

        else:
            print(' net type not supported')

    # return upper and lower bounds
    dims = elina_abstract0_dimension(man, element)
    bounds_size = dims.intdim + dims.realdim
    bounds = elina_abstract0_to_box(man,element)
    # free the element
    elina_abstract0_free(man,element)
    return bounds_size, bounds


def get_relu_bounds(weights, biases, lbounds, ubounds):
    """"Computes the relu bounds for a layer in the neural network.

    Args:
        - weights: the weights between the current layer and the previous one
        - biases: the biases of the current layer
        - lbounds: the lower bounds of the neurons of the layer previous to the
          one on which the relu bounds are computed
        - ubounds: the upper bounds of the neurons of the layer previous to the
          one on which the relu bounds are computed

    Returns:
        A list of the lower bounds and a list of the upper bounds of the affine
        transformations on the `layer` given in the argumens.
    """
    assert len(lbounds) == len(ubounds), "bounds should have the same length"
    relu_lbounds = []
    relu_ubounds = []
    for neuron, neuron_weights in enumerate(weights):
        bias = biases[neuron]
        min_val = bias
        max_val = bias
        for prev_neuron, weight in enumerate(neuron_weights):
            if weight >= 0:
                min_val += weight * lbounds[prev_neuron]
                max_val += weight * ubounds[prev_neuron]
            else:
                min_val += weight * ubounds[prev_neuron]
                max_val += weight * lbounds[prev_neuron]
        relu_lbounds.append(min_val)
        relu_ubounds.append(max_val)
    return relu_lbounds, relu_ubounds


def linear_solver_layerwise(nn, man, lbounds, ubounds, layer_start, layer_stop,
                            timeout):
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
    m.setParam("OutputFlag", False)
    m.setParam("Presolve", 2)
    m.setParam("TimeLimit", timeout)

    # create box constraints on input to initial layer
    for neuron, (lb, ub) in enumerate(zip(lbounds, ubounds)):
        if ub > 0:
            neuron_name = f"x{layer_start-1}_{neuron}"
            neuron_var = m.addVar(lb=lb, ub=ub, name=neuron_name)

    m.update()   # process pending modifications

    # for each layer, add the constraints
    for layer in range(layer_start, layer_stop):
        weights = nn.weights[layer] # weights for each neuron in this layer
        biases = nn.biases[layer]   # biases of each neuron in this layer

        # compute interval bounds on layer
        prev_lbounds, prev_ubounds = lbounds, ubounds
        bound_size, bounds = interval_propagation(nn, man, lbounds, ubounds,
                                                  layer, layer+1)
        lbounds, ubounds = el_bounds_to_list(bounds, bound_size)
        relu_lbnds, relu_ubnds = get_relu_bounds(weights, biases, prev_lbounds,
                                                 prev_ubounds)

        # loop over neuron in next layer to set contraints
        for neuron, neuron_weights in enumerate(weights):
            lb = lbounds[neuron]
            ub = ubounds[neuron]

            # if neuron zero anyways, skip it
            if not ub > 0:
                continue

            # create the neuron value
            neuron_name = f"x{layer}_{neuron}"
            neuron_var = m.addVar(lb=lb, ub=ub, name=neuron_name)

            affine_sum = LinExpr(biases[neuron])
            # build affine sum
            for prev_neuron, synapse in enumerate(neuron_weights):
                # only add the neuron if it contributes
                if prev_ubounds[prev_neuron] > 0 and synapse != 0:
                    prev_neuron_name = f"x{layer-1}_{prev_neuron}"
                    affine_sum += synapse * m.getVarByName(prev_neuron_name)

            if nn.layertypes[layer] == "Affine" or lb > 0:
                m.addConstr(neuron_var, GRB.EQUAL, affine_sum)
            elif nn.layertypes[layer] == "ReLU":
                # ReLU(z) >= z
                m.addConstr(neuron_var, GRB.GREATER_EQUAL, affine_sum)
                # ReLU(z) <= (ub_z / ub_z - lb_z) * z +\
                #            (ub_z * lb_z / lb_z - ub_z)
                rub = relu_ubnds[neuron]
                rlb = relu_lbnds[neuron]
                rhs = LinExpr()
                rhs += (rub / (rub - rlb)) * (affine_sum)
                rhs += ((rub * rlb) / (rlb - rub))
                m.addConstr(neuron_var, GRB.LESS_EQUAL, rhs)
            else:
                print("layer type not supported")

        # make neuron vars available for constraint building in next layer
        m.update()

    # all layers built, compute value of layer_stop - 1
    curr_lbounds, curr_ubounds = lbounds, ubounds
    lbounds = []
    ubounds = []
    for neuron in range(len(nn.weights[layer_stop-1])):
        if curr_ubounds[neuron] > 0:
            neuron_name = f"x{layer_stop-1}_{neuron}"
            m.setObjective(m.getVarByName(neuron_name), GRB.MAXIMIZE)
            m.optimize()
            ubounds.append(m.getVarByName(neuron_name).X)
            m.setObjective(m.getVarByName(neuron_name), GRB.MINIMIZE)
            m.optimize()
            lbounds.append(m.getVarByName(neuron_name).X)
        else:
            ubounds.append(0)
            lbounds.append(0)

    return lbounds, ubounds


def analyze(nn, LB_N0, UB_N0, label):
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer
    man = elina_box_manager_alloc()

    # if prediction perform interval analysis
    if LB_N0[0] == UB_N0[0]:
        bounds_size, bounds = interval_propagation(nn, man, LB_N0, UB_N0,
                                                   0, numlayer)
        lbounds, ubounds = el_bounds_to_list(bounds, bounds_size)
    # else perform actual robustness analysis
    else:
        # TODO implement a strategy on how many layers to apply linear
        # programming to
        # use linear programming on two first layers (1 sec timeout)
        lbounds, ubounds = linear_solver_layerwise(nn, man, LB_N0, UB_N0,
                                                   0, 2, 1)

        # continue with interval propagation to verify robustness
        bounds_size, bounds = interval_propagation(nn, man, lbounds, ubounds,
                                                   2, numlayer)
        lbounds, ubounds = el_bounds_to_list(bounds, bounds_size)

    # if epsilon is zero, try to classify else verify robustness
    verified_flag = True
    predicted_label = 0
    if LB_N0[0] == UB_N0[0]:
        for i in range(len(lbounds)):
            inf = lbounds[i]
            flag = True
            for j in range(len(ubounds)):
                if j != i:
                   sup = ubounds[j]
                   if(inf<=sup):
                      flag = False
                      break
            if flag:
                predicted_label = i
                break
    else:
        inf = lbounds[label]
        for j in range(len(ubounds)):
            if j != label:
                sup = ubounds[j]
                if inf <= sup:
                    predicted_label = label
                    verified_flag = False
                    break

    elina_manager_free(man)
    return predicted_label, verified_flag



if __name__ == '__main__':
    from sys import argv

    # TODO remove =================================================

    # if len(argv) > 1 and argv[1] == "test":
    #     nn = layers()
    #     nn.biases = [[0.1, -0.5, 1.2],
    #                  [0, -0.2],
    #                  [0.5]]
    #     nn.weights = [[[0.2, 0.5, -1], [1.5, -0.2, 0.2], [0.7, 0.1, 0]],
    #                   [[-2, -1.5, 0.1], [0, -0.2, 3]],
    #                   [[-0.2, 0.4]]]
    #     nn.layertypes = ["ReLU", "ReLU", "ReLU"]
    #     nn.numlayer = 3

    #     lbounds = [-0.0, 0.5, 0.2]
    #     ubounds = [0.5, 0.7, 0.3]

    #     analyze(nn, lbounds, ubounds, 0)

        # TODO
    # =============================================================


    if len(argv) < 3 or len(argv) > 4:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])

    #c_label = int(argv[4])
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low,0)    # get original image

    return_code = 3
    # get actual prediction without perturbation
    label, _ = analyze(nn,LB_N0,UB_N0,0)
    start = time.time()
    if(label==int(x0_low[0])):
        # get perturbed image
        LB_N0, UB_N0 = get_perturbed_image(x0_low, epsilon)
        # get prediction on perturbed image
        _, verified_flag = analyze(nn, LB_N0, UB_N0, label)
        if(verified_flag):
            print("verified")
            return_code = 0
        else:
            print("can not be verified")
            return_code = 1
    else:
        print("image not correctly classified by the network. expected label ",
              int(x0_low[0]), " classified label: ", label)
    end = time.time()
    print("analysis time: ", (end-start), " seconds")
    exit(return_code)
