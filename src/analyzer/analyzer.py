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

from net import Net
import heuristics

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


def elina_bounds_to_lists(bounds, bounds_size):
    lbounds = []
    ubounds = []
    for i in range(bounds_size):
        lbounds.append(bounds[i].contents.inf.contents.val.dbl)
        ubounds.append(bounds[i].contents.sup.contents.val.dbl)
    elina_interval_array_free(bounds, bounds_size)
    return lbounds, ubounds


def analyze(nn, LB_N0, UB_N0, label):
    if LB_N0[0] == UB_N0[0]:
        num_pixels = len(LB_N0)
        nn.ffn_counter = 0
        numlayer = nn.numlayer
        man = elina_box_manager_alloc()
        itv = elina_interval_array_alloc(num_pixels)
        for i in range(num_pixels):
            elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

        ## construct input abstraction
        element = elina_abstract0_of_box(man, 0, num_pixels, itv)
        elina_interval_array_free(itv,num_pixels)
        for layerno in range(numlayer):
            if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
                weights = nn.weights[nn.ffn_counter]
                biases = nn.biases[nn.ffn_counter]
                dims = elina_abstract0_dimension(man,element)
                num_in_pixels = dims.intdim + dims.realdim
                num_out_pixels = len(weights)

                dimadd = elina_dimchange_alloc(0,num_out_pixels)
                for i in range(num_out_pixels):
                    dimadd.contents.dim[i] = num_in_pixels
                elina_abstract0_add_dimensions(man, True, element, dimadd, False)
                elina_dimchange_free(dimadd)
                np.ascontiguousarray(weights, dtype=np.double)
                np.ascontiguousarray(biases, dtype=np.double)
                var = num_in_pixels
                # handle affine layer
                for i in range(num_out_pixels):
                    tdim= ElinaDim(var)
                    linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
                    element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
                    var+=1
                dimrem = elina_dimchange_alloc(0,num_in_pixels)
                for i in range(num_in_pixels):
                    dimrem.contents.dim[i] = i
                elina_abstract0_remove_dimensions(man, True, element, dimrem)
                elina_dimchange_free(dimrem)
                # handle ReLU layer
                if(nn.layertypes[layerno]=='ReLU'):
                    element = relu_box_layerwise(man,True,element,0, num_out_pixels)
                nn.ffn_counter+=1

            else:
                print(' net type not supported')

        dims = elina_abstract0_dimension(man,element)
        output_size = dims.intdim + dims.realdim
        # get bounds for each output neuron
        bounds = elina_abstract0_to_box(man,element)
        lbounds, ubounds = elina_bounds_to_lists(bounds, output_size)
        elina_abstract0_free(man,element)
        elina_manager_free(man)
    else:
        net = Net.from_layers("gurobi_net", nn, LB_N0, UB_N0)
        heuristic1 = heuristics.neuron_weights_out_score_diff
        heuristic2 = heuristics.neuron_weights_out_score_lb
        heuristic3 = heuristics.neuron_weights_out_score_ub
        # lbounds, ubounds = net.linear_programming()
        # lbounds, ubounds = net.interval_propagation()
        # lbounds, ubounds = net.interval_propagation_from(4)
        # lbounds, ubounds = net.incomplete_linear_programming()
        # lbounds, ubounds = net.incomplete_linear_programming_from(4)
        # lbounds, ubounds = net.incomplete_linear_programming_reset_from(4)
        # lbounds, ubounds = net.neuronwise_heuristic_per_l_abs(heuristic1, 20)
        # lbounds, ubounds = net.neuronwise_heuristic_per_l_abs_from(heuristic1, 20, 4)
        # lbounds, ubounds = net.neuronwise_heuristic_per_l_abs_reset_from(heuristic1, 20, 4)
        # lbounds, ubounds = net.neuronwise_heuristic_per_l_fr(heuristic2, 0.7)
        # lbounds, ubounds = net.neuronwise_heuristic_per_l_fr_from(heuristic2, 0.7, 4)
        # lbounds, ubounds = net.neuronwise_heuristic_per_l_fr_reset_from(heuristic2, 0.7, 4)
        lbounds, ubounds = net.window_linear_programming(4)

        # print some information about the network
        # for layer in net:
        #     print(str(layer))

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

    return predicted_label, verified_flag


if __name__ == '__main__':
    from sys import argv

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
