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
import analyzer as analyzer

import solver_functions as solvers


#Extract the current bounds of the neurons zi of the all current forward layer
def extract_xi_bounds(man, element, num_in_pixels):

    lower_bounds = np.zeros(num_in_pixels)
    upper_bounds = np.zeros(num_in_pixels)
    bounds = elina_abstract0_to_box(man,element)

    for idx in range(num_in_pixels):
        lower_bounds[idx] = bounds[idx].contents.inf.contents.val.dbl
        upper_bounds[idx] = bounds[idx].contents.sup.contents.val.dbl

    return lower_bounds, upper_bounds

#Inject the new bounds to the next layer neuron zj
def inject_zj_bounds(man, element, idx_zjs, lower_bounds, upper_bounds):
    """
    Params:

    idx_zjs: array k x 1
    lower_bounds: array k x 1
    upper_bounds: array k x 1

    Return:

    element, man
    where element has updated bound for the zjs
    """

    new_bounds =  len(lower_bounds)

    for idx_bound in range(0,new_bounds):

        idx_zj = idx_zjs[idx_bound]
        l_bound = lower_bounds[idx_bound]
        u_bound = upper_bounds[idx_bound]

        #create an array of two linear constraints
        lincons0_array = elina_lincons0_array_make(2)

        #Create a greater than or equal to inequality for the lower bound
        lincons0_array.p[0].constyp = c_uint(ElinaConstyp.ELINA_CONS_SUPEQ)
        linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
        cst = pointer(linexpr0.contents.cst)

        #plug the lower bound “l_bound” here
        elina_scalar_set_double(cst.contents.val.scalar, -l_bound)
        linterm = pointer(linexpr0.contents.p.linterm[0])

        #plug the dimension “i” here
        linterm.contents.dim = ElinaDim(idx_zj)
        coeff = pointer(linterm.contents.coeff)
        elina_scalar_set_double(coeff.contents.val.scalar, 1)
        lincons0_array.p[0].linexpr0 = linexpr0

        #create a greater than or equal to inequality for the upper bound
        lincons0_array.p[1].constyp = c_uint(ElinaConstyp.ELINA_CONS_SUPEQ)
        linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
        cst = pointer(linexpr0.contents.cst)

        #plug the upper bound “u_bound” here
        elina_scalar_set_double(cst.contents.val.scalar, u_bound)
        linterm = pointer(linexpr0.contents.p.linterm[0])

        #plug the dimension “i” here
        linterm.contents.dim = ElinaDim(idx_zj)
        coeff = pointer(linterm.contents.coeff)
        elina_scalar_set_double(coeff.contents.val.scalar, -1)
        lincons0_array.p[1].linexpr0 = linexpr0

        element = elina_abstract0_meet_lincons_array(man,True,element,lincons0_array)

    return element, man

def analyze_gurobi(nn, LB_N0, UB_N0, label):
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer
    man = elina_box_manager_alloc()
    print("Number of pixels -> ",num_pixels)
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

            print("Layer number -> ",layerno)
            print("Current layer neurons -> ", len(weights[0]))
            print("Next layer number of neurons -> ", len(weights))
            print("Next layer number of biases -> ", len(biases))

            dims = elina_abstract0_dimension(man,element)
            num_in_pixels = dims.intdim + dims.realdim
            num_out_pixels = len(weights)
            dimadd = elina_dimchange_alloc(0,num_out_pixels)

            #TODO: to test and complete, first the neuronwise LP solver, then try the layerwise one
            if(layerno == 0):
            #if(False):#layerno == 0):
                zj_lbs = np.zeros(num_out_pixels)
                zj_ubs = np.zeros(num_out_pixels)
                xi_lbounds, xi_ubounds = extract_xi_bounds(man,element,num_in_pixels)
                weights_j = np.array(range(num_in_pixels))
                for j in range(num_out_pixels):
                    zj_lb, zj_ub = solvers.get_bounds_linear_solver_neuronwise(weights[j], biases[j], xi_lbounds, xi_ubounds)
                    zj_lbs[j] = zj_lb
                    zj_ubs[j] = zj_ub
                element, man = inject_zj_bounds(man = man, element = element, idx_zjs = [i for i in range(num_out_pixels)], lower_bounds=zj_lbs, upper_bounds=zj_ubs)
            
            else:

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
                    linexpr0 = analyzer.generate_linexpr0(weights[i],biases[i],num_in_pixels)
                    element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
                    var+=1

                dimrem = elina_dimchange_alloc(0,num_in_pixels)
                for i in range(num_in_pixels):
                    dimrem.contents.dim[i] = i

                elina_abstract0_remove_dimensions(man, True, element, dimrem)
                elina_dimchange_free(dimrem)

                # handle ReLU layer
                #TODO currently core dumping here after first layer went through look for elina_box.py
                print("Here right before core dumping in layer ", layerno)
                if(nn.layertypes[layerno]=='ReLU'):
                    element = relu_box_layerwise(man,True,element,0, num_out_pixels)

                nn.ffn_counter+=1

        else:
           print('net type not supported')

    dims = elina_abstract0_dimension(man,element)
    output_size = dims.intdim + dims.realdim
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)


    # if epsilon is zero, try to classify else verify robustness

    verified_flag = True
    predicted_label = 0
    if(LB_N0[0]==UB_N0[0]):
        for i in range(output_size):
            inf = bounds[i].contents.inf.contents.val.dbl
            flag = True
            for j in range(output_size):
                if(j!=i):
                   sup = bounds[j].contents.sup.contents.val.dbl
                   if(inf<=sup):
                      flag = False
                      break
            if(flag):
                predicted_label = i
                break
    else:
        inf = bounds[label].contents.inf.contents.val.dbl
        for j in range(output_size):
            if(j!=label):
                sup = bounds[j].contents.sup.contents.val.dbl
                if(inf<=sup):
                    predicted_label = label
                    verified_flag = False
                    break

    elina_interval_array_free(bounds,output_size)
    elina_abstract0_free(man,element)
    elina_manager_free(man)
    return predicted_label, verified_flag