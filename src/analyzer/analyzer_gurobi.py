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

def analyze_gurobi(nn, xi_lbounds, xi_ubounds, label):
    
    num_pixels = len(xi_lbounds)
    nn.ffn_counter = 0
    numlayer = nn.numlayer

    print("Number of image pixels -> ",num_pixels)
    print(numlayer)
    zj_lbounds=[]
    zj_ubounds=[]

    nn.ffn_counter=0
    for i in range(xi_lbounds.shape[0]):
        assert xi_lbounds[i] < xi_ubounds[i], "Image lower bounds must be the same number as the upper bounds"

    for layerno in range(numlayer-1):

        weights_zj = nn.weights[nn.ffn_counter]
        weights_yk = nn.weights[nn.ffn_counter+1]
        biases_zj = nn.biases[nn.ffn_counter]
        biases_yk = nn.biases[nn.ffn_counter+1]

        num_out_pixels = len(weights_yk)




        print("Layer number -> ",layerno)
        print("Number of lower bounds of the units of current layer -> ",xi_lbounds.shape[0])
        print("Number of upper bounds of the units of current layer -> ",xi_ubounds.shape[0])
        print("Current layer neurons -> ", len(weights_zj[0]))
        print("Next layer number of neurons -> ", len(weights_zj))
        print("Next layer number of biases -> ", len(biases_zj))
        print("Total number of layers -> ", range(numlayer))
        print("number of weights_zj to zjs -> ", len(weights_zj))
        print("number of weights_yk to yks -> ", num_out_pixels)
        print("number of biases_yk to yks -> ", len(biases_zj))
        print("Total yk to calculate the bounds of -> ",num_out_pixels)
        print("number of yks of layer -> ", nn.ffn_counter+1)


        np.ascontiguousarray(weights_zj, dtype=np.double)
        np.ascontiguousarray(biases_zj, dtype=np.double)
        np.ascontiguousarray(weights_yk, dtype=np.double)
        np.ascontiguousarray(biases_yk, dtype=np.double)
        #TODO: to test and complete, first the neuronwise LP solver, then try the layerwise one


        #yk_lbounds=np.zeros(num_out_pixels)
        #yk_ubounds=np.zeros(num_out_pixels)

        #for k in range(num_out_pixels):
             
        zj_lbounds, zj_ubounds, yk_lbounds, yk_ubounds = solvers.get_bounds_linear_solver_layerwise(weights_xi_zj = weights_zj, weights_zj_yk = weights_yk, biases_zj = biases_zj, biases_yk = biases_yk, 
                                                                                              xi_lbounds = xi_lbounds, xi_ubounds = xi_ubounds, zj_lbounds=zj_lbounds, zj_ubounds=zj_ubounds)
        #yk_lbounds[k] = yk_lb
        #yk_ubounds[k] = yk_ub
        #print("Done with unit yk -> ",k)

        xi_ubounds = zj_ubounds
        xi_lbounds = zj_lbounds

        #ReLU bounds
        xi_lbounds[xi_lbounds <0 ] = 0 
        xi_ubounds[xi_ubounds <0 ] = 0
    

        zj_lbounds = yk_lbounds
        zj_ubounds = yk_ubounds
        print(zj_lbounds)
        print(zj_ubounds)
        nn.ffn_counter+=1

    print("Final bounds")
    print(yk_lbounds)
    print(yk_ubounds)

    # if epsilon is zero, try to classify else verify robustness

    verified_flag = True
    predicted_label = 0

    inf = yk_lbounds[label]
    for j in range(len(yk_ubounds)):
        if(j!=label):
            sup = yk_ubounds[j]
            if(inf<=sup):
                predicted_label = label
                verified_flag = False
                break

    print("Verified -> ",verified_flag)
    return predicted_label, verified_flag