#!/usr/bin/env python3

import sys
sys.path.insert(0, '../ELINA/python_interface/')
#TODO import just one time all the stuff
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

from analyzer import *

"""Module containing helper functions use to solve linear optimisations."""

def linear_solver_neuronwise(weights_xi_zj, biases_zj, xi_lbounds, xi_ubounds):

    #TODO: to be tested

    """
    Params: see bounds_linear_solver_neuronwise

    Return: gurobi LP model, objective expression
    """

    numberof_zj = weights_xi_zj.shape[0]
    numberof_xi = xi_lbounds.shape[0]

    #Create gurobipy linear solver
    m = Model("neuronwise_linear_solver")
    m.setParam("OutputFlag", False)

    zjs = []

    for j in range(numberof_zj):


        #Create variables and constraints of linear solver
        for i in range(numberof_xi):
            xi="x"+str(i)
            #xi >= lower bound && xi <= upper bound
            m.addVar(lb=xi_lbounds[i], ub=xi_ubounds[i], vtype=GRB.CONTINUOUS, name=xi )
    
        m.update()
        #z next layer neuron output
        zj = LinExpr()

        #z = sum(wi*xi)
        for i in range(numberof_xi):
            zj += weights_xi_zj[j][i] * m.getVarByName("x"+str(i))

        zj += biases_zj[j]
        zjs.append(zj)

    return m, zjs

def get_bounds_linear_solver_neuronwise(weights_xi_zj, biases_zj, xi_lbounds, xi_ubounds):

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

    numberof_zj = weights_xi_zj.shape[0]
    
    neurons_lbs = np.zeros(numberof_zj)
    neurons_ubs = np.zeros(numberof_zj)

    model, zjs = linear_solver_neuronwise(weights_xi_zj, biases_zj, xi_lbounds, xi_ubounds)

    for j in range(numberof_zj):

        #Find upper bound of the neuron z
        model.setObjective(zjs[j], GRB.MAXIMIZE)
        model.optimize()
  
        neuron_ub = zjs[j].getValue()

        model.reset(0)
    
        #Find lower bound of the neuron z
        model.setObjective(zjs[j], GRB.MINIMIZE)
        model.optimize()

        neuron_lb = zjs[j].getValue() 
    
        print("z new bounds -> [",neuron_lb,",",neuron_ub,"]")

        neurons_lbs[j] = neuron_lb
        neurons_ubs[j] = neuron_ub

    return neurons_lbs,neurons_ubs

def linear_solver_layerwise(weights_xi_zj, weights_zj_yk, biases_zj, biases_yk, xi_lbounds, xi_ubounds, zj_lbounds, zj_ubounds):

    #TODO: to test

    """
    Params: see bounds_linear_solver_layerwise

    Return: gurobi LP model, objective expression
    """

    numberof_xi = xi_lbounds.shape[0]
    numberof_zj = weights_xi_zj.shape[0]
    numberof_yk = weights_zj_yk.shape[0]

    print("Building LP gurobi model with params")
    print("Number of xi -> ",numberof_xi)
    print("Number of xi -> ", weights_xi_zj.shape[1])

    print("Number of zj -> ",numberof_zj)
    print("Number of zj -> ", weights_zj_yk.shape[1])
    print("Number of biases zj -> ", biases_zj.shape[0])

    print("Number of yk -> ",numberof_yk)
    print("Number of biases yk -> ", biases_yk.shape[0])

    
    assert len(xi_lbounds) == len(xi_ubounds), "lower bounds must be the same number as the upper bounds"

    #Create gurobipy linear solver
    m = Model("layerwise_linear_solver")
    
    #Create variables and constraints of linear solver
    for i in range(numberof_xi):
        xi="x"+str(i)
        #xi >= lower bound && xi <= upper bound
        assert xi_lbounds[i] <= xi_ubounds[i], "lower bounds must be less or equal to upper bound"
        m.addVar(lb=xi_lbounds[i], ub=xi_ubounds[i], vtype=GRB.CONTINUOUS, name=xi)

    #Update the model because of the lazy evaluation
    m.update()


    for j in range(numberof_zj):

        sum_xi_wij=LinExpr()
        
        #z_j = sum(Wij*xi) + bias
        for i in range(numberof_xi):
            sum_xi_wij += weights_xi_zj[j][i] * m.getVarByName("x"+str(i))

        sum_xi_wij+=biases_zj[j]

        if zj_ubounds[j] > 0: 
            
            #zj will be the neuron values after applying ReLU(zj)
            zj = "z"+str(j)

        
            if zj_lbounds[j]>=0:

                m.addVar(lb=zj_lbounds[j], ub=zj_ubounds[j],vtype=GRB.CONTINUOUS,name=zj)
                m.update()
                m.addConstr(m.getVarByName(zj),GRB.EQUAL,sum_xi_wij,"c1_"+str(j))

            else:

                m.addVar(lb=0, ub=zj_ubounds[j],vtype=GRB.CONTINUOUS,name=zj)
                m.update()
        
                #ReLU(z) >= z
                m.addConstr(m.getVarByName(zj),GRB.GREATER_EQUAL,sum_xi_wij,"c1_"+str(j))
        
                #ReLU(z) <= (ub_z / ub_z - lb_z) * z - (ub_z * lb_z / ub_z - lb_z)
                zj_ReLU_ub = LinExpr()
                zj_ReLU_ub+= (zj_ubounds[j]/(zj_ubounds[j]-zj_lbounds[j]))*sum_xi_wij - ((zj_ubounds[j]*zj_lbounds[j])/(zj_ubounds[j]-zj_lbounds[j]))
                m.addConstr(m.getVarByName(zj),GRB.LESS_EQUAL,zj_ReLU_ub,"c2_"+str(j))
        else:
            zj = "z"+str(j)
            m.addVar(lb=0, ub=0,vtype=GRB.CONTINUOUS,name=zj)


    m.update()

    #It will contain all the linear expression for the yk neurons
    yks = []

    for k in range(numberof_yk):

        yk = LinExpr()

        for j in range(numberof_zj):

            zj = "z"+str(j)
            yk+=weights_zj_yk[k][j]*m.getVarByName(zj)

        yk+= biases_yk[k]
        yks.append(yk)


    return m, yks

def get_bounds_linear_solver_layerwise(weights_xi_zj, weights_zj_yk, biases_zj, biases_yk, xi_lbounds, xi_ubounds, zj_lbounds, zj_ubounds):

    #TODO: to be revised and tested

    """
    Params:

    -weights_xi_zj: n x m vector   | representing the n weights coming from the X layer n neurons to the m next layer Z neurons z_j
    -biases_zj: m x 1 vector       | representing the biases to the m next layer Z neurons z_j values

    -weights_zj_yk: m x k vector | representing the m weights coming from the Z layer m neurons to the next layer Y neurons
    -biases_yk: p x 1            | representing the biases to the next layer Y neurons

    -xi_lbounds: n x 1 vector | representing the lower bounds of the previous layer neurons
    -xi_ubounds: n x 1 vector | representing the upper bounds of the previous layer neurons

    where n is the number of the X layer neurons
    where m is the number of the Y layer neurons
    where k is the number of the Z layer neurons

    Return:
    - neurons_lbs: m x 1 vector | representing the scalar values of the next layer Y neurons' lower bounds
    - neurons_ubs: m x 1 vector | representing the scalar values of the next layer Y neurons' upper bounds
    """


    numberof_zj = weights_xi_zj.shape[0]
    numberof_yk = weights_zj_yk.shape[0]

    #yks new bounds
    neurons_lbs = np.zeros(numberof_yk)
    neurons_ubs = np.zeros(numberof_yk)

    if len(zj_lbounds) == 0:
        #First compute the interval upper and lower bounds of all the zjs of the next layer Z right before the ReLU
        zj_lbounds, zj_ubounds = get_bounds_linear_solver_neuronwise(weights_xi_zj = weights_xi_zj, biases_zj = biases_zj, xi_lbounds = xi_lbounds, xi_ubounds = xi_ubounds)


    model, yks = linear_solver_layerwise(weights_xi_zj = weights_xi_zj, weights_zj_yk=weights_zj_yk, biases_zj=biases_zj, biases_yk=biases_yk, 
                                       xi_lbounds=xi_lbounds, xi_ubounds=xi_ubounds, zj_lbounds=zj_lbounds, zj_ubounds=zj_ubounds)

    for k in range(numberof_yk):
    
    
        #Find upper bound of the neuron y
        model.setObjective(yks[k], GRB.MAXIMIZE)
        model.setParam("OutputFlag", False)

        model.optimize()
 
        neuron_ub = yks[k].getValue()
     
        model.reset(0)

        #Find lower bound of the neuron z
        model.setObjective(yks[k], GRB.MINIMIZE)
        model.optimize()

        neuron_lb = yks[k].getValue()
    
        print("y new bounds -> [",neuron_lb,",",neuron_ub,"]")

        neurons_lbs[k] = neuron_lb
        neurons_ubs[k] = neuron_ub

    print(neurons_lbs)
    print(neurons_ubs)

    return model, yks, zj_lbounds, zj_ubounds, neurons_lbs, neurons_ubs


def addNextLayer(m, yks_LP, yk_lbounds, yk_ubounds, weights_zj_yk, biases_yk, layerno):

    print("Adding next layer")
    #The zj are the fresh yks computed the iteration before
    numberof_zj = weights_zj_yk.shape[1]
    #The yk are the new yks to be computed during the current iteration
    numberof_yk = weights_zj_yk.shape[0]

    print("numberof_zj -> ", numberof_zj)
    print("numberof_yk -> ", numberof_yk)

    for j in range(numberof_zj):

        zj = "z"+str(j)+"l"+str(layerno)

        if yk_ubounds[j] > 0: 
            
            #zj will be the neuron values after applying ReLU(zj)

        
            if yk_lbounds[j]>=0:

                m.addVar(lb=yk_lbounds[j], ub=yk_ubounds[j],vtype=GRB.CONTINUOUS,name=zj)
                m.update()
                m.addConstr(m.getVarByName(zj),GRB.EQUAL,yks_LP[j],"c1_"+str(j)+"l"+str(layerno))

            else:

                m.addVar(lb=0, ub=yk_ubounds[j],vtype=GRB.CONTINUOUS,name=zj)
                m.update()
        
                #ReLU(z) >= z
                m.addConstr(m.getVarByName(zj),GRB.GREATER_EQUAL,yks_LP[j],"c1_"+str(j)+"l"+str(layerno))
        
                #ReLU(z) <= (ub_z / ub_z - lb_z) * z - (ub_z * lb_z / ub_z - lb_z)
                zj_ReLU_ub = LinExpr()
                zj_ReLU_ub+= (yk_ubounds[j]/(yk_ubounds[j]-yk_lbounds[j]))*yks_LP[j] - ((yk_ubounds[j]*yk_lbounds[j])/(yk_ubounds[j]-yk_lbounds[j]))
                m.addConstr(m.getVarByName(zj),GRB.LESS_EQUAL,zj_ReLU_ub,"c2_"+str(j)+"l"+str(layerno))
        else:
            m.addVar(lb=0, ub=0,vtype=GRB.CONTINUOUS,name=zj)


    m.update()
    #It will contain all the linear expression for the next yk neurons
    yks = []

    for k in range(numberof_yk):

        yk = LinExpr()

        for j in range(numberof_zj):

            zj = "z"+str(j)+"l"+str(layerno)
            yk+=weights_zj_yk[k][j]*m.getVarByName(zj)

        yk+= biases_yk[k]
        yks.append(yk)


    return m, yks




def get_bounds_linear_solver_entire_net(model, yks_LP, yk_lbounds, yk_ubounds, weights_zj_yk, biases_yk, layerno):


    print("Starting the precise analysis")


    #The yk are the new yks to be computed during the current iteration
    numberof_yk = weights_zj_yk.shape[0]
    #new yks new bounds
    neurons_lbs = np.zeros(numberof_yk)
    neurons_ubs = np.zeros(numberof_yk)


    model, yks = addNextLayer(m=model, yks_LP=yks_LP, yk_lbounds=yk_lbounds, yk_ubounds=yk_ubounds, weights_zj_yk=weights_zj_yk, biases_yk=biases_yk, layerno = layerno) #GIVE A NAME TO THE Y IN DIFFERENT LAYER WITH THE LAYER NUMBER OTHERWISE THEY OVERWRITE EACH OTHER

    for k in range(numberof_yk):

        #Find upper bound of the neuron y
        model.setObjective(yks[k], GRB.MAXIMIZE)
        model.setParam("OutputFlag", False)

        model.optimize()
 
        neuron_ub = yks[k].getValue()
     
        model.reset(0)

        #Find lower bound of the neuron z
        model.setObjective(yks[k], GRB.MINIMIZE)
        model.optimize()

        neuron_lb = yks[k].getValue()
    
        print("y new bounds -> [",neuron_lb,",",neuron_ub,"]")

        neurons_lbs[k] = neuron_lb
        neurons_ubs[k] = neuron_ub

    print(neurons_lbs)
    print(neurons_ubs)

    return model, yks, neurons_lbs, neurons_ubs