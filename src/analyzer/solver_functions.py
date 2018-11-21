
#TODO: to complete
def linear_solver_layerwise(weights, biases, l_bounds, u_bounds, neurons_next_l):


    #Create gurobipy linear solver
    m = Model("layerwise_linear_solver")
    n_bounds = l_bounds.shape()[0]

    #Create variables and constraints of linear solver
    for i in range(n_bounds):
        
        x_i="x"+str(i)
        #xi >= lower bound && xi <= upper bound
        m.addVar(lb=l_bounds[i], ub=u_bounds[i], vtype=GRB.CONTINUOUS, name=x_i ) 

    #sum of all z_i objective
    z_i_sum=LinExpr()

    for i in range(neurons_next_l):

        zi="z"+str(i)
        
        m.addVar(vtype=GRB.CONTINUOUS,name=z_i)

        zi_expr=LinExpr()
        #z_i = sum(Wij*xi) --> for both lower and upper bounds
        for j in range(n_bounds):
            zi_expr += weights[i][j] * m.getVarByName("x"+str(j)) 
        
        m.addConstr(m.getVarByName(zi),GRB.EQUAL,zi_expr,"c"+i)
        #TODO: decide what to optiize as the objective
    
    return m, obj

def linear_solver_neuronwise(weights, bias, xi_lbounds, xi_ubounds):

    """
    Params: see bounds_linear_solver_neuronwise
    
    Return: linear model, objective expression
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
    
    """
    Params:
    -weights: m x 1 vector
    -biases: m x 1 vector
    -xi_lbounds: n x 1 vector
    -xi_lbounds: n x 1 vector
    
    Return:
    Lower Bound, Upper Bound (of z)
    where z is the value of the neuron in the next layer
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
