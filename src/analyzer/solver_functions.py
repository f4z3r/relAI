
def linear_solver_neuronwise(weights, biases, xi_lbounds, xi_ubounds):

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
    
    return m, z

def bounds_linear_solver_neuronwise(weights, biases, xi_lbounds, xi_ubounds):
    
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

    model, z = linear_solver_neuronwise(weights, biases, xi_lbounds, xi_ubounds)
    
    #Find upper bound of the neuron z
    model.SetObjective(z, GRB.MAXIMIZE)
    model.optimize()
    neuron_ub = z.X

    model.reset(0)

    #Find lower bound of the neuron z    
    model.SetObjective(z, GRB.MINIMIZE)
    model.optimize()
    neuron_lb = z.X

    #Applying ReLU([neuron_lb,neuron_up])
    if neuron_ub < 0:
        return 0,0
    else:
        return neuron_lb,neuron_ub
