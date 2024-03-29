#!/usr/bin/env python3

"""Module containing the neuron definition."""

from gurobipy import *

class Neuron:
    """Neuron. This can perform all the computations required on its own values
    independently of other objects.

    Note:
        Most getters from this class leak objects. This is done for
        performance. Please do not modify the leaked variables.
    """
    def __init__(self, model, neuron_id, layer_id, weights_in, weights_out,
                 bias, _type):
        """Constructor.

        Args:
            - model: the gurobi model that the neuron links to
            - neuron_id: the identifier of the neuron, should be unique per
              layer
            - layer_id: the identifier of the layer on which the neuron lies
            - weights_in: the weights coming into the neuron
            - weights_out: the weights going out of the neuron
            - bias: the bias used for this neuron
            - _type: the type of the neuron, this can be "input" or "ReLU"
        """
        self.model = model
        self._uses_lp = False
        self.layer_id = layer_id
        self.id = neuron_id
        self.name = "layer({0})_neuron({1})".format(layer_id, neuron_id)
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.bias = bias
        self._affine_le = None
        self._relu_constraints = []
        self._output_var = model.addVar(name=self.name)
        self._affine_bounds = [-GRB.INFINITY, GRB.INFINITY]
        self._output_bounds = [0, GRB.INFINITY]
        self._type = _type

    def update_bounds_naive(self, layer):
        """Updates the neuron's bounds in a naive way. This is equivalent
        than a per-neuron interval propagation.

        Args:
            - layer: the layer of neurons previous to this neuron's layer

        Note:
            This lazily updates the bounds on the output variable. In order to
            ensure model consistency, please call `model.update()` before
            using the output variable from this neuron in an optimisation step.
        """
        assert self.layer_id == layer.id + 1, "should be the previous layer"
        self._uses_lp = False
        min_val = self.bias
        max_val = self.bias
        for neuron_id, neuron in enumerate(layer):
            synapse = self.weights_in[neuron_id]
            bounds = neuron.get_output_bounds()
            if synapse >= 0:
                min_val += synapse * bounds[0]
                max_val += synapse * bounds[1]
            else:
                min_val += synapse * bounds[1]
                max_val += synapse * bounds[0]
        self._set_affine_bounds(min_val, max_val)

    def update_bounds_lp_lazy(self, layer):
        """Update the bounds on the neuron using interval propagation, but
        create dependencies with previous layer in order to have relational
        information in later layers.

        Args:
            - layer: the layer of neurons previous to this neuron's layer

        Note:
            This lazily updates the bounds on the output variable. In order to
            ensure model consistency, please call `model.update()` before
            using the output variable from this neuron in an optimisation step.
        """
        assert self.layer_id == layer.id + 1, "should be the previous layer"
        self._uses_lp = True
        # build affine sum linear expression
        self._affine_le = LinExpr(self.bias)
        for neuron_id, neuron in enumerate(layer):
            neuron_var = neuron.get_output_var()
            self._affine_le += self.weights_in[neuron_id] * neuron_var

        self.update_bounds_naive(layer)
        self._set_relu_constraints(*self._affine_bounds)

    def update_bounds_lp(self, layer):
        """Updates the neuron's bounds using linear programming.

        Args:
            - layer: the layer of neurons previous to this neuron's layer

        Note:
            This lazily updates the bounds on the output variable. In order to
            ensure model consistency, please call `model.update()` before
            using the output variable from this neuron in an optimisation step.
        """
        assert self.layer_id == layer.id + 1, "should be the previous layer"
        self._uses_lp = True
        # build affine sum linear expression
        self._affine_le = LinExpr(self.bias)
        for neuron_id, neuron in enumerate(layer):
            neuron_var = neuron.get_output_var()
            self._affine_le += self.weights_in[neuron_id] * neuron_var
        # get optima
        self.model.setObjective(self._affine_le, GRB.MINIMIZE)
        self.model.optimize()
        lb = self._affine_le.getValue()

        self.model.setObjective(self._affine_le, GRB.MAXIMIZE)
        self.model.optimize()
        ub = self._affine_le.getValue()

        self._set_affine_bounds(lb, ub)
        self._set_relu_constraints(lb, ub)

    def set_bounds(self, ubound, lbound):
        """Set the bounds for the neuron manually. This is only allowed on
        input type neurons. All other types of neurons are restricted to having
        bounds set using interval propagation or linear programming.

        Args:
            - lbound: the lower bound of the output of the neuron
            - ubound: the upper bound of the output of the neuron
        """
        assert self._type == "input", "manual bound setting only allowd on" +\
                                      "input neurons"
        self._set_affine_bounds(lbound, ubound)

    def set_lower_bound(self, lbound):
        """set the lower bound for the neuron manually. this is only allowed on
        input type neurons. all other types of neurons are restricted to having
        bounds set using interval propagation or linear programming.

        args:
            - lbound: the lower bound of the output of the neuron
        """
        assert self._type == "input", "manual bound setting only allowd on" +\
                                      "input neurons"
        self._set_affine_lb(lbound)

    def set_upper_bound(self, ubound):
        """set the upper bound for the neuron manually. this is only allowed on
        input type neurons. all other types of neurons are restricted to having
        bounds set using interval propagation or linear programming.

        args:
            - lbound: the lower bound of the output of the neuron
        """
        assert self._type == "input", "manual bound setting only allowd on" +\
                                      "input neurons"
        self._set_affine_ub(ubound)

    def get_output_bounds(self):
        """Returns the output bounds of this neurons.

        Returns:
            A list of two elements. The first being the lower bounds, the
            second being the upper bound.
        """
        return self._output_bounds

    def get_output_var(self):
        """Returns the Gurobi variable equal to the output of the neuron.

        Returns:
            A Gurobi variable.
        """
        return self._output_var

    def apply_scoring(self, func):
        """Apply some scoring mechanism based on the neuron's internal values.
        The available values are the weights coming in to the neuron, the
        weights going out, the upper and lower bounds of the affine sum and
        the upper and lower bounds of the relu of the neuron. On top of that,
        the neuron id and layer id are also passed to the scoring function.

        Args:
            - func: a scoring function. This should take the weights in, the
              weights out, the affine sum bounds, the output bounds, the
              neuron's ID, and its layer ID as arguments. Moreover it should
              return an number.

        Returns:
            A numerical score.
        """
        return func(self.weights_in, self.weights_out, self._affine_bounds,
                    self._output_bounds, self.id, self.layer_id)

    def remove_lp_constraints(self):
        """Removes the constraints set on the output variable by the ReLU if
        this was set before.

        Note:
            This is done lazily. Therefore please call `model.update()` at some
            point after calling this function for the modifications to take
            effect.
        """
        self_uses_lp = False
        self.model.remove(self._relu_constraints)

    def high_impact_idxs(self, layer, capacity):
        """Gets the neurons having a large impact on this neuron's value. This
        function actually returns the indexes of the high impact neurons.

        Args:
            - layer: the layer previous to the one of this neuron in the
              network.
            - capacity: the number of high impact neurons to recover.

        Returns:
            A *set* of indexes of the neurons in the previous layer.
        """
        scores = []
        for idx, neuron in enumerate(layer):
            synapse = self.weights_in[idx]
            bounds = neuron.get_output_bounds()
            score = synapse * (bounds[1] - bounds[0])
            scores.append((neuron.id, score))
        scores.sort(key=lambda x: x[1])
        return set(list(zip(*scores))[0][-capacity:])

    def _set_relu_constraints(self, a, b):
        """Set the contraints on how the output variable refers to the affine
        input sum.

        Args:
            - a: the affine sum lower bound
            - b: the affine sum upper bound

        Note:
            This function will panic if the affine linear expression is not
            defined.
        """
        self._relu_contraints = []
        assert self._affine_le is not None, "affine LE must be non null"
        if a > 0:
            constr = self.model.addConstr(self._output_var, GRB.EQUAL,
                                          self._affine_le)
            self._relu_constraints.append(constr)
        elif b > 0:
            constr1 = self.model.addConstr(self._output_var, GRB.GREATER_EQUAL,
                                           self._affine_le)
            rhs = LinExpr()
            rhs += (b / (b - a)) * self._affine_le
            rhs += ((b * a) / (a - b))
            constr2 = self.model.addConstr(self._output_var, GRB.LESS_EQUAL,
                                           rhs)
            self._relu_constraints += [constr1, constr2]

    def _set_affine_lb(self, a):
        """Sets the affine bounds on this neuron. This automatically udpates
        the output bounds on the neuron as well.

        Args:
            - a: the lower bound
        """
        self._affine_bounds[0] = a
        self._update_output_bounds()

    def _set_affine_ub(self, a):
        """Sets the affine bounds on this neuron. This automatically udpates
        the output bounds on the neuron as well.

        Args:
            - a: the upper bound
        """
        self._affine_bounds[1] = a
        self._update_output_bounds()

    def _set_affine_bounds(self, a, b):
        """Sets the affine bounds on this neuron. This automatically udpates
        the output bounds on the neuron as well.

        Args:
            - a: the lower bound
            - b: the upper bound
        """
        self._affine_bounds = [a, b]
        self._update_output_bounds()

    def _update_output_bounds(self):
        """Updates the output bounds based on the affine sum bounds of the
        neuron."""
        self._output_bounds = list(map(lambda x: max([0, x]),
                                       self._affine_bounds))
        self._output_var.lb = self._output_bounds[0]
        self._output_var.ub = self._output_bounds[1]

    def __str__(self):
        return "Neuron: " + self.name +\
               "\n  affine sum bounds = " + str(self._affine_bounds) +\
               "\n  output bounds     = " + str(self._output_bounds) +\
               "\n  uses linear prog  = " + str(self._uses_lp)
