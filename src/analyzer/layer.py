#!/usr/bin/env python3

"""Module containing the layer definition."""

from neuron import Neuron

class Layer:
    """Layer. Respresents a layer of neurons inside the neural network. This
    only supports fully connected layers."""
    def __init__(self, model, id, weights_in, weights_out, biases, _type,
                 lbounds=None, ubounds=None):
        """Constructor.

        Args:
            - model: the gurobi model that the layer links to
            - id: the identifier of the layer
            - weights_in: a list of list of the weights coming in to the layer
            - weights_out: a lost of list of the weights going out of the layer
            - biases: the biases of the neurons contained in this layer
            - _type: the type of the layer
            - lbounds: the lower bounds of the neurons on this layer
            - ubounds: the upper bounds of the neurons on this layer

        Note:
            Setting bounds is only allowed on input layers.

            In order to create an input layer, simply supply weights_in and
            biases as `None`. For the final output layer, simply supply
            weights_out as `None`.
        """
        if ubounds is not None or ubounds is not None:
            assert _type == "input", "manual bounds only alloed on input layer"

        self.model = model
        self.id = id
        self.name = "layer({0})".format(id)
        self._type = _type
        self._neurons = []
        self._uses_lp = "N/A"
        self._sum_gaps = None
        self._total_cap = None
        if weights_out is not None:
            weights_out = Layer.convert_weights_out(weights_out)

        neuron_count = len(weights_in) if weights_in is not None\
                                       else len(weights_out)
        for neuron_id in range(neuron_count):
            w_in = weights_in[neuron_id] if weights_in is not None else None
            w_out = weights_out[neuron_id] if weights_out is not None else None
            bias = biases[neuron_id] if biases is not None else None
            neuron = Neuron(model, neuron_id, self.id, w_in, w_out, bias,
                            self._type)
            if lbounds is not None:
                neuron.set_lower_bound(lbounds[neuron_id])
            if ubounds is not None:
                neuron.set_upper_bound(ubounds[neuron_id])

            self._neurons.append(neuron)
        self.model.update()

    def update_bounds_naive(self, layer):
        """Update the bounds of each neuron in the layer using a naive
        approach.

        Args:
            - layer: the previous layer to this one in the network
        """
        self._uses_lp = "no"
        for neuron in self:
            neuron.update_bounds_naive(layer)
        self.model.update()

    def update_bounds_lp(self, layer):
        """Update the bounds of each neuron in the layer using linear
        programming.

        Args:
            - layer: the previous layer to this one in the network
        """
        self._uses_lp = "yes"
        for neuron in self:
            neuron.update_bounds_lp(layer)
        self.model.update()

    def update_bounds_lp_subset(self, layer, subset):
        """Update the bounds of the indexed neurons in the layer using linear
        programming. The rest is updated using naive interval propagation.

        Args:
            - layer: the previous layer to this one in the network.
            - subset: a set of indexes of this layer on which to perform LP.
        """
        self._uses_lp = "mixed"
        for neuron in self:
            if neuron.id in subset:
                neuron.update_bounds_lp(layer)
            else:
                neuron.update_bounds_naive(layer)
        self.model.update()

    def update_bounds_lp_lazy(self, layer):
        """Update the bounds on the layer using interval propagation, but
        create dependencies with previous layer in order to have relational
        information in later layers.

        Args:
            - layer: the previous layer to this one in the network
        """
        self._uses_lp = "lazy"
        for neuron in self:
            neuron.update_bounds_lp_lazy(layer)
        self.model.update()

    def lp_score_based_absolute(self, func, capacity, layer):
        """Perform linear programming on an absolute number of neurons based
        on some scoring mechanism.

        Args:
            - func: the neuron-wise heuristic used to compute the score of each
              neuron.
            - capacity: the absolute number of best scoring neurons on which
              to perform linear programming.
            - layer: the previous layer to this one in the network.
        """
        self._uses_lp = "mixed"
        best = self.get_best_neurons(func, capacity)
        for neuron in self:
            if neuron.id in best:
                neuron.update_bounds_lp(layer)
            else:
                neuron.update_bounds_naive(layer)
        self.model.update()

    def lp_score_based_fraction(self, func, fraction, layer):
        """Perform linear programming on an fraction of neurons based on some
        scoring mechanism.

        Args:
            - func: the neuron-wise heuristic used to compute the score of each
              neuron.
            - fraction: the fraction of the neuron on which to apply linear
              programming.
            - layer: the previous layer to this one in the network.
        """
        capacity = int(len(self) * fraction)
        self.lp_score_based_absolute(func, capacity, layer)

    def remove_lp_constraints(self):
        """Removes all linear programming constraints from all neurons in this
        layer."""
        self._uses_lp = "yes, removed from model"
        for neuron in self:
            neuron.remove_lp_constraints()
        self.model.update()

    def get_best_neurons(self, func, capacity):
        """Returns the best `capacity` neurons' IDs based on the `func`
        function.

        Args:
            - func: the scoring function to apply to the neurons.
            - capacity: the best to select.

        Returns:
            A list of the IDs of the best values in the input list.
        """
        assert capacity <= len(self), "capacity is greater than layer length"
        scores = []
        for neuron in self:
            score = neuron.apply_scoring(func)
            scores.append((neuron.id, score))
        scores.sort(key=lambda x: x[1])
        return list(zip(*scores))[0][-capacity:]

    def high_impact_idxs(self, layer, capacity, index_list, label_lb=None):
        """Gets the neurons having a large impact on this layer's neuron values
        for the neurons indicated in the `index_list`.

        Args:
            - layer: the layer previous to this one in the network.
            - capacity: the number of highest impact neurons to pick for each
              neuron in the `index_list`.
            - index_list: the list of important neurons in this layer for which
              to check for high impact neurons.
            - label_lb: the lower bound to which to compare the neuron's
              upper bound to check how much capacity to give to each neuron. If
              `None`, the same capacity is given to each neuron.

        Returns:
            A *set* of indexes of the important neurons in the previous layer.
        """
        high_impact_neurons = set()
        for idx in index_list:
            if label_lb:
                weighted_cap = self._get_capacity(capacity, index_list, idx,
                                                  label_lb)
            else:
                weighted_cap = capacity
            neuron = self._neurons[idx]
            high_impact_neurons |= neuron.high_impact_idxs(layer, weighted_cap)
        return high_impact_neurons

    def get_output_bounds(self):
        """Returns the output bounds of this layer.

        Returns:
            Two lists representing the lower and upper bounds of the output
            neurons respectively.
        """
        lbounds = []
        ubounds = []
        for neuron in self:
            bounds = neuron.get_output_bounds()
            lbounds.append(bounds[0])
            ubounds.append(bounds[1])
        return lbounds, ubounds

    def __str__(self):
        return "Layer: " + self.name +\
               "\n  neuron count = " + str(len(self)) +\
               "\n  type         = " + self._type +\
               "\n  uses LP      = " + self._uses_lp

    def __iter__(self):
        return iter(self._neurons)

    def __len__(self):
        return len(self._neurons)

    def _get_capacity(self, capacity, index_list, idx, label_lb):
        """Get the capacity of a specific neuron based on its upper bound. The
        higher the upper bounds the more capacity the neuron will get. This
        function should only be called on the output layer!

        Args:
            - capacity: the base capacity of neurons.
            - index_list: the list of indexes of the neurons that require
              propagation.
            - idx: the actual index of the neuron for which we are computing
              the capacity.
            - label_lb: the lower bound of the label neuron.

        Returns:
            An integer capacity.
        """
        if not self._total_cap:
            self._total_cap = capacity * 10
        if not self._sum_gaps:
            self._sum_gaps = 0
            for _idx in index_list:
                ub = self._neurons[_idx].get_output_bounds()[1]
                self._sum_gaps += ub - label_lb
        true_index = index_list[idx]
        ub = self._neurons[true_index].get_output_bounds()[1]
        gap = ub - label_lb
        return capacity // 2 + int((capacity / 2) * (gap / self._sum_gaps))

    @staticmethod
    def convert_weights_out(weights_out):
        """Converts a list of lists for the next neuron inputs into another
        list of lists where the primary key reflects the neurons on the
        current layer.

        Args:
            - weights_out: a list of lists where the primary key is the
              receiver of the weights.

        Returns:
             A list of lists where the primary key is the sender of the
        weights.
        """
        return list(zip(*weights_out))
