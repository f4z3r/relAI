#!/usr/bin/env python3

"""Module containing the network definition."""

from layer import Layer

from gurobipy import *


class Net:
    """Model. This contains a full neural and network and encompasses the
    linear program that can be run over said network."""
    def __init__(self, name):
        """Constructor.

        Args:
            - name: the name of the model.
        """
        self._layers = []
        self.name = name
        self.model = Model(name)
        self.model.setParam("OutputFlag", False)
        self.model.setParam("Presolve", 2)
        self.model.setParam("Timelimit", 100)

    @staticmethod
    def from_layers(name, layers, lbounds, ubounds):
        """Constructs a network from a layer definition. See the main
        analyzer.py file to see what a layer definition consists of.

        Args:
            - name: the name of the model
            - layers: the layer definition used to construct the model
            - lbounds: the lower bounds of the input layer
            - ubounds: the upper bounds of the input layer

        Returns:
            A network.
        """
        net = Net(name)

        # add the input layer
        id = 0
        layer_type = "input"
        weights_in = None
        weights_out = layers.weights[0]
        biases = None
        layer = Layer(net.model, id, weights_in, weights_out, biases,
                      layer_type, lbounds, ubounds)
        net._layers.append(layer)

        # add the hidden and output layers
        for layer_num in range(len(layers.biases)):
            id = layer_num + 1
            layer_type = layers.layertypes[layer_num]
            weights_in = layers.weights[layer_num]
            if layer_num + 1 == len(layers.biases):
                weights_out = None
            else:
                weights_out = layers.weights[layer_num + 1]
            biases = layers.biases[layer_num]
            layer = Layer(net.model, id, weights_in, weights_out, biases,
                          layer_type)
            net._layers.append(layer)
            net.model.update()

        return net

    def interval_propagation(self):
        """Perform interval propagation on the entire network.

        Returns:
            Two lists representing the lower and upper bounds of the output
            neurons respectively.
        """
        for prev_num, layer in enumerate(self.hidden_layers()):
            layer.update_bounds_naive(self._layers[prev_num])

        return self.get_output_layer_bounds()

    def linear_programming(self):
        """Perform linear programminig on the entire network.

        Returns:
            Two lists representing the lower and upper bounds of the output
            neurons respectively.
        """
        for prev_num, layer in enumerate(self.hidden_layers()):
            layer.update_bounds_lp(self._layers[prev_num])

        return self.get_output_layer_bounds()

    def partial_linear_programming(self, boundary):
        """Performs linear programming on all layers up to the boundary. Then
        performs interval propagation from the boundary until the last layer.

        Args:
            - boundary: the boundary until which to perform linear programming.

        Returns:
            Two lists representing the lower and upper bounds of the output
            neurons respectively.
        """
        assert 0 < boundary < len(self._layers),\
            "boundary must be between 0 and the layer count."
        for prev_num, layer in enumerate(self.hidden_layers()[:boundary]):
            layer.update_bounds_lp(self._layers[prev_num])
        for prev_num, layer in enumerate(self.hidden_layers()[boundary:]):
            layer.update_bounds_naive(self._layers[prev_num])

        return self.get_output_layer_bounds()

    def incomplete_linear_programming(self):
        """Perform update all contraints using interval propagation bounds and
        perform linear programming only on the very last layer.

        Returns:
            Two lists representing the lower and upper bounds of the output
            neurons respectively.
        """
        for prev_num, layer in enumerate(self.hidden_layers()[:-1]):
            layer.update_bounds_lp_lazy(self._layers[prev_num])
        self._layers[-1].update_bounds_lp(self._layers[-2])

        return self.get_output_layer_bounds()

    def neuronwise_heuristic_per_l_abs(self, func, capacity):
        """Apply a neuronwise scoring heuristic on each neuron and choose the
        best `capacity` neurons to apply linear programmming. Apply interval
        propagation on the remaining neurons.

        Args:
            - func: the neuronwise scoring heuristic.
            - capacity: the absolute number of neurons per layer to choose to
              perform linear programming on.
        """
        for prev_num, layer in enumerate(self.hidden_layers()[:-1]):
            layer.lp_score_based_absolute(func, capacity,
                                          self._layers[prev_num])
        self._layers[-1].update_bounds_lp(self._layers[-2])

        return self.get_output_layer_bounds()

    def neuronwise_heuristic_per_l_fr(self, func, fraction):
        """Apply a neuronwise scoring heuristic on each neuron and choose the
        best `capacity` neurons to apply linear programmming. Apply interval
        propagation on the remaining neurons.

        Args:
            - func: the neuronwise scoring heuristic.
            - fraction: the fraction of neurons per layer to choose to perform
              linear programming on.
        """
        for prev_num, layer in enumerate(self.hidden_layers()[:-1]):
            layer.lp_score_based_fraction(func, fraction,
                                          self._layers[prev_num])
        self._layers[-1].update_bounds_lp(self._layers[-2])

        return self.get_output_layer_bounds()

    def window_linear_programming(self, window_size):
        """Perform linear programming by propagating a window over the network
        and not keeping the model reference all the way back to the beginning
        of the network.

        Args:
            - window_size: the number of layers contained in the window. Should
              be at least 3.
        """
        # TODO implmement the function
        # To remove constraints from the model, simply relax the output
        # constraints to be between 0 and infinity.
        pass

    def hidden_layers(self):
        """Returns the list of hidden layers contained in the network. This
        function simply returns all its layers but the first.

        Returns:
            A list of layers.
        """
        return self._layers[1:]

    def get_output_layer_bounds(self):
        """Return the output bounds of the final layer of the network.

        Returns:
            Two lists representing the lower and upper bounds of the final
            layer respectively.
        """
        return self._layers[-1].get_output_bounds()

    def print_debug_info(self):
        """Prints debug information about the state of the network."""
        print(str(self))
        for layer in self:
            print(str(layer))
            for neuron in layer:
                print(str(neuron))

    def __str__(self):
        return "Net: " + self.name +\
               "\n  layer count = " + str(len(self))

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)
