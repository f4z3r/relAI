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

        return net

    def interval_propagation(self):
        """Perform interval propagation on the entire network.

        Returns:
            Two lists representing the lower and upper bounds of the output
            neurons respectively.
        """
        for prev_num, layer in enumerate(self.hidden_layers()):
            layer.update_bounds_naive(self._layers[prev_num])

        lbounds = []
        ubounds = []
        for neuron in self._layers[-1]:
            bounds = neuron.get_output_bounds()
            lbounds.append(bounds[0])
            ubounds.append(bounds[1])
        return lbounds, ubounds

    def hidden_layers(self):
        """Returns the list of hidden layers contained in the network. This
        function simply returns all its layers but the first.

        Returns:
            A list of layers.
        """
        return self._layers[1:]

    def print_debug_info(self):
        """Prints debug information about the state of the network."""
        print(str(self))
        for layer in self:
            print(str(layer))
            for neuron in layer:
                print(str(neuron))

    def __str__(self):
        return "Net: " + self.name +\
               "\n  layer count = " + str(len(self._layers))

    def __iter__(self):
        return iter(self._layers)
