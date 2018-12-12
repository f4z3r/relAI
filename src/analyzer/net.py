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
    def from_layers(name, layers):
        """Constructs a network from a layer definition. See the main
        analyzer.py file to see what a layer definition consists of.

        Args:
            - name: the name of the model
            - layers: the layer definition used to construct the model
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
                      layer_type)
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

    def __str__(self):
        return "Net: " + self.name +\
               "\n  layer count = " + str(len(self._layers))

    def __iter__(self):
        return iter(self._layers)
