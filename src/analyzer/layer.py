#!/usr/bin/env python3

"""Module containing the layer definition."""

from neuron import Neuron

class Layer:
    """Layer. Respresents a layer of neurons inside the neural network. This
    only supports fully connected layers."""
    def __init__(self, model, id, weights_in, weights_out, biases, _type):
        """Constructor.

        Args:
            - model: the gurobi model that the layer links to
            - id: the identifier of the layer
            - weights_in: a list of list of the weights coming in to the layer
            - weights_out: a lost of list of the weights going out of the layer
            - biases: the biases of the neurons contained in this layer
            - _type: the type of the layer

        Note:
            In order to create an input layer, simply supply weights_in and
            biases as `None`. For the final output layer, simply supply
            weights_out as `None`.
        """
        self.model = model
        self.id = id
        self.name = "layer({0})".format(id)
        self._type = _type
        self._neurons = []
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
            self._neurons.append(neuron)

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

    def update_bounds_naive(self, layer):
        """Update the bounds of each neuron in the layer using a naive
        approach.

        Args:
            - layer: the previous layer to this one in the network
        """
        # TODO: implement the naive bound update
        pass

    def update_bounds_lp(self, layer):
        """Update the bounds of each neuron in the layer using linear
        programming.

        Args:
            - layer: the previous layer to this one in the network
        """
        # TODO: implement the linear programming update
        pass

    def __str__(self):
        return "Layer: " + self.name +\
               "\n  neuron count = " + str(len(self._neurons)) +\
               "\n  type         = " + self._type

    def __iter__(self):
        return iter(self._neurons)