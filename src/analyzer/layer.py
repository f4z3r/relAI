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
        for neuron in self:
            neuron.update_bounds_naive(layer)

    def update_bounds_lp(self, layer):
        """Update the bounds of each neuron in the layer using linear
        programming.

        Args:
            - layer: the previous layer to this one in the network
        """
        for neuron in self:
            neuron.update_bounds_lp(layer)

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
               "\n  type         = " + self._type

    def __iter__(self):
        return iter(self._neurons)

    def __len__(self):
        return len(self._neurons)
