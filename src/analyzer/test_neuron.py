#!/usr/bin/env python3

import pytest

from neuron import *
import test_utils

"""Unit test for neuron class."""

class TestNeuron:
    def test_construction(self):
        neuron = TestNeuron.get_neuron()
        assert neuron.id == 4
        assert neuron.layer_id == 1
        assert neuron.name == "layer(1)_neuron(4)"
        assert neuron._type == "ReLU"
        assert len(neuron.weights_in) == 784
        assert len(neuron.weights_out) == 10

    def test_bound_assignment(self):
        neuron = TestNeuron.get_neuron()
        with pytest.raises(AssertionError):
            neuron.set_bounds(1, 2)
        with pytest.raises(AssertionError):
            neuron.set_upper_bound(0.2)
        with pytest.raises(AssertionError):
            neuron.set_lower_bound(0.1)

    def test_naive_bound_update(self):
        neuron = TestNeuron.get_neuron()
        neuron._set_affine_bounds(1, 4)
        bounds = neuron.get_output_bounds()
        neuron.model.update()
        assert bounds[0] == 1
        assert bounds[1] == 4
        model_var = neuron.get_output_var()
        assert model_var.lb == 1
        assert model_var.ub == 4

    @staticmethod
    def get_neuron():
        net = test_utils.load_network()
        return net._layers[1]._neurons[4]
