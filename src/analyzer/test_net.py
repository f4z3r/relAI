#!/usr/bin/env python3

from net import Net
import test_utils

"""Unit test for the net class."""

class TestNet:
    def test_construction(self):
        layers = TestNet.build_layers()
        lbounds = [0, 0.5, 0.2]
        ubounds = [0.5, 0.7, 0.3]
        net = Net.from_layers("test_network", layers, lbounds, ubounds)
        expected_output = "Net: test_network"
        expected_output += "\n  layer count = 4"
        assert str(net) == expected_output

    def test_contraction_2(self):
        layers = test_utils.load_test_layers()
        lbounds, ubounds = test_utils.load_test_image()
        net = Net.from_layers("test_network_2", layers, lbounds, ubounds)
        expected_output = "Net: test_network_2"
        expected_output += "\n  layer count = 4"
        assert str(net) == expected_output

    def test_naive_propagation(self):
        net = TestNet.build_net()
        lbounds, ubounds = net.interval_propagation()
        assert len(lbounds) == 1
        assert len(ubounds) == 1
        assert lbounds[0] == 1.8908
        assert ubounds[0] == 2.364

    def test_linear_programming(self):
        net = TestNet.build_net()
        lbounds, ubounds = net.linear_programming()
        assert len(lbounds) == 1
        assert len(ubounds) == 1
        assert lbounds[0] >= 1.8908
        assert ubounds[0] <= 2.364

    @staticmethod
    def build_layers():
        class Layers:
            def __init__(self):
                self.layertypes = ["ReLU", "ReLU", "ReLU"]
                self.weights = [
                    [[0.2, 0.5, -1], [1.5, -0.2, 0.2], [0.7, 0.1, 0]],
                    [[-2, -1.5, 0.1], [0, -0.2, 3]],
                    [[-0.2, 0.4]]
                ]
                self.biases = [
                    [0.1, -0.5, 1.2],
                    [0, -0.2],
                    [0.5]
                ]
                self.numlayer = 3
                self.ffn_counter = 0

        return Layers()

    @staticmethod
    def build_net():
        layers = TestNet.build_layers()
        lbounds = [0, 0.5, 0.2]
        ubounds = [0.5, 0.7, 0.3]
        net = Net.from_layers("test_network", layers, lbounds, ubounds)
        return net
