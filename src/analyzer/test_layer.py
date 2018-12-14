#!/usr/bin/env python3

from layer import *
import test_utils

"""Unit test for the layer class."""

class TestLayer:
    def test_construction(self):
        net = test_utils.load_network()
        for layer in net:
            assert layer.name == "layer({0})".format(layer.id)
            if layer.id == 0:
                assert layer._type == "input"
                assert len(layer) == 784
            else:
                assert layer._type == "ReLU"
                assert len(layer) == 10

    def test_lp_vs_naive_comp(self):
        net = test_utils.load_network()
        layer0 = net._layers[0]
        layer1 = net._layers[1]
        layer1.update_bounds_lp(layer0)
        lp_lbs, lp_ubs = layer1.get_output_bounds()
        net = test_utils.load_network()
        layer0 = net._layers[0]
        layer1 = net._layers[1]
        layer1.update_bounds_naive(layer0)
        el_lbs, el_ubs = layer1.get_output_bounds()
        for lp_lb, el_lb in zip(lp_lbs, el_lbs):
            assert lp_lb == el_lb
        for lp_ub, el_ub in zip(lp_ubs, el_ubs):
            assert lp_ub == el_ub

    def test_improved_precision(self):
        net = test_utils.load_network()
        layer0 = net._layers[0]
        layer1 = net._layers[1]
        layer2 = net._layers[2]
        layer1.update_bounds_lp(layer0)
        layer2.update_bounds_lp(layer1)
        lp_lbs, lp_ubs = layer2.get_output_bounds()
        net = test_utils.load_network()
        layer0 = net._layers[0]
        layer1 = net._layers[1]
        layer2 = net._layers[2]
        layer1.update_bounds_naive(layer0)
        layer2.update_bounds_naive(layer1)
        el_lbs, el_ubs = layer2.get_output_bounds()
        for lp_lb, el_lb in zip(lp_lbs, el_lbs):
            assert lp_lb >= el_lb
        for lp_ub, el_ub in zip(lp_ubs, el_ubs):
            assert lp_ub <= el_ub

    def test_weight_conversion(self):
        weights = [
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4]
        ]
        expected = [
            (1, 2, 3, 4),
            (1, 2, 3, 4),
            (1, 2, 3, 4),
            (1, 2, 3, 4),
            (1, 2, 3, 4),
            (1, 2, 3, 4)
        ]
        result = Layer.convert_weights_out(weights)
        for corr, res in zip(expected, result):
            for corr_el, res_el in zip(corr, res):
                assert corr_el == res_el
