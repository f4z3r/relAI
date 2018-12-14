#!/usr/bin/env python3

"""Module providing test utilities such as network loading."""

from analyzer import parse_net, parse_spec, get_perturbed_image
from net import Net


def load_test_layers():
    """Load a network as it it in the main function. This is a rather small
    network of 3x10 neurons.

    Returns:
        A `layers` object as defined in the analyzer file.
    """
    netname = "../mnist_nets/mnist_relu_3_10.txt"

    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    net = parse_net(netstring)
    return net


def load_test_image():
    """Load a test image perturned with epsilon value 0.01. This image should
    verify.

    Returns:
        A pair of lists representing the lower and upper bounds of the image
        pixel value respectively.
    """
    specname = "../mnist_images/img0.txt"
    epsilon = 0.01
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low, epsilon)
    return LB_N0, UB_N0

def load_network():
    """Loads a test network model from one of the test networks.

    Returns:
        A fully initialised `Net` object.
    """
    layers = load_test_layers()
    lbounds, ubounds = load_test_image()
    return Net.from_layers("test_model", layers, lbounds, ubounds)
