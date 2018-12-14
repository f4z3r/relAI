#!/usr/bin/env python3

"""A module containing functions defining heuristics on entire models, layers,
and even neurons."""


def neuron_weights_out_score_ub(weights_in, weights_out, affine_bounds,
                                output_bounds, neuron_id, layer_id):
    """A heuristic scoring a neuron based on the sum of absolute values of the
    product of its outgoing weights with its maximal value.

    Args:
        The default required arguments for a neuron-wise heuristic function.

    Returns:
        A numerical score.
    """
    if weights_out is None:
        return 1
    return output_bounds[1] * sum(map(abs, weights_out))


def neuron_weights_out_score_lb(weights_in, weights_out, affine_bounds,
                                output_bounds, neuron_id, layer_id):
    """A heuristic scoring a neuron based on the sum of absolute values of the
    product of its outgoing weights with its minimal value.

    Args:
        The default required arguments for a neuron-wise heuristic function.

    Returns:
        A numerical score.
    """
    if weights_out is None:
        return 1
    return output_bounds[0] * sum(map(abs, weights_out))


def neuron_weights_out_score_diff(weights_in, weights_out, affine_bounds,
                                  output_bounds, neuron_id, layer_id):
    """A heuristic scoring a neuron based on the sum of absolute values of the
    product of its outgoing weights with the size of its output interval.

    Args:
        The default required arguments for a neuron-wise heuristic function.

    Returns:
        A numerical score.
    """
    if weights_out is None:
        return 1
    return (output_bounds[1] - output_bounds[0]) * sum(map(abs, weights_out))
