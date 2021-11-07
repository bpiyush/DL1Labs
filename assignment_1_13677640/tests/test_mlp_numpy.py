################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
from os.path import join, abspath, dirname
import unittest
import numpy as np
import torch
import torch.nn as nn
from random import randrange

import sys
sys.path.insert(0, abspath(join(dirname(__file__), '../')))

from mlp_numpy import MLP
from modules import CrossEntropyModule


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    fx = f(x)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval
        
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    
    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        
        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval
        
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


class TestMLP(unittest.TestCase):
    np.random.seed(42)
    
    def test_forward_and_backward(self):
        mlp = MLP(2, [3], 2)
        x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([1, 0, 0, 1])
        y_pred = mlp.forward(x)

        cross_entropy = CrossEntropyModule()
        loss = cross_entropy.forward(y_pred, y)
        assert loss - 0.779387907424942 < 1e-5

        assert y_pred.shape == (4, 2)
        assert (y_pred.sum(1) == 1).all()

        dout = cross_entropy.backward(x, y)
        assert np.allclose(
            dout,
            np.array(
                [[-0.0e+00, -2.5e+07],
                [-2.5e+07, -0.0e+00],
                [-2.5e-01, -0.0e+00],
                [-0.0e+00, -2.5e-01]],
            )
        )


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMLP)
    unittest.TextTestRunner(verbosity=2).run(suite)
