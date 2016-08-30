# test_logsumexp.py
# Test for numerically stable logsumexp (logadd) reduction along a tensor axis.

# Authors: Pascal Vincent and Alexandre de Brebisson (2016)
# License: same license as theano

import sys

import numpy as np

from theano.tensor.nnet import softmax_op
from theano.tensor.nnet.logsumexp import LogSumExp, logsumexp, softmax_npy, logsumexp_npy

from theano import tensor as T
from theano import scalar 
from theano import config, shared, printing, pp
from theano import function, function_dump 

# Test stuff
# from theano.gradient import verify_grad
import unittest
from theano.tests import unittest_tools 
from theano.tests.unittest_tools import verify_grad
from theano.tensor.tests.test_basic import makeTester


def tofloatX(x):
    return np.asarray(x, dtype=config.floatX)

# unittest_tools.seed_rng()
rng = np.random.RandomState(unittest_tools.fetch_seed())

# numpy matrix with large enough number to cause numpy.exp to explode
# but *stabilized* softmax and logsumexp should be able to handle it

if config.floatX=='float64':
    largenum_2d_input = tofloatX(rng.rand(3,5)*10+1000.)
    grad_eps = 1e-7
elif config.floatX=='float32':
    largenum_2d_input = tofloatX(rng.rand(3,5)*5)
    grad_eps = 1e-4
else:
    raise ValueError("logsumexp test with config.floatX="+config.floatX+" not verified")

    

softmax_npy_tester = makeTester(name = 'softmax_npy_tester',
                                op = softmax_op,
                                expected = lambda x: softmax_npy(x),
                                checks = {},
                                good = dict(correct1 = [tofloatX(np.random.rand(5,7))],
                                            correct2 = [largenum_2d_input]),
                                bad_build = dict(),
                                bad_runtime = dict(),
                                grad = dict(grad1 = [tofloatX(np.random.rand(5,7))],
                                            # grad2 = [largenum_2d_input]
                                            ),
                                grad_eps = grad_eps
                                )

def makeLogSumExpTester(name,axis,keepdims):
    good_inputs_list = [ ("correct_d1", [tofloatX(np.random.rand(12,))]),
                         ("correct_d2", [tofloatX(np.random.rand(3,7))]),
                         ("correct_d3", [tofloatX(np.random.rand(4,3,2))]),
                         ("correct_d4", [tofloatX(np.random.rand(2,5,3,7))]),
                         ("correct_d2_largenum", [tofloatX(largenum_2d_input)]),
                         # ("correct_d3_largenum", [tofloatX(np.random.rand(4,3,2)*10+largenum_offset)]),
                         # ("correct_d4_largenum", [tofloatX(np.random.rand(2,5,3,7)*10+largenum_offset)])
                         ]
    # filter list to contain only the shapes compatible with the axis parameter
    if axis>=0:
        good_inputs_list = [ pair for pair in good_inputs_list if axis < len(pair[1][0].shape) ]
    elif axis<0:
        good_inputs_list = [ pair for pair in good_inputs_list if len(pair[1][0].shape) + axis >=0 ]
    # print >> sys.stderr, "len(good_input_list) =" + str(len(good_inputs_list))
            
    return makeTester(name = name,
                      op = LogSumExp(axis=axis,keepdims=keepdims),
                      expected = lambda x: logsumexp_npy(x, axis=axis, keepdims=keepdims),
                      checks = { "output contains nan or inf (op not numerically stable)":
                                 (lambda inputs,outputs: np.all(np.isfinite(outputs[0])))
                                 },
                      good = dict(good_inputs_list),
                      bad_build = {"bad_d0": [np.array(1.5)]}, # op does not support scalar input
                      bad_runtime = dict(),
                      grad = dict(good_inputs_list),
                      grad_eps = grad_eps)

logsumexp_tester_0_False = makeLogSumExpTester("logsumexp_tester_0_False", axis=0, keepdims=False)
logsumexp_tester_1_False = makeLogSumExpTester("logsumexp_tester_1_False", axis=1, keepdims=False)
logsumexp_tester_2_False = makeLogSumExpTester("logsumexp_tester_2_False", axis=2, keepdims=False)
logsumexp_tester_3_False = makeLogSumExpTester("logsumexp_tester_3_False", axis=3, keepdims=False)
logsumexp_tester_minus1_False = makeLogSumExpTester("logsumexp_tester_minus1_False", axis=-1, keepdims=False)
logsumexp_tester_minus2_False = makeLogSumExpTester("logsumexp_tester_minus2_False", axis=-2, keepdims=False)
logsumexp_tester_minus3_False = makeLogSumExpTester("logsumexp_tester_minus3_False", axis=-3, keepdims=False)
logsumexp_tester_minus4_False = makeLogSumExpTester("logsumexp_tester_minus4_False", axis=-4, keepdims=False)

logsumexp_tester_0_True = makeLogSumExpTester("logsumexp_tester_0_True", axis=0, keepdims=True)
logsumexp_tester_1_True = makeLogSumExpTester("logsumexp_tester_1_True", axis=1, keepdims=True)
logsumexp_tester_2_True = makeLogSumExpTester("logsumexp_tester_2_True", axis=2, keepdims=True)
logsumexp_tester_3_True = makeLogSumExpTester("logsumexp_tester_3_True", axis=3, keepdims=True)
logsumexp_tester_minus1_True = makeLogSumExpTester("logsumexp_tester_minus1_True", axis=-1, keepdims=True)
logsumexp_tester_minus2_True = makeLogSumExpTester("logsumexp_tester_minus2_True", axis=-2, keepdims=True)
logsumexp_tester_minus3_True = makeLogSumExpTester("logsumexp_tester_minus3_True", axis=-3, keepdims=True)
logsumexp_tester_minus4_True = makeLogSumExpTester("logsumexp_tester_minus4_True", axis=-4, keepdims=True)


class TestLogSumExpStabilityAndOptimisation(unittest.TestCase):

    # The above softmax_npy_tester already verifies that the softmax_op (theano op)
    # computes the same output as softmax_npy for largenum_2d_input. So it
    # suffices to verify that exp (which naive softmax uses carelessly) is
    # unstable while softmax_npy is stable for that input to ensure that
    # softmax_op is also stable
    def test_softmax_npy_stability(self):
        """This test is only actually done for float64, because for
        float32 largenum_2d_inpout wasn't made that large because otherwise
        float32 gradient computations are too imprecise"""
        if config.floatX=='float64':
            self.assertFalse(np.all(np.isfinite(np.exp(largenum_2d_input))))
            self.assertTrue(np.all(np.isfinite(softmax_npy(largenum_2d_input))))
        # test passes cleanly

    def test_logsumexp_graph(self):
        """test_logsumexp_graph

        This test builds a small graph using logsumexp and computing its gradient,
        and verifies in its description whether the LogSumExp op was optimised away, and
        whether gradient computation introduced a Softmax or GPUSoftmax op"""

        for Htensor,axis in [ (T.matrix(),-1),
                              (T.tensor4(),-1),
                              (T.matrix(),0),
                              (T.tensor4(),1) ]:

            Hred = logsumexp(Htensor, axis=axis, keepdims=False)
            # Hred = T.log(T.sum(T.exp(Htensor_prep), axis=0))
            L = T.sum(Hred)*3.5
            Hgrad = T.grad(L, Htensor)

            Hredfunc = function([Htensor], [L,Hgrad])
            graph_description = printing.debugprint(Hredfunc, file='str')
            # print >> sys.stderr, ">>>>> Theano graph for Htensor.ndim",Htensor.ndim,"axis=",axis,":", graph_description
            self.assertTrue(graph_description is not None)

            # does the gradient computation indeed contain a Softmax or GPUSoftmax op ?
            self.assertTrue( "Softmax" in graph_description )

            # was LogSumExp op properly optimised away in the appropriate modes?
            # print >> sys.stderr, ">>>>> config.mode=" , config.mode #p.mode
            if config.mode in ['FAST_RUN','DebugMode']:
                self.assertTrue( "LogSumExp" not in graph_description )
                self.assertTrue( "+ log" in graph_description )
                self.assertTrue( "{exp(" in graph_description )
                self.assertTrue( "Reduce{maximum}" in graph_description)
            else: # other modes, check LogSumExp was not optimised away
                self.assertTrue( "LogSumExp" in graph_description )
                


def my_prepend_1_to_each_row(x):
    # theano.tensor.nnet version buggy?, in particular grad?? no cpu no gpu version? nor optimization??
    # return theano.tensor.nnet.prepend_1_to_each_row(x)
    return T.concatenate([T.ones_like(x[:,0:1]),x],axis=1)


def essai_logsumexp():

    print "*** Testing logsumexp"
    op = LogSumExp(axis=-1,keepdims=False)
    # verify_grad(op,[np.random.rand(5, 7, 2)])        

    #print "*** softmax_with_axis ***"
    #softmax_with_axis_func = function([H], [softmax_with_axis(H,1)])
    #print printing.debugprint(softmax_with_axis_func)

    # Htensor = T.ftensor4()
    # Htensor = my_prepend_1_to_each_row(T.matrix())
    Htensor = T.matrix()
    Htensor_prep = my_prepend_1_to_each_row(Htensor)
    
    print "*** losumexp and softmax optimizations ***"
    Hred = logsumexp(Htensor_prep, axis=-1, keepdims=False)
    # Hred = T.log(T.sum(T.exp(Htensor_prep), axis=0))
    L = T.sum(Hred)*3.5
    Hgrad = T.grad(L, Htensor)
    # Hred = logsumexp(H)
    Hredfunc = function([Htensor], [L,Hgrad])
    print printing.debugprint(Hredfunc)

