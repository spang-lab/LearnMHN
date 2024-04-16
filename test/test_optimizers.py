from __future__ import annotations

import unittest
import numpy as np
import mhn
from mhn.optimizers import _Optimizer, StateSpaceOptimizer, DUAOptimizer, OmegaOptimizer
from mhn.original import ModelConstruction, Likelihood, UtilityFunctions


class BaseOptimizerTestClass:
    class TestOptimizer(unittest.TestCase):

        DEFAULT_NUMBER_OF_EVENTS = 5

        def setUp(self) -> None:
            np.random.seed(0)
            self.opt: _Optimizer | None = None

        def test_init_stays_same(self):
            """
            Make sure that the train method does not modify the init value for training.
            """
            n = BaseOptimizerTestClass.TestOptimizer.DEFAULT_NUMBER_OF_EVENTS
            random_init = np.random.random((n, n))
            self.opt.set_init_theta(random_init)
            self.opt.train(maxit=2)
            np.testing.assert_array_equal(random_init, self.opt._init_theta)

        def test_set_callback(self):
            """
            Test if callback functions are called.
            """
            def some_callback_function(theta: np.ndarray):
                some_callback_function.callback_function_called = True

            some_callback_function.callback_function_called = False

            self.opt.set_callback_func(some_callback_function)
            self.opt.train(maxit=2)
            self.assertEqual(some_callback_function.callback_function_called, True)

        def test_set_device(self):
            """
            Test if set_device acts as expected.
            """
            if mhn.cuda_available() != mhn.CUDA_AVAILABLE:
                self.skipTest("CUDA is not available on this device, so skip the set_device() test.")

            for device in self.opt.Device:
                # should work and not throw an error
                self.opt.set_device(device)

            # if the value given to set_device() is not part of the Device Enum, it should raise a ValueError
            self.assertRaises(ValueError, lambda: self.opt.set_device("GPU"))

        def test_set_penalty(self):
            """
            Test if set_penalty acts as expected.
            """
            for penalty in self.opt.Penalty:
                self.opt.set_penalty(penalty)

            # if the value given to set_penalty() is not part of the Penalty Enum, it should raise a ValueError
            self.assertRaises(ValueError, lambda: self.opt.set_penalty("L1"))


class TestStateSpaceOptimizer(BaseOptimizerTestClass.TestOptimizer):

    def setUp(self) -> None:
        super().setUp()
        dummy_data = np.random.choice([0, 1], (20, BaseOptimizerTestClass.TestOptimizer.DEFAULT_NUMBER_OF_EVENTS))
        self.opt = StateSpaceOptimizer()
        self.opt.load_data_matrix(dummy_data)

    def test_learn_model(self):
        """
        Simply tests if learning a new model works with no errors.
        """
        random_model = self._get_random_model(event_num=5)
        random_model = np.around(random_model, decimals=2)
        mhn_object = self.opt._OutputMHNClass(random_model)
        random_data = mhn_object.sample_artificial_data(200)
        self.opt.load_data_matrix(random_data)
        self.opt.train()

    def test_find_lambda(self):
        """
        Tests if cross-validation works with no errors.
        """
        nfolds = 2
        steps = 3
        lambda_min = 0.05
        lambda_max = 0.5
        lambda_vector = np.array([0.09, 0.1])

        # test with lambda_min/max
        np.random.seed(0)
        best_lambda = self.opt.find_lambda(lambda_min, lambda_max, steps, nfolds)
        np.random.seed(0)
        best_lambda2, df = self.opt.find_lambda(lambda_min, lambda_max, steps, nfolds, return_lambda_scores=True)
        # test reproducibility
        self.assertEqual(best_lambda, best_lambda2)

        # test with lambda_vector, also test that steps parameter is ignored
        np.random.seed(0)
        best_lambda = self.opt.find_lambda(lambda_vector=lambda_vector, steps=1, nfolds=nfolds)
        np.random.seed(0)
        best_lambda2, df = self.opt.find_lambda(lambda_vector=lambda_vector, steps=1, nfolds=nfolds,
                                                return_lambda_scores=True)
        # test reproducibility
        self.assertEqual(best_lambda, best_lambda2)

    @staticmethod
    def _get_random_model(event_num: int) -> np.ndarray:
        """
        Helper method to create a random MHN.
        """
        return ModelConstruction.random_theta(event_num, sparsity=0.3)


class TestOmegaOptimizer(TestStateSpaceOptimizer):

    def setUp(self) -> None:
        super().setUp()
        dummy_data = np.random.choice([0, 1], (20, BaseOptimizerTestClass.TestOptimizer.DEFAULT_NUMBER_OF_EVENTS))
        self.opt = OmegaOptimizer()
        self.opt.load_data_matrix(dummy_data)

    def test_init_stays_same(self):
        """
        Make sure that the train method does not modify the init value for training.
        """
        n = BaseOptimizerTestClass.TestOptimizer.DEFAULT_NUMBER_OF_EVENTS
        random_init = np.random.random((n + 1, n))
        self.opt.set_init_theta(random_init)
        self.opt.train(maxit=2)
        np.testing.assert_array_equal(random_init, self.opt._init_theta)

    @staticmethod
    def _get_random_model(event_num: int) -> np.ndarray:
        """
        Helper method to create a random MHN.
        """
        classical_theta = TestStateSpaceOptimizer._get_random_model(event_num)
        return np.vstack((classical_theta, np.random.random((1, event_num))))


if __name__ == '__main__':
    unittest.main()
