"""
Mean function classes to be used in the Gaussian Process regression.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod


class GPMeanFunction(ABC):
    """
    Abstract class for the mean functions in the GP.
    """

    def __init__(self, input_dim: int, use_single_gp: bool = False):
        """
        Constructor.
        :param input_dim: number of states of the SDE dynamical system;
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        """
        self.dimensionality = tf.constant(input_dim, dtype=tf.int32)
        self._initialize_variables(use_single_gp)
        return

    @abstractmethod
    def _initialize_variables(self, use_single_gp: bool = False) -> None:
        """
        Creates the attributes needed to store the TensorFlow Variables
        optimized in the log-likelihood maximization.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        """
        pass

    @abstractmethod
    def compute_mean_function(self, xx: tf.Tensor) -> tf.Tensor:
        """
        Compute the mean function for the input points passed as argument and
        return it.
        :param xx: Tensor containing the input points [n_points, 1].
        :return: Tensor containing the mean function, of size
        [1, n_dim, n_points].
        """
        pass

    @abstractmethod
    def compute_derivative_mean_function(self, xx: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the mean function with respect to xx for the
        input points passed as argument and return it.
        :param xx: Tensor containing the input points [n_points, 1].
        :return: Tensor containing the mean function, of size
        [1, n_dim, n_points].
        """
        pass


class ZeroMean(GPMeanFunction):
    """
    Zero-mean function.
    """

    def _initialize_variables(self, use_single_gp: bool = False) -> None:
        """
        Creates the attributes needed to store the TensorFlow Variables
        optimized in the log-likelihood maximization.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        """
        pass

    def compute_mean_function(self, xx: tf.Tensor) -> tf.Tensor:
        """
        Compute the mean function for the input points passed as argument and
        return it.
        :param xx: Tensor containing the input points [n_points, 1].
        :return: Tensor containing the mean function, of size
        [1, n_dim, n_points].
        """
        return tf.zeros([self.dimensionality, xx.shape[0]], dtype=tf.float64)

    def compute_derivative_mean_function(self, xx: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the mean function with respect to xx for the
        input points passed as argument and return it.
        :param xx: Tensor containing the input points [n_points, 1].
        :return: Tensor containing the mean function, of size
        [1, n_dim, n_points].
        """
        return tf.zeros([self.dimensionality, xx.shape[0]], dtype=tf.float64)


class ExpDecayMean(GPMeanFunction):
    """
    Exponentially decaying mean function, according to:
            m(x) = amplitude * exp(- rate * t) + offset.
    """

    def _initialize_variables(self, use_single_gp: bool = False) -> None:
        """
        Creates the attributes needed to store the TensorFlow Variables
        optimized in the log-likelihood maximization.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        """
        with tf.variable_scope('gaussian_process_mean'):
            if use_single_gp:
                self.amplitude = tf.Variable(1.0,
                                             dtype=tf.float64,
                                             trainable=True,
                                             name='amplitude')
                self.log_rate = tf.Variable(np.log(1.0),
                                            dtype=tf.float64,
                                            trainable=True,
                                            name='log_rate')
                self.offset = tf.Variable(0.0, dtype=tf.float64,
                                          trainable=True, name='offset')
                self.amplitudes = \
                    self.amplitude * tf.ones([self.dimensionality, 1],
                                             dtype=tf.float64)
                self.log_rates = \
                    self.log_rate * tf.ones([self.dimensionality, 1],
                                            dtype=tf.float64)
                self.offsets = \
                    self.offset * tf.ones([self.dimensionality, 1],
                                          dtype=tf.float64)
            else:
                self.amplitudes = tf.Variable(
                    tf.ones([self.dimensionality, 1], dtype=tf.float64),
                    dtype=tf.float64, trainable=True, name='amplitudes')
                self.log_rates = tf.Variable(np.log(1.0) *
                                             tf.ones([self.dimensionality, 1],
                                                     dtype=tf.float64),
                                             dtype=tf.float64,
                                             trainable=True,
                                             name='log_rates')
                self.offsets = tf.Variable(
                    tf.ones([self.dimensionality, 1], dtype=tf.float64),
                    dtype=tf.float64, trainable=True, name='offsets')
        return

    def compute_mean_function(self, xx: tf.Tensor) -> tf.Tensor:
        """
        Compute the mean function for the input points passed as argument and
        return it.
        :param xx: Tensor containing the input points [n_points, 1].
        :return: Tensor containing the mean function, of size
        [1, n_dim, n_points].
        """
        exponent = - tf.exp(self.log_rates) * tf.transpose(xx)
        mean = self.amplitudes * tf.exp(exponent) + self.offsets
        return mean

    def compute_derivative_mean_function(self, xx: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the mean function with respect to xx for the
        input points passed as argument and return it.
        :param xx: Tensor containing the input points [n_points, 1].
        :return: Tensor containing the mean function, of size
        [1, n_dim, n_points].
        """
        exponent = - tf.exp(self.log_rates) * tf.transpose(xx)
        derivative_mean = - self.amplitudes * tf.exp(exponent)\
            * tf.exp(self.log_rates)
        return derivative_mean
