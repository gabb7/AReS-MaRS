"""
Implementations of trainable SDE models, the classes that contain the theta
variables optimized during training.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class TrainableModelSDE(ABC):
    """
    Abstract class of a trainable stochastic dynamical system. The parameters
    to be estimated are contained in the TensorFlow Variable self.theta and will
    be optimized during training.
    """

    def __init__(self, theta_init: np.array = None,
                 initialize_at_ground_truth: bool = False):
        """
        Constructor.
        """
        with tf.variable_scope('system_parameters'):
            self._initialize_parameter_variables(theta_init,
                                                 initialize_at_ground_truth)
        self.n_params = tf.constant(self.theta.shape[0], dtype=tf.int32)
        self.theta = tf.reshape(self.theta, shape=[self.n_params, 1])
        return

    @abstractmethod
    def _initialize_parameter_variables(
            self, theta_init: np.array = None,
            initialize_at_ground_truth: bool = False) -> None:
        """
        Initialize the TensorFlow variables containing the parameters of the
        SDE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        :param theta_init: initial value of theta;
        :param initialize_at_ground_truth: initialize theta at ground truth,
        for testing purposes.
        """
        self.theta = tf.Variable(0.0)
        return

    @abstractmethod
    def compute_gradients(self, system_states: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the SDE, meaning if returns f(X, self.theta).
        :param system_states: value of the time series observed.
        :return: TensorFlow Tensor containing the gradients.
        """
        pass


class TrainableOrnsteinUhlenbeck(TrainableModelSDE):
    """
    Trainable Ornstein-Uhlenbeck SDE 1D model. The parameters to be
    estimated are contained in the TensorFlow Variable self.theta and will
    be optimized during training.
    """

    def _initialize_parameter_variables(
            self, theta_init: np.array = None,
            initialize_at_ground_truth: bool = False) -> None:
        """
        Initialize the TensorFlow variables containing the parameters of the
        SDE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        :param theta_init: initial value of theta;
        :param initialize_at_ground_truth: initialize theta at ground truth,
        for testing purposes.
        """
        if theta_init is not None:
            self.theta = tf.Variable(theta_init,
                                     name='theta',
                                     dtype=tf.float64,
                                     trainable=True)
        elif initialize_at_ground_truth:
            self.theta = tf.Variable([0.5, 1.0],
                                     dtype=tf.float64,
                                     name='theta',
                                     trainable=True)
        else:
            self.theta = tf.Variable(tf.abs(tf.random_normal([2],
                                                             mean=0.0,
                                                             stddev=1.0,
                                                             dtype=tf.float64)),
                                     name='theta',
                                     trainable=True,
                                     constraint=lambda t: tf.clip_by_value(
                                         t, 0.001, 100.0))
        return

    def compute_gradients(self, system_states: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the SDE, meaning if returns f(X, self.theta).
        :param system_states: value of the time series observed.
        :return: TensorFlow Tensor containing the gradients.
        """
        grad1 = self.theta[0] * (self.theta[1] - system_states[:, 0:1, :])
        return grad1


class TrainableDoubleWellPotential(TrainableModelSDE):
    """
    Trainable double-well potential SDE 1D model. The parameters to be
    estimated are contained in the TensorFlow Variable self.theta and will
    be optimized during training.
    """

    def _initialize_parameter_variables(
            self, theta_init: np.array = None,
            initialize_at_ground_truth: bool = False) -> None:
        """
        Initialize the TensorFlow variables containing the parameters of the
        SDE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        :param theta_init: initial value of theta;
        :param initialize_at_ground_truth: initialize theta at ground truth,
        for testing purposes.
        """
        if theta_init is not None:
            self.theta = tf.Variable(theta_init,
                                     name='theta',
                                     dtype=tf.float64,
                                     trainable=True,
                                     constraint=lambda t: tf.clip_by_value(
                                         t, 0.001, 100.0))
        elif initialize_at_ground_truth:
            self.theta = tf.Variable([0.1, 4.0],
                                     dtype=tf.float64,
                                     name='theta',
                                     trainable=True,
                                     constraint=lambda t: tf.clip_by_value(
                                         t, 0.001, 100.0))
        else:
            self.theta = tf.Variable(tf.abs(tf.random_normal([2],
                                                             mean=0.0,
                                                             stddev=1.0,
                                                             dtype=tf.float64)),
                                     name='theta',
                                     trainable=True,
                                     constraint=lambda t: tf.clip_by_value(
                                         t, 0.001, 100.0))
        return

    def compute_gradients(self, system_states: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the SDE, meaning if returns f(X, self.theta).
        :param system_states: value of the time series observed.
        :return: TensorFlow Tensor containing the gradients.
        """
        grad1 = self.theta[0] * system_states[:, 0:1, :]\
            * (self.theta[1] - system_states[:, 0:1, :] ** 2)
        return grad1


class TrainableLotkaVolterra(TrainableModelSDE):
    """
    Trainable Lotka-Volterra model. The parameters to be
    estimated are contained in the TensorFlow Variable self.theta and will
    be optimized during training.
    """

    def _initialize_parameter_variables(
            self, theta_init: np.array = None,
            initialize_at_ground_truth: bool = False) -> None:
        """
        Initialize the TensorFlow variables containing the parameters of the
        SDE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        :param theta_init: initial value of theta;
        :param initialize_at_ground_truth: initialize theta at ground truth,
        for testing purposes.
        """
        if theta_init is not None:
            self.theta = tf.Variable(theta_init,
                                     name='theta',
                                     dtype=tf.float64,
                                     trainable=True)
        elif initialize_at_ground_truth:
            self.theta = tf.Variable([2.0, 1.0, 4.0, 1.0],
                                     name='theta',
                                     dtype=tf.float64,
                                     trainable=True)
        else:
            self.theta = tf.Variable(tf.abs(tf.random_normal([4, 1],
                                                             mean=0.0,
                                                             stddev=1.0,
                                                             dtype=tf.float64)),
                                     name='theta',
                                     trainable=True,
                                     constraint=lambda t: tf.clip_by_value(
                                         t, 0.0, 100.0))
        return

    def compute_gradients(self, system_states: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the SDE, meaning if returns f(X, self.theta).
        :param system_states: value of the time series observed.
        :return: TensorFlow Tensor containing the gradients.
        """
        grad1 = system_states[:, 0:1, :] * self.theta[0] - \
            self.theta[1] * system_states[:, 0:1, :] * system_states[:, 1:2, :]
        grad2 = - system_states[:, 1:2, :] * self.theta[2] + \
            self.theta[3] * system_states[:, 0:1, :] * system_states[:, 1:2, :]
        return tf.concat([grad1, grad2], axis=1)


class TrainableLorenz63(TrainableModelSDE):
    """
    Trainable Lorenz '63 model. The parameters to be
    estimated are contained in the TensorFlow Variable self.theta and will
    be optimized during training.
    """

    def _initialize_parameter_variables(
            self, theta_init: np.array = None,
            initialize_at_ground_truth: bool = False) -> None:
        """
        Initialize the TensorFlow variables containing the parameters of the
        SDE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        :param theta_init: initial value of theta;
        :param initialize_at_ground_truth: initialize theta at ground truth,
        for testing purposes.
        """
        if theta_init is not None:
            self.theta = tf.Variable(theta_init,
                                     name='theta',
                                     dtype=tf.float64,
                                     trainable=True)
        elif initialize_at_ground_truth:
            self.theta = tf.Variable([10.0, 28.0, 2.66667],
                                     name='theta',
                                     dtype=tf.float64,
                                     trainable=True)
        else:
            self.theta = tf.Variable(tf.abs(tf.random_normal([3, 1],
                                                             mean=0.0,
                                                             stddev=1.0,
                                                             dtype=tf.float64)),
                                     name='theta',
                                     trainable=True,
                                     constraint=lambda t: tf.clip_by_value(
                                         t, 0.0, 100.0))
        return

    def compute_gradients(self, system_states: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the SDE, meaning if returns f(X, self.theta).
        :param system_states: value of the time series observed.
        :return: TensorFlow Tensor containing the gradients.
        """
        grad1 = self.theta[0] * (system_states[:, 1:2, :]
                                 - system_states[:, 0:1, :])
        grad2 = self.theta[1] * system_states[:, 0:1, :]\
            - system_states[:, 1:2, :] - system_states[:, 0:1, :]\
            * system_states[:, 2:3, :]
        grad3 = system_states[:, 0:1, :] * system_states[:, 1:2, :] - \
            self.theta[2] * system_states[:, 2:3, :]
        return tf.concat([grad1, grad2, grad3], axis=1)
