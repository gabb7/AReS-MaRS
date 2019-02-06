"""
Collection of kernel classes to be used in the Gaussian Process Regression.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod


class GenericKernel(ABC):
    """
    Generic class for a Gaussian Process kernel.
    """

    def __init__(self, input_dim: int, use_single_gp: bool = False):
        """
        Constructor.
        :param input_dim: number of states.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        """
        self.dimensionality = tf.constant(input_dim, dtype=tf.int32)
        self._initialize_variables(use_single_gp)
        return

    def _initialize_variables(self, use_single_gp: bool = False) -> None:
        """
        Initialize the hyperparameters of the kernel as TensorFlow variables.
        A logarithm-exponential transformation is used to ensure positivity
        during optimization.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        """
        with tf.variable_scope('gaussian_process_kernel'):
            if use_single_gp:
                self.log_lengthscale = tf.Variable(np.log(1.0),
                                                   dtype=tf.float64,
                                                   trainable=True,
                                                   name='log_lengthscale')
                self.log_variance = tf.Variable(np.log(1.0),
                                                dtype=tf.float64,
                                                trainable=True,
                                                name='log_variance')
                self.lengthscales = \
                    tf.exp(self.log_lengthscale)\
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
                self.variances = \
                    tf.exp(self.log_variance)\
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
            else:
                self.log_lengthscales = tf.Variable(
                    np.log(1.0) * tf.ones([self.dimensionality, 1, 1],
                                          dtype=tf.float64),
                    dtype=tf.float64, trainable=True, name='log_lengthscales')
                self.log_variances = tf.Variable(
                    tf.ones([self.dimensionality, 1, 1],
                            dtype=tf.float64),
                    dtype=tf.float64, trainable=True, name='log_variances')
                self.variances = tf.exp(self.log_variances)
                self.lengthscales = tf.exp(self.log_lengthscales)
        return

    @staticmethod
    def _compute_squared_distances(xx: tf.Tensor,
                                   yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the matrices of the squared distances between the tensors xx
        and yy.
                    squared_distances[0, i, j] = || x[i] - y[j] ||**2
        The shape of the returned tensor is [1, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the squared distances.
        """
        r_xx = xx * xx
        r_xx = tf.reshape(r_xx, [-1, 1])
        r_yy = yy * yy
        r_yy = tf.reshape(r_yy, [1, -1])
        r_xy = tf.matmul(tf.reshape(xx, [-1, 1]), tf.reshape(yy, [1, -1]))
        squared_distances = r_xx - 2.0 * r_xy + r_yy
        return tf.expand_dims(squared_distances, 0)

    @staticmethod
    def _compute_absolute_distances(xx: tf.Tensor,
                                    yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the matrices of the absolute distances between the tensors xx
        and yy.
                    distances[0, i, j] = | x[i] - y[j] |
        The shape of the returned tensor is [1, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the squared distances.
        """
        distances = tf.reshape(xx, [-1, 1]) - tf.reshape(yy, [1, -1])
        return tf.expand_dims(tf.abs(distances), 0)

    @abstractmethod
    def compute_c_phi(self, xx: tf.Tensor,
                      yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the kernel covariance matrix between xx and
        yy for each state:
                    c_phi[n_s, i, j] = kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        return tf.constant()

    @abstractmethod
    def compute_diff_c_phi(self, xx: tf.Tensor,
                           yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to xx, for each state:
                    diff_c_phi[n_s, i, j] = d/dxx kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: tensor containing the derivatives of the covariance matrices
        w.r.t. xx.
        """
        return tf.constant()

    def compute_c_phi_diff(self, xx: tf.Tensor,
                           yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to xx, for each state:
                    c_phi_diff[n_s, i, j] = d/dyy kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        Note: for stationary kernels this is just the negative diff_c_phi
        matrix. Non-stationary kernels should override this method too.
        :param xx: input tensor;
        :param yy: input tensor;
        :return: tensor containing the derivatives of the covariance matrices
        w.r.t. yy.
        """
        return - self.compute_diff_c_phi(xx, yy)

    @abstractmethod
    def compute_diff_c_phi_diff(self, xx: tf.Tensor,
                                yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, compute the cross terms of the derivative of the
        kernel covariance matrix between xx and yy, for each state:
                    diff_c_phi_diff[n_s, i, j] =
                        d^2/(dxx dyy) kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the derivative of cross-covariate tensor w.r.t. x2.
        """
        return tf.constant()


class RBFKernel(GenericKernel):
    """
    Implementation of the Radial Basis Function kernel.
    """

    def compute_c_phi(self, xx: tf.Tensor,
                      yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the kernel covariance matrix between xx and yy for each state:
                    c_phi[n_s, i, j] =
                        {var * exp( || x[i] - y[j] ||**2 / (2 * l**2))}_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        scaled_distances = - squared_distances /\
            tf.pow(self.lengthscales, 2.0) * 0.5
        cov_matrix = self.variances * tf.exp(scaled_distances)
        return cov_matrix

    def compute_diff_c_phi(self, xx: tf.Tensor,
                           yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the kernel covariance matrix between xx and
        yy with respect to xx, for each state:
                    diff_c_phi[n_s, i, j] = d/dxx kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: tensor containing the derivatives of the covariance matrices
        w.r.t. xx.
        """
        cov_matrix = self.compute_c_phi(xx, yy)
        distances = tf.expand_dims(xx - tf.transpose(yy), 0)
        return - distances / tf.pow(self.lengthscales, 2.0) * cov_matrix

    def compute_diff_c_phi_diff(self, xx: tf.Tensor,
                                yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the cross terms of the derivative of the kernel covariance
        matrix between xx and yy, for each state:
                    diff_c_phi_diff[n_s, i, j] =
                        d^2/(dxx dyy) kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the derivative of cross-covariate tensor w.r.t. x2.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        cov_matrix = self.compute_c_phi(xx, yy)
        d2k_dxdy = cov_matrix * (tf.pow(self.lengthscales, -2.0) -
                                 squared_distances /
                                 tf.pow(self.lengthscales, 4.0))
        return d2k_dxdy


class Matern52Kernel(RBFKernel):
    """
    Implementation of the Matern 5/2 kernel.
    """

    def compute_c_phi(self, xx: tf.Tensor,
                      yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the kernel covariance matrix between xx and yy for each state:
                    c_phi[n_s, i, j] =
                        {var * (1 + sqrt(5) / l * |xx[i] - yy[j]| +
                                5 * || x[i] - y[j] ||**2 / (3 * l^2)) *
                            exp(- sqrt(5) / l * |xx[i] - yy[j]|)}_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        distances = self._compute_absolute_distances(xx, yy)
        cov_matrix = self.variances * (1.0 + tf.cast(tf.sqrt(5.0),
                                                     dtype=tf.float64) *
                                       distances / self.lengthscales +
                                       5.0 * squared_distances /
                                       (3.0 * self.lengthscales ** 2)) *\
            tf.exp(- tf.cast(tf.sqrt(5.0), dtype=tf.float64) /
                   self.lengthscales * distances)
        return cov_matrix

    def compute_diff_c_phi(self, xx: tf.Tensor,
                           yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the kernel covariance matrix between xx and
        yy with respect to xx, for each state:
                    diff_c_phi[n_s, i, j] = d/dxx kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: tensor containing the derivatives of the covariance matrices
        w.r.t. xx.
        """
        distances = self._compute_absolute_distances(xx, yy)
        dk_dr = - self.variances * (5.0 * tf.cast(tf.sqrt(5.0),
                                                  tf.float64) / 3.0
                                    / tf.pow(self.lengthscales, 3) * distances
                                    + 5.0 / 3.0 / tf.pow(self.lengthscales,
                                                         2)) *\
            tf.exp(- tf.cast(tf.sqrt(5.0), dtype=tf.float64)
                   / self.lengthscales * distances)
        dk_dx1 = dk_dr * tf.expand_dims(xx - tf.transpose(yy), 0)
        return dk_dx1

    def compute_diff_c_phi_diff(self, xx: tf.Tensor,
                                yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the cross terms of the derivative of the kernel covariance
        matrix between xx and yy, for each state:
                    diff_c_phi_diff[n_s, i, j] =
                        d^2/(dxx dyy) kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the derivative of cross-covariate tensor w.r.t. x2.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        distances = tf.sqrt(squared_distances)
        d2k_dxdy = self.variances\
            * (5.0 / (3.0 * tf.pow(self.lengthscales, 2)) +
               5.0 * tf.cast(tf.sqrt(5.0), tf.float64)
               / (3.0 * tf.pow(self.lengthscales, 3.0)) * distances -
               25.0 / (3.0 * tf.pow(self.lengthscales, 4.0))
               * squared_distances) * tf.exp(- tf.cast(tf.sqrt(5.0),
                                                       dtype=tf.float64)
                                             / self.lengthscales * distances)
        return d2k_dxdy


class Matern32Kernel(RBFKernel):
    """
    Implementation of the Matern 3/2 kernel.
    """

    def compute_c_phi(self, xx: tf.Tensor,
                      yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the kernel covariance matrix between xx and yy for each state:
                    c_phi[n_s, i, j] =
                        {var * (1 + sqrt(3) / l * |xx[i] - yy[j]|) *
                            exp(- sqrt(3) / l * |xx[i] - yy[j]|)}_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        distances = self._compute_absolute_distances(xx, yy)
        cov_matrix = self.variances * (1.0 + tf.cast(tf.sqrt(3.0), tf.float64) /
                                       self.lengthscales * distances) *\
            tf.exp(- tf.cast(tf.sqrt(3.0), tf.float64) /
                   self.lengthscales * distances)
        return cov_matrix

    def compute_diff_c_phi(self, xx: tf.Tensor,
                           yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the kernel covariance matrix between xx and
        yy with respect to xx, for each state:
                    diff_c_phi[n_s, i, j] = d/dxx kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: tensor containing the derivatives of the covariance matrices
        w.r.t. xx.
        """
        distances = self._compute_absolute_distances(xx, yy)
        dk_dx1 = - self.variances * 3.0 / self.lengthscales ** 2 * \
            tf.exp(- tf.cast(tf.sqrt(3.0), tf.float64) /
                   self.lengthscales * distances)\
            * tf.expand_dims(xx - tf.transpose(yy), 0)
        return dk_dx1

    def compute_diff_c_phi_diff(self, xx: tf.Tensor,
                                yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the cross terms of the derivative of the kernel covariance
        matrix between xx and yy, for each state:
                    diff_c_phi_diff[n_s, i, j] =
                        d^2/(dxx dyy) kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the derivative of cross-covariate tensor w.r.t. x2.
        """
        distances = self._compute_absolute_distances(xx, yy)
        d2k_dx1dx2 = self.variances * 3.0 *\
            (1.0 / tf.pow(self.lengthscales, 2)
             - tf.cast(tf.sqrt(3.0), tf.float64)
             / tf.pow(self.lengthscales, 3) * distances) *\
            tf.exp(- tf.cast(tf.sqrt(3.0), tf.float64) /
                   self.lengthscales * distances)
        return d2k_dx1dx2


class RationalQuadraticKernel(GenericKernel):

    def __init__(self, input_dim: int, use_single_gp: bool = False,
                 alpha: float = 1.0):
        """
        Constructor.
        :param input_dim: number of states.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting);
        :param alpha: alpha parameter in the rational quadratic kernel.
        """
        super(RationalQuadraticKernel,
              self).__init__(input_dim, use_single_gp)
        self.alpha = tf.constant(alpha, dtype=tf.float64)
        return

    def compute_c_phi(self, xx: tf.Tensor,
                      yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the kernel covariance matrix between xx and yy for each state:
                    c_phi[n_s, i, j] = {var *
                        (1 + || x[i] - y[j] ||**2 /
                              (2 * alpha * l^2))^-alpha}_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        scaled_distances = squared_distances\
            / (2.0 * self.alpha * self.lengthscales ** 2)
        cov_matrix = tf.pow(scaled_distances + 1.0, - self.alpha)
        return self.variances * cov_matrix

    def compute_diff_c_phi(self, xx: tf.Tensor,
                           yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the kernel covariance matrix between xx and
        yy with respect to xx, for each state:
                    diff_c_phi[n_s, i, j] = d/dxx kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: tensor containing the derivatives of the covariance matrices
        w.r.t. xx.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        dk_dx1 = - self.variances * tf.pow((1.0 + squared_distances /
                                            (2.0 * self.alpha *
                                             tf.pow(self.lengthscales, 2))),
                                           - self.alpha - 1.0) *\
            tf.expand_dims(xx - tf.transpose(yy), 0)\
            / tf.pow(self.lengthscales, 2)
        return dk_dx1

    def compute_diff_c_phi_diff(self, xx: tf.Tensor,
                                yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the cross terms of the derivative of the kernel covariance
        matrix between xx and yy, for each state:
                    diff_c_phi_diff[n_s, i, j] =
                        d^2/(dxx dyy) kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the derivative of cross-covariate tensor w.r.t. x2.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        term1 = self.variances / tf.pow(self.lengthscales, 2) *\
            tf.pow((1.0 + squared_distances / (2.0 * self.alpha *
                                               tf.pow(self.lengthscales, 2))),
                   - self.alpha - 1.0)
        term2 = - self.variances * (1.0 + self.alpha) * squared_distances /\
            (self.alpha * tf.pow(self.lengthscales, 4)) *\
            tf.pow(1.0 + squared_distances /
                   (2.0 * self.alpha * tf.pow(self.lengthscales, 2)),
                   - self.alpha - 2.0)
        return term1 + term2


class SigmoidKernel(GenericKernel):

    def _initialize_variables(self, use_single_gp: bool = False) -> None:
        """
        Initialize the hyperparameters of the kernel as TensorFlow variables.
        A logarithm-exponential transformation is used to ensure positivity
        during optimization.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        """
        with tf.variable_scope('gaussian_process_kernel'):
            if use_single_gp:
                self.log_a_single = tf.Variable(np.log(1.0), dtype=tf.float64,
                                                trainable=True,
                                                name='sigmoid_a')
                self.log_b_single = tf.Variable(np.log(1.0), dtype=tf.float64,
                                                trainable=True,
                                                name='sigmoid_b')
                self.log_variance = tf.Variable(np.log(1.0), dtype=tf.float64,
                                                trainable=True,
                                                name='log_variance')
                self.a = \
                    tf.exp(self.log_a_single)\
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
                self.b = \
                    tf.exp(self.log_b_single)\
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
                self.variances = \
                    tf.exp(self.log_variance)\
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
            else:
                self.log_a = tf.Variable(np.log(1.0) *
                                         tf.ones([self.dimensionality, 1, 1],
                                                 dtype=tf.float64),
                                         dtype=tf.float64,
                                         trainable=True,
                                         name='sigmoid_a')
                self.log_b = tf.Variable(np.log(1.0) *
                                         tf.ones([self.dimensionality, 1, 1],
                                                 dtype=tf.float64),
                                         dtype=tf.float64,
                                         trainable=True,
                                         name='sigmoid_b')
                self.log_variances = tf.Variable(
                    np.log(1.0) * tf.ones([self.dimensionality, 1, 1],
                                          dtype=tf.float64),
                    dtype=tf.float64, trainable=True, name='variances')
                self.a = tf.exp(self.log_a)
                self.b = tf.exp(self.log_b)
                self.variances = tf.exp(self.log_variances)
        return

    def compute_c_phi(self, xx: tf.Tensor,
                      yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the kernel covariance matrix between xx and yy for each state:
                    c_phi[n_s, i, j] =
                        {var * asin((a + b * xx[i] * yy[j]) /
                            sqrt((a + b * xx[i]^2 + 1)
                                 * (a + b * yy[j]^2 + 1)))}_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        x_matrix = tf.expand_dims(tf.tile(xx, [1, xx.shape[0]]), 0)
        y_matrix = tf.expand_dims(tf.transpose(tf.tile(yy,
                                                       [1, yy.shape[0]])), 0)
        cov_matrix = tf.asin((self.a + self.b * x_matrix * y_matrix)
                             / tf.sqrt((self.a + self.b * x_matrix ** 2 + 1.0)
                                       * (self.a + self.b * y_matrix ** 2
                                          + 1.0)))
        return self.variances * cov_matrix

    def compute_diff_c_phi(self, xx: tf.Tensor,
                           yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the kernel covariance matrix between xx and
        yy with respect to xx, for each state:
                    diff_c_phi[n_s, i, j] = d/dxx kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: tensor containing the derivatives of the covariance matrices
        w.r.t. xx.
        """
        x_matrix = tf.expand_dims(tf.tile(xx, [1, xx.shape[0]]), 0)
        y_matrix = tf.expand_dims(tf.transpose(tf.tile(yy,
                                                       [1, yy.shape[0]])), 0)
        sqrt_term_num = tf.sqrt((self.a + self.b * x_matrix ** 2 + 1.0)
                                * (self.a + self.b * y_matrix ** 2 + 1.0))
        num_term2 = - x_matrix * (self.a + self.b * x_matrix * y_matrix)\
            + y_matrix * (self.a + self.b * x_matrix ** 2 + 1.0)
        numerator = self.b * sqrt_term_num * num_term2
        sqrt_term_den = tf.sqrt(1.0
                                - (self.a + self.b * x_matrix * y_matrix) ** 2
                                / ((self.a + self.b * x_matrix ** 2 + 1.0)
                                   * (self.a + self.b * y_matrix ** 2 + 1.0)))
        denominator = sqrt_term_den *\
            (self.a + self.b * x_matrix ** 2 + 1.0) ** 2 \
            * (self.a + self.b * y_matrix ** 2 + 1.0)
        return self.variances * numerator / denominator

    def compute_c_phi_diff(self, xx: tf.Tensor,
                           yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the kernel covariance matrix between xx and
        yy with respect to xx, for each state:
                    c_phi_diff[n_s, i, j] = d/dyy kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        Note: Non-stationary kernel!
        :param xx: input tensor;
        :param yy: input tensor;
        :return: tensor containing the derivatives of the covariance matrices
        w.r.t. yy.
        """
        x_matrix = tf.expand_dims(tf.tile(xx, [1, xx.shape[0]]), 0)
        y_matrix = tf.expand_dims(tf.transpose(tf.tile(yy,
                                                       [1, yy.shape[0]])), 0)
        sqrt_term_num = tf.sqrt((self.a + self.b * x_matrix ** 2 + 1.0)
                                * (self.a + self.b * y_matrix ** 2 + 1.0))
        num_term2 = x_matrix * (self.a + self.b * y_matrix ** 2 + 1.0)\
            - y_matrix * (self.a + self.b * x_matrix * y_matrix)
        numerator = self.b * sqrt_term_num * num_term2
        sqrt_term_den = tf.sqrt(1.0
                                - (self.a + self.b * x_matrix * y_matrix) ** 2
                                / ((self.a + self.b * x_matrix ** 2 + 1.0)
                                   * (self.a + self.b * y_matrix ** 2 + 1.0)))
        denominator = sqrt_term_den * (self.a + self.b * x_matrix ** 2 + 1.0) \
            * (self.a + self.b * y_matrix ** 2 + 1.0) ** 2
        return self.variances * numerator / denominator

    def compute_diff_c_phi_diff(self, xx: tf.Tensor,
                                yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the cross terms of the derivative of the kernel covariance
        matrix between xx and yy, for each state:
                    diff_c_phi_diff[n_s, i, j] =
                        d^2/(dxx dyy) kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the derivative of cross-covariate tensor w.r.t. x2.
        """
        x_matrix = tf.expand_dims(tf.tile(xx, [1, xx.shape[0]]), 0)
        y_matrix = tf.expand_dims(tf.transpose(tf.tile(yy,
                                                       [1, yy.shape[0]])), 0)
        numerator = self.b * (2.0 * self.a + 1.0)
        den_term1 = tf.sqrt((self.a + self.b * x_matrix ** 2 + 1.0)
                            * (self.a + self.b * y_matrix ** 2 + 1.0))
        den_term2 = tf.sqrt((self.a * self.b * x_matrix ** 2
                             - 2.0 * self.a * self.b * x_matrix * y_matrix
                             + self.a * self.b * y_matrix ** 2 + 2.0 * self.a
                             + self.b * x_matrix ** 2
                             + self.b * y_matrix ** 2 + 1.0) /
                            (self.a ** 2 + self.a * self.b * x_matrix ** 2
                             + self.a * self.b * y_matrix ** 2 + 2.0 * self.a
                             + (self.b * x_matrix * y_matrix) ** 2
                             + self.b * x_matrix ** 2
                             + self.b * y_matrix ** 2 + 1.0))
        den_term3 = self.a * self.b * x_matrix ** 2\
            - 2.0 * self.a * self.b * x_matrix * y_matrix\
            + self.a * self.b * y_matrix ** 2 + 2.0 * self.a\
            + self.b * x_matrix ** 2\
            + self.b * y_matrix ** 2 + 1.0
        denominator = den_term1 * den_term2 * den_term3
        dk2_dxdy = self.variances * numerator / denominator
        return dk2_dxdy
