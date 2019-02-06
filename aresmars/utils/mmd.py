"""
Class the computes the Maximum Mean Discrepancy between two different
probability distributions by calculating the value of an estimator

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
import tensorflow as tf
from abc import ABC, abstractmethod


class GenericKernel(ABC):
    """
    Generic class for kernel, to be used in a reproducing kernel Hilbert space
    (RKHS) framework.
    """

    @staticmethod
    def _compute_squared_distances(xx: tf.Tensor,
                                   yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the matrices of the squared distances between the tensors xx
        and yy.
                squared_distances[i,j] = || x[i] - y[j] ||**2
        :param xx: input tensor;
        :param yy: input tensor, equal to xx if None;
        :return: the tensor containing the squared distances.
        """
        r_xx = tf.reduce_sum(tf.multiply(xx, xx), [1, 2])
        r_xx = tf.reshape(r_xx, [-1, 1])
        r_yy = tf.reduce_sum(tf.multiply(yy, yy), [1, 2])
        xx_vector = tf.reshape(xx, [xx.shape[0], -1])
        yy_vector = tf.reshape(yy, [yy.shape[0], -1])
        r_xy = tf.matmul(xx_vector, yy_vector, transpose_b=True)
        squared_distances = tf.add(tf.add(r_xx, r_yy),
                                   tf.scalar_mul(-2.0, r_xy))
        return squared_distances

    def update_lengthscale(self, xx: tf.Tensor, yy: tf.Tensor) -> tf.Tensor:
        """
        Updates the lengthscale of the kernel by using the median of the
        pairwise distances in the batch.
        :param xx: input tensor;
        :param yy: input tensor.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        distances = tf.sqrt(squared_distances)
        v = tf.reshape(distances, [-1])
        m = tf.cast(tf.cast(v.shape[0], dtype=tf.int32) / 2, tf.int32)
        lengthscale = tf.nn.top_k(v, m).values[m - 1]
        return lengthscale

    @abstractmethod
    def compute_covariance_matrix(self, xx: tf.Tensor,
                                  yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, returns the cross-covariance matrix between xx and
        yy.
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the cross-covariate tensor.
        """
        return tf.constant()


class LinearKernel(GenericKernel):

    def compute_covariance_matrix(self, xx: tf.Tensor,
                                  yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, returns the cross-covariance matrix between xx and
        yy according to:
                K[i,j] = x^T y
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the cross-covariate tensor.
        """
        xx_vector = tf.reshape(xx, [xx.get_shape()[0], -1])
        yy_vector = tf.reshape(yy, [yy.get_shape()[0], -1])
        cov_matrix = tf.matmul(xx_vector, yy_vector, transpose_b=True)
        return cov_matrix


class RBFKernel(GenericKernel):
    """
    Implementation of the classic Radial Basis Function kernel.
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        """
        Constructor.
        :param lengthscale: lengthscale of kernel;
        :param variance: variance of kernel.
        """
        assert lengthscale > 0,\
            "Error, negative input lengthscale in RBF kernel"
        assert variance > 0, "Error, negative input variance in RBF kernel"
        self.lengthscale = tf.constant(lengthscale, dtype=tf.float64)
        self.variance = tf.constant(variance, dtype=tf.float64)
        return

    def compute_covariance_matrix(self, xx: tf.Tensor,
                                  yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, returns the cross-covariance matrix between xx and
        yy according to:
                K[i,j] = var * exp( || x[i] - y[j] ||**2 / (2 * l**2))
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the cross-covariate tensor.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        cov_matrix = self.variance * tf.exp(- squared_distances /
                                            (2.0 * self.lengthscale))
        return cov_matrix


class RBFMixtureKernel(GenericKernel):
    """
    Implementation of the classic Radial Basis Function kernel.
    """

    def __init__(self, n_components: int = 10, base: float = 2.0):
        """
        Constructor.
        :param n_components: number of components of the kernel mixture.
        """
        self.n_components = tf.constant(n_components, dtype=tf.float64)
        self.base = tf.constant(base, dtype=tf.float64)
        exponents = tf.range(0, self.n_components, dtype=tf.float64)
        self.lengthscales = tf.reshape(self.base ** exponents,
                                       [self.n_components, 1, 1])
        return

    def compute_covariance_matrix(self, xx: tf.Tensor,
                                  yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, returns the cross-covariance matrix between xx and
        yy according to:
                K[i,j] = var * exp( || x[i] - y[j] ||**2 / (2 * l**2))
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the cross-covariate tensor.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        cov_matrix = tf.exp(- squared_distances / (2.0 * self.lengthscales))
        return tf.reduce_mean(cov_matrix, axis=0)


class RationalQuadraticKernel(RBFKernel):

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0,
                 alpha: float = 1.0):
        """
        Constructor.
        :param lengthscale: lengthscale of kernel;
        :param variance: variance of kernel.
        """
        super(RationalQuadraticKernel, self).__init__(lengthscale, variance)
        self.alpha = tf.constant(alpha, dtype=tf.float64)
        return

    def compute_covariance_matrix(self, xx: tf.Tensor,
                                  yy: tf.Tensor) -> tf.Tensor:
        """
        To be implemented, returns the cross-covariance matrix between xx and
        yy according to:
                K[i,j] = var * exp( || x[i] - y[j] ||**2 / (2 * l**2))
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the cross-covariate tensor.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        scaled_distances = self.variance *\
            tf.scalar_mul(1.0 / (2.0 * self.alpha * self.lengthscale ** 2),
                          squared_distances)
        cov_matrix = tf.pow(1.0 + scaled_distances, - self.alpha)
        return cov_matrix


class Matern52Kernel(RBFKernel):
    """
    Implementation of the Matern 5/2 kernel.
    """

    def compute_covariance_matrix(self, xx: tf.Tensor,
                                  yy: tf.Tensor = None) -> tf.Tensor:
        """
        To be implemented, returns the cross-covariance matrix between xx and
        yy according to:
                K[i,j] = var * (1 + sqrt(5) r / l + 5 * r^2 / (3 * l^2)) *
                         exp( - sqrt(5) * r / l)
        where r = || x[i] - y[j] ||.
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the cross-covariate tensor.
        """
        if yy is None:
            return self.compute_covariance_matrix(xx, xx)
        squared_distances = self._compute_squared_distances(xx, yy)
        distances = tf.sqrt(squared_distances)
        cov_matrix = self.variance * (1.0 + tf.cast(tf.sqrt(5.0),
                                                    dtype=tf.float64) *
                                      distances / self.lengthscale +
                                      5.0 * squared_distances /
                                      (3.0 * self.lengthscale ** 2)) *\
            tf.exp(- tf.cast(tf.sqrt(5.0), dtype=tf.float64) /
                   self.lengthscale * distances)
        return cov_matrix


class Matern32Kernel(RBFKernel):
    """
    Implementation of the Matern 3/2 kernel.
    """

    def compute_covariance_matrix(self, xx: tf.Tensor,
                                  yy: tf.Tensor = None) -> tf.Tensor:
        """
        To be implemented, returns the cross-covariance matrix between xx and
        yy according to:
                K[i,j] = var * (1 + sqrt(3) r / l ) *
                         exp( - sqrt(3) * r / l)
        where r = || x[i] - y[j] ||.
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the cross-covariate tensor.
        """
        if yy is None:
            return self.compute_covariance_matrix(xx, xx)
        squared_distances = self._compute_squared_distances(xx, yy)
        distances = tf.sqrt(squared_distances)
        cov_matrix = self.variance * (1.0 + tf.cast(tf.sqrt(3.0), tf.float64) /
                                      self.lengthscale * distances) *\
            tf.exp(- tf.cast(tf.sqrt(3.0), tf.float64) /
                   self.lengthscale * distances)
        return cov_matrix


class Matern12Kernel(RBFKernel):
    """
    Implementation of the Matern 1/2 kernel.
    """

    def compute_covariance_matrix(self, xx: tf.Tensor,
                                  yy: tf.Tensor = None) -> tf.Tensor:
        """
        To be implemented, returns the cross-covariance matrix between xx and
        yy according to:
                K[i,j] = var * exp(- r / l)
        where r = || x[i] - y[j] ||.
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the cross-covariate tensor.
        """
        if yy is None:
            return self.compute_covariance_matrix(xx, xx)
        squared_distances = self._compute_squared_distances(xx, yy)
        distances = tf.sqrt(squared_distances)
        scaled_distances = tf.scalar_mul(1.0 / self.lengthscale, distances)
        cov_matrix = tf.scalar_mul(self.variance, tf.exp(-scaled_distances))
        return cov_matrix


class MMD(object):
    """
    Maximum Mean Discrepancy Estimator class.
    """

    def __init__(self, kernel: GenericKernel = RBFKernel(1.0, 1.0)):
        """
        Constructor.
        :param kernel: GenericKernel type, used to model the reproducing kernel
        Hilbert space.
        """
        self.kernel = kernel
        return

    def compute_mmd_estimator(self, xx: tf.Tensor, yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the Maximum Mean Discrepancy Estimator according to:
        MMD^2_u(xx,yy) = 1 / (m * (m-1) / 2) * Sum_{i != j} k(xx_i, xx_j)
                         + 1 / (m * (m-1) / 2) * Sum_{i != j} k(yy_i, yy_j)
                         - 2 / (m * m) * Sum_{i,j} k(xx_i, yy_j)
        :param xx: tensor containing m samples drawn from distribution P;
        :param yy: tensor containing m samples drawn from distribution Q;
        :return: the value of the estimator.
        """
        self.kernel.lengthscale = self.kernel.update_lengthscale(xx, yy)
        cov_xx = self.kernel.compute_covariance_matrix(xx, xx)
        cov_yy = self.kernel.compute_covariance_matrix(yy, yy)
        cov_xy = self.kernel.compute_covariance_matrix(xx, yy)
        n_samples = tf.cast(xx.shape[0], dtype=tf.float64)
        coeff = 1.0 / (n_samples * (n_samples - 1.0))
        coeff2 = 1.0 / (n_samples * n_samples)
        term_1 = tf.scalar_mul(coeff, (tf.reduce_sum(cov_xx) -
                                       tf.reduce_sum(tf.diag_part(cov_xx))))
        term_2 = tf.scalar_mul(coeff, (tf.reduce_sum(cov_yy) -
                                       tf.reduce_sum(tf.diag_part(cov_yy))))
        term_3 = tf.scalar_mul(- 2.0 * coeff2, tf.reduce_sum(cov_xy))
        return term_1 + term_2 + term_3

    def compute_mmd_estimator_gretton(self, xx: tf.Tensor,
                                      yy: tf.Tensor) -> tf.Tensor:
        """
        Compute the Maximum Mean Discrepancy Estimator according to:
        MMD^2_u(xx,yy) = 1 / (m * (m-1) / 2) * Sum_{i != j} k(xx_i, xx_j)
                         + 1 / (m * (m-1) / 2) * Sum_{i != j} k(yy_i, yy_j)
                         - 2 / (m * (m-1) / 2) * Sum_{i != j} k(xx_i, yy_j)
        :param xx: tensor containing m samples drawn from distribution P;
        :param yy: tensor containing m samples drawn from distribution Q;
        :return: the value of the estimator.
        """
        self.kernel.lengthscale = self.kernel.update_lengthscale(xx, yy)
        cov_xx = self.kernel.compute_covariance_matrix(xx, xx)
        cov_yy = self.kernel.compute_covariance_matrix(yy, yy)
        cov_xy = self.kernel.compute_covariance_matrix(xx, yy)
        n_samples = tf.cast(xx.shape[0], dtype=tf.float64)
        coeff = 1.0 / (n_samples * (n_samples - 1.0) / 2.0)
        term_1 = tf.scalar_mul(coeff, (tf.reduce_sum(cov_xx) -
                                       tf.reduce_sum(tf.diag_part(cov_xx))))
        term_2 = tf.scalar_mul(coeff, (tf.reduce_sum(cov_yy) -
                                       tf.reduce_sum(tf.diag_part(cov_yy))))
        term_3 = tf.scalar_mul(- 2.0 * coeff, (tf.reduce_sum(cov_xy) -
                                               tf.reduce_sum(tf.diag_part(
                                                   cov_xy))))
        return term_1 + term_2 + term_3
