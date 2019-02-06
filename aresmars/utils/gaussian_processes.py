"""
Gaussian Process class, designed to operate on data coming from SDEs.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from aresmars.utils.kernels import *
from aresmars.utils.mean_functions import *
import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.ops.distributions.util import fill_triangular


class GaussianProcessSDE(object):
    """
    Gaussian Process class that allows sampling from the posterior and its
    derivative. The whole pipeline runs in TensorFlow in such a way that it's
    automatically differentiable.
    """

    def __init__(self,
                 input_dim: int,
                 n_points: int,
                 kernel: str = 'RBF',
                 mean_function: str = None,
                 use_single_gp: bool = False,
                 diagonal_wiener_process: bool = False,
                 single_variable_wiener_process: bool = False,
                 initial_gmat_value: float = None):
        """
        Constructor.
        :param input_dim: number of states of the SDE dynamical system;
        :param n_points: number of observation points;
        :param kernel: string indicating which kernel to use for regression.
        Valid options are 'RBF', 'Matern52', 'Matern32', 'RationalQuadratic',
        'Sigmoid';
        :param mean_function: string indicating which mean function to use for
        regression. Valid options are 'ExpDecay', if None the zero-mean function
        is used;
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting);
        :param diagonal_wiener_process: use a diagonal Wiener process (in which
        every term on the diagonal is a separate optimization variable);
        :param single_variable_wiener_process: use a diagonal Wiener process in
        which all the elements on the diagonal are equal to each other
        (e.g. variable * I);
        :param initial_gmat_value: initial value for the matrix G multiplicating
        the Wiener process.
        """
        # Save the inputs
        self.dimensionality = tf.constant(input_dim, dtype=tf.int32)
        self.n_points = tf.constant(n_points, dtype=tf.int32)
        self.n_points_int = n_points
        self.dimensionality_int = input_dim
        self.jitter = tf.constant(1e-4, dtype=tf.float64)
        # Initialize the kernel, the hyperparameters and the mean function
        self.kernel = self._initialize_kernel(input_dim,
                                              kernel,
                                              use_single_gp)
        self._initialize_variables(use_single_gp,
                                   diagonal_wiener_process,
                                   single_variable_wiener_process,
                                   initial_gmat_value)
        self._initialize_mean_function(mean_function, input_dim, use_single_gp)
        # GP Regression matrices
        self.computed_mean_function = None
        self.computed_derivative_mean_function = None
        self.c_phi_matrices = None
        self.c_phi_matrices_noiseless = None
        self.diff_c_phi_matrices = None
        self.c_phi_diff_matrices = None
        self.diff_c_phi_diff_matrices = None
        self.omega_matrix = None
        self.t_matrix = None
        return

    @staticmethod
    def _initialize_kernel(input_dim: int,
                           kernel: str = 'RBF',
                           use_single_gp: bool = False) -> GenericKernel:
        """
        Initialize the kernel of the Gaussian Process.
        :param input_dim: number of states of the SDE dynamical system;
        :param kernel: string indicating which kernel to use for regression.
        Valid options are 'RBF', 'Matern52', 'Matern32', 'RationalQuadratic',
        'Sigmoid';
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting);
        :return: the GenericKernel object.
        """
        if kernel == 'RBF':
            return RBFKernel(input_dim, use_single_gp)
        elif kernel == 'Matern52':
            return Matern52Kernel(input_dim, use_single_gp)
        elif kernel == 'Matern32':
            return Matern32Kernel(input_dim, use_single_gp)
        elif kernel == 'RationalQuadratic':
            return RationalQuadraticKernel(
                input_dim=input_dim, use_single_gp=use_single_gp)
        elif kernel == 'Sigmoid':
            return SigmoidKernel(input_dim, use_single_gp)
        else:
            sys.exit("Error: specified Gaussian Process kernel not valid")

    def _initialize_variables(self, use_single_gp: bool,
                              diagonal_wiener_process: bool = False,
                              single_variable_wiener_process: bool = False,
                              initial_gmat_value: float = None) -> None:
        """
        Initialize the variance of the log-likelihood of the GP as a TensorFlow
        variable. A logarithm-exponential transformation is used to ensure
        positivity during optimization.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting);
        :param diagonal_wiener_process: use a diagonal Wiener process (in which
        every term on the diagonal is a separate optimization variable);
        :param single_variable_wiener_process: use a diagonal Wiener process in
        which all the elements on the diagonal are equal to each other
        (e.g. variable * I);
        :param initial_gmat_value: initial value for the matrix G multiplicating
        the Wiener process.
        """
        with tf.variable_scope('gaussian_process'):
            # Log-likelihood variances
            if use_single_gp:
                self.likelihood_logvariance = tf.Variable(
                    np.log(1.0), dtype=tf.float64, trainable=True,
                    name='variance_loglik')
                self.likelihood_logvariances =\
                    self.likelihood_logvariance * tf.ones([self.dimensionality,
                                                           1, 1],
                                                          dtype=tf.float64)
            else:
                self.likelihood_logvariances = tf.Variable(
                    np.log(1.0) * tf.ones([self.dimensionality, 1, 1],
                                          dtype=tf.float64),
                    dtype=tf.float64, trainable=True,
                    name='variances_loglik')
            self.likelihood_variances = tf.exp(self.likelihood_logvariances)
            # G matrix
            if single_variable_wiener_process:
                g_vector_np = np.array([np.log(1.0)]).reshape(1, 1)
                self.g_vector = tf.Variable(g_vector_np,
                                            dtype=tf.float64,
                                            name='g_vector',
                                            trainable=True)
                self.g_matrix = tf.exp(self.g_vector)\
                    * tf.eye(self.dimensionality_int, dtype=tf.float64)
                return
            elif diagonal_wiener_process:
                n_elements_sigma = self.dimensionality_int
            else:
                n_elements_sigma = int(self.dimensionality_int
                                       * (self.dimensionality_int + 1) / 2)
            if initial_gmat_value:
                self.g_vector = tf.Variable(
                    np.log(initial_gmat_value) *
                    tf.ones([n_elements_sigma], dtype=tf.float64),
                    name='g_vector', trainable=True, dtype=tf.float64)
            else:
                self.g_vector = tf.Variable(
                    tf.random_normal([n_elements_sigma],
                                     mean=np.log(0.1),
                                     stddev=0.1,
                                     dtype=tf.float64),
                    name='g_vector', trainable=True, dtype=tf.float64)
            self.g_matrix = tf.diag(tf.exp(
                self.g_vector[0:self.dimensionality]))
            if initial_gmat_value:
                self.g_matrix = self.g_matrix \
                                * tf.eye(self.dimensionality, dtype=tf.float64)
            if not diagonal_wiener_process:
                paddings = tf.constant([[1, 0], [0, 1]])
                self.g_matrix += tf.pad(fill_triangular(
                    (self.g_vector[self.dimensionality:n_elements_sigma])),
                    paddings)
        return

    def _initialize_mean_function(self,
                                  mean_function: str,
                                  input_dim: int,
                                  use_single_gp: bool) -> None:
        """
        Initialize the mean function as specified in the argument.
        :param mean_function: string indicating which mean function to use for
        regression. Valid options are 'ExpDecay', if None the zero-mean function
        is used;
        :param input_dim: number of states of the SDE dynamical system;
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting);
        """
        if not mean_function:
            self.mean_function = ZeroMean(input_dim, use_single_gp)
        elif mean_function == "ExpDecay":
            self.mean_function = ExpDecayMean(input_dim, use_single_gp)
        return

    @staticmethod
    def _kronecker_product(mat1: tf.Tensor, mat2: tf.Tensor) -> tf.Tensor:
        """
        Compute the Kronecker product between two 2D tensors.
        :param mat1: input matrix 1;
        :param mat2: input matrix 2;
        :return: the Kronecker product between mat1 and mat2.
        """
        m1, n1 = mat1.get_shape().as_list()
        mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
        m2, n2 = mat2.get_shape().as_list()
        mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
        return tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])

    @staticmethod
    def _unroll_block_matrix(mat1: tf.Tensor) -> tf.Tensor:
        """
        Unroll the 3D tensor to a block-diagonal matrix.
        :param mat1: input matrix 1;
        :return: the corresponding unrolled block-diagonal matrix.
        """
        n_dim, m1, n1 = mat1.get_shape().as_list()
        mat1_rsh = tf.reshape(mat1, [n_dim, m1, 1, n1])
        mat2 = tf.eye(n_dim, dtype=tf.float64)
        mat2_rsh = tf.reshape(mat2, [n_dim, 1, n_dim, 1])
        return tf.reshape(mat1_rsh * mat2_rsh, [n_dim * m1, n_dim * n1])

    def build_supporting_covariance_matrices(self,
                                             t: tf.Tensor,
                                             system_std_dev: tf.Tensor,
                                             t_mean: tf.Tensor,
                                             t_std_dev: tf.Tensor) -> None:
        """
        Compute and save the covariance matrices as TensorFlow. Only the ones
        that won't change during the optimization are computed here.
        :param t: t values of the training set;
        :param system_std_dev: std dev of the system states, one for each
        dimension;
        :param t_mean: mean value of the time stamps;
        :param t_std_dev: std deviation of the time stamps;
        """
        # Mean function
        self.computed_mean_function =\
            self.mean_function.compute_mean_function(t)
        self.computed_derivative_mean_function =\
            self.mean_function.compute_derivative_mean_function(t)
        # Covariance matrices
        self.c_phi_matrices = self._build_c_phi_matrices(t)
        self.c_phi_matrices_noiseless =\
            self._build_c_phi_matrices_noiseless(t)
        self.diff_c_phi_matrices = self._build_diff_c_phi_matrices(t)
        self.c_phi_diff_matrices = self._build_c_phi_diff_matrices(t)
        self.diff_c_phi_diff_matrices = self._build_diff_c_phi_diff_matrices(t)
        # Total Covariance Matrix utilities
        self._compute_unrolled_total_c_phi()
        self._compute_unrolled_total_diff_c_phi()
        self._compute_unrolled_total_c_phi_diff()
        self._compute_unrolled_total_diff_c_phi_diff()
        self._compute_omega_matrix(t, t_mean, t_std_dev)
        self._compute_s_matrix(system_std_dev)
        self._compute_t_matrix()
        self._compute_b_matrix()
        return

    def _build_c_phi_matrices(self, t: tf.Tensor) -> tf.Tensor:
        """
        Build the covariance matrices K(x_train, x_train) + sigma_y^2 I.
        :param t: time stamps of the training set;
        :return: the tensors containing the matrices.
        """
        c_phi_matrices = self.kernel.compute_c_phi(t, t)\
            + tf.expand_dims(tf.eye(self.n_points_int, dtype=tf.float64), 0)\
            * self.likelihood_variances
        return c_phi_matrices

    def _build_c_phi_matrices_noiseless(self, t: tf.Tensor) -> tf.Tensor:
        """
        Build the covariance matrices K(x_train, x_train).
        :param t: time stamps of the training set;
        :return: the tensors containing the matrices.
        """
        c_phi_matrices = self.kernel.compute_c_phi(t, t)\
            + tf.expand_dims(tf.eye(self.n_points_int, dtype=tf.float64), 0)\
            * self.jitter
        return c_phi_matrices

    def _build_diff_c_phi_matrices(self, t: tf.Tensor) -> tf.Tensor:
        """
        Builds the matrices diff_c_phi: dK(t,t') / dt.
        :param t: time stamps of the training set;
        :return the tensor containing the matrices.
        """
        diff_c_phi_matrices = self.kernel.compute_diff_c_phi(t, t)
        return diff_c_phi_matrices

    def _build_c_phi_diff_matrices(self, t: tf.Tensor) -> tf.Tensor:
        """
        Builds the matrices diff_c_phi: dK(t,t') / dt'.
        :param t: time stamps of the training set;
        :return the tensor containing the matrices.
        """
        c_phi_diff_matrices = self.kernel.compute_c_phi_diff(t, t)
        return c_phi_diff_matrices

    def _build_diff_c_phi_diff_matrices(self, t: tf.Tensor) -> tf.Tensor:
        """
        Builds the matrices diff_c_phi_diff: d^2K(t,t') / dt dt'.
        :param t: time stamps of the training set;
        :return the tensor containing the matrices.
        """
        diff_c_phi_diff_matrices = self.kernel.compute_diff_c_phi_diff(t, t)
        return diff_c_phi_diff_matrices

    def _compute_unrolled_total_c_phi(self) -> None:
        """
        Unroll the covariance matrices C_phi given into a block diagonal
        single matrix.
        """
        self.total_c_phi = self._unroll_block_matrix(
            self.c_phi_matrices_noiseless)
        return

    def _compute_unrolled_total_diff_c_phi(self) -> None:
        """
        Unroll the covariance matrices diff_C_phi into a block diagonal single
        matrix.
        """
        self.total_diff_c_phi = \
            self._unroll_block_matrix(self.diff_c_phi_matrices)
        return

    def _compute_unrolled_total_c_phi_diff(self) -> None:
        """
        Unroll the covariance matrices C_phi_diff into a block diagonal single
        matrix.
        """
        self.total_c_phi_diff = \
            self._unroll_block_matrix(self.c_phi_diff_matrices)
        return

    def _compute_unrolled_total_diff_c_phi_diff(self) -> None:
        """
        Unroll the covariance matrices diff_C_phi_diff into a block diagonal
        single matrix.
        """
        self.total_diff_c_phi_diff = \
            self._unroll_block_matrix(self.diff_c_phi_diff_matrices)
        return

    def _compute_omega_matrix(self, t: tf.Tensor, t_mean: tf.Tensor,
                              t_std_dev: tf.Tensor) -> None:
        """
        Compute the Omega matrix which takes into account the correlation over
        time of the Ornstein Uhlenbeck process.
        :param t: t values of the training set;
        """
        t_original = t_std_dev * t + t_mean
        diff_matrix = t_original - tf.transpose(t_original)
        sum_matrix = t_original + tf.transpose(t_original)
        self.omega_k = 0.5 * tf.exp(-tf.abs(diff_matrix))\
            - 0.5 * tf.exp(- sum_matrix)
        self.omega_matrix = self._kronecker_product(
            tf.eye(self.dimensionality_int, dtype=tf.float64), self.omega_k)
        return

    def _compute_s_matrix(self, system_std_dev: tf.Tensor) -> None:
        """
        Compute the inverse of the matrix S that scales the system for
        normalization.
        :param system_std_dev: standard deviations for each state of the system.
        """
        self.s_matrix_inv = self._kronecker_product(
            tf.diag(tf.reshape(tf.ones_like(system_std_dev, dtype=tf.float64)
                               / system_std_dev, [-1])),
            tf.eye(self.n_points_int, dtype=tf.float64))
        return

    def _compute_t_matrix(self):
        """
        Compute the matrix T containing the likelihood log_variances on the
        diagonal.
        """
        self.t_matrix = self._kronecker_product(
            tf.diag(tf.reshape(self.likelihood_variances, [-1])),
            tf.eye(self.n_points_int, dtype=tf.float64))
        return

    def _compute_b_matrix(self) -> None:
        """
        Compute the B matrix, merely an expansion of the Wiener sigma.
        """
        self.b_matrix = self._kronecker_product(tf.eye(self.n_points_int,
                                                       dtype=tf.float64),
                                                self.g_matrix)
        self.b_matrix = tf.reshape(self.b_matrix,
                                   [self.n_points, self.dimensionality,
                                    self.n_points, self.dimensionality])
        self.b_matrix = tf.transpose(self.b_matrix, [1, 0, 3, 2])
        self.b_matrix = tf.reshape(self.b_matrix,
                                   [self.n_points * self.dimensionality,
                                    self.n_points * self.dimensionality])
        return

    def _compute_total_covariance_matrix_states_time(self) -> tf.Tensor:
        """
        Compute the final and complete covariance matrix that takes into account
        the correlations over time and space.
        :return: the covariance matrix.
        """
        aux_matrix = tf.matmul(self.b_matrix, tf.matmul(self.omega_matrix,
                                                        self.b_matrix,
                                                        transpose_b=True))\
            + self.t_matrix
        total_covariance_matrix = self.total_c_phi\
            + tf.matmul(self.s_matrix_inv,
                        tf.matmul(aux_matrix, self.s_matrix_inv))
        return total_covariance_matrix

    def _compute_total_covariance_matrix(self) -> tf.Tensor:
        """
        Compute the final and complete covariance matrix.
        :return: the covariance matrix.
        """
        total_covariance_matrix = self.total_c_phi\
            + tf.matmul(self.s_matrix_inv,
                        tf.matmul(self.t_matrix, self.s_matrix_inv))
        return total_covariance_matrix

    def compute_average_log_likelihood(self, system: tf.Tensor) -> tf.Tensor:
        """
        Compute the log-likelihood of the data passed as argument.
        :param system: values of the states of the system;
        :return: the tensor containing the log-likelihood.
        """
        y_vector = tf.reshape(system - self.computed_mean_function, [-1, 1])
        total_covariance_matrix =\
            self._compute_total_covariance_matrix_states_time()
        y_matrix = tf.linalg.solve(total_covariance_matrix, y_vector)
        first_term = tf.reduce_sum(y_vector * y_matrix)
        logdet_cov_matrix = tf.linalg.logdet(total_covariance_matrix)
        log_likelihood = - 0.5 * (first_term + logdet_cov_matrix)
        return tf.reduce_mean(log_likelihood) / tf.cast(self.n_points,
                                                        dtype=tf.float64)

    def compute_posterior_mean(self, system: tf.Tensor) -> tf.Tensor:
        """
        Compute the mean of GP the posterior.
        :param system: values of the states of the system;
        :return: the TensorFlow tensor with the mean.
        """
        y_vector = tf.reshape(system - self.computed_mean_function, [-1, 1])
        total_covariance_matrix =\
            self._compute_total_covariance_matrix_states_time()
        aux_matrix = tf.linalg.solve(total_covariance_matrix, y_vector)
        mu = tf.transpose(tf.matmul(self.total_c_phi, aux_matrix))
        return self.computed_mean_function\
            + tf.reshape(mu, [self.dimensionality, self.n_points])

    def compute_posterior_variance(self) -> tf.Tensor:
        """
        Compute the log_variance matrix of GP the posterior.
        :return: the TensorFlow tensor with the mean.
        """
        total_covariance_matrix =\
            self._compute_total_covariance_matrix_states_time()
        fvar = self.total_c_phi\
            - tf.matmul(self.total_c_phi,
                        tf.linalg.solve(total_covariance_matrix,
                                        self.total_c_phi))
        return fvar

    def sample_from_posterior(self,
                              system: tf.Tensor,
                              n_batch: tf.Tensor) -> tf.Tensor:
        """
        Draws batch_size different samples from the data-based GP posterior.
        :param system: values of the states of the system.
        :param n_batch: number of samples in the batch;
        :return: the TensorFlow tensor with the samples, with dimensionality:
        [self.batch_size, self.dimensionality, self.n_points]
        """
        gaussian_noise = tf.random_normal(shape=[self.dimensionality *
                                                 self.n_points,
                                                 n_batch],
                                          dtype=tf.float64)
        mu = self.compute_posterior_mean(system)
        f_var = self.compute_posterior_variance()
        chol_var = tf.cholesky(f_var)
        noise_samples = tf.matmul(chol_var, gaussian_noise)
        samples = tf.reshape(mu, [-1, 1]) + noise_samples
        return tf.reshape(tf.transpose(samples),
                          [n_batch, self.dimensionality, self.n_points])

    def sample_from_model_posterior(self, noisy_samples: tf.Tensor)\
            -> tf.Tensor:
        """
        Draws batch_size different samples from the data-based GP posterior.
        :return: the TensorFlow tensor with the samples, with dimensionality:
        [self.batch_size, self.dimensionality, self.n_points]
        """
        n_batch = noisy_samples.shape[0]
        samples = tf.transpose(tf.reshape(noisy_samples, [n_batch, -1]))
        y_vector = samples - tf.reshape(self.computed_mean_function, [-1, 1])
        total_covariance_matrix = self._compute_total_covariance_matrix()
        aux_matrix = tf.linalg.solve(total_covariance_matrix, y_vector)
        mu = tf.transpose(tf.matmul(self.total_c_phi, aux_matrix))
        gaussian_noise = tf.random_normal(shape=[self.dimensionality *
                                                 self.n_points,
                                                 n_batch],
                                          dtype=tf.float64)
        f_var = self.total_c_phi\
            - tf.matmul(self.total_c_phi,
                        tf.linalg.solve(total_covariance_matrix,
                                        self.total_c_phi))
        chol_var = tf.cholesky(f_var)
        noise = tf.transpose(tf.matmul(chol_var, gaussian_noise))
        return tf.reshape(mu + noise,
                          [n_batch, self.dimensionality, self.n_points])

    def sample_from_derivative_posterior(self, system: tf.Tensor,
                                         n_batch: tf.Tensor)\
            -> tf.Tensor:
        """
        Draws batch_size different samples from the derivative GP posterior.
        :param system: values of the states of the system.
        :param n_batch: number of samples in the batch;
        :return: the TensorFlow tensor with the samples, with dimensionality:
        [self.batch_size, self.dimensionality, self.n_points]
        """
        # Sample z
        posterior_samples = self.sample_from_posterior(system, n_batch)
        posterior_samples = tf.transpose(
            tf.reshape(posterior_samples, [n_batch, -1]))
        # Compute derivative means
        aux_matrix = tf.linalg.solve(self.total_c_phi, posterior_samples)
        samples_mean = tf.reshape(self.computed_derivative_mean_function,
                                  [-1, 1])\
            + tf.matmul(self.total_diff_c_phi, aux_matrix)
        # Add variance
        samples_var = self.total_diff_c_phi_diff\
            - tf.matmul(self.total_diff_c_phi,
                        tf.linalg.solve(self.total_c_phi,
                                        self.total_c_phi_diff))\
            + self.jitter * tf.eye(self.n_points_int * self.dimensionality_int,
                                   dtype=tf.float64)
        chol_var = tf.cholesky(samples_var)
        gaussian_noise = tf.random_normal(shape=[self.dimensionality *
                                                 self.n_points,
                                                 n_batch],
                                          dtype=tf.float64)
        derivative_posterior_samples\
            = tf.transpose(samples_mean + tf.matmul(chol_var,
                                                    gaussian_noise))
        return tf.reshape(derivative_posterior_samples,
                          [n_batch, self.dimensionality, self.n_points])
