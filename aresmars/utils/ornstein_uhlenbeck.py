"""
Implementation of the Ornstein-Uhlenbeck process. Returns noise realizations
generated by a zero-mean OU-process with variance 1.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
import tensorflow as tf


class OrnsteinUhlenbeckProcess(object):
    """
    Ornstein-Uhlenbeck process.
    """

    def __init__(self,
                 t: tf.Tensor,
                 dimensionality: tf.Tensor,
                 n_points: tf.Tensor,
                 c_phi_matrix: tf.Tensor,
                 omega_matrix: tf.Tensor,
                 b_matrix: tf.Tensor,
                 t_matrix: tf.Tensor,
                 s_inv_matrix: tf.Tensor):
        """
        Constructor.
        :param t: TensorFlow tensor containing the observation time stamps;
        :param dimensionality: TensorFlow tensor containing the number of
        states in the system;
        :param n_points: TensorFlow tensor containing the number of
        observations;
        :param c_phi_matrix: TensorFlow tensor containing the C_phi matrices
        as computed by the Gaussian process;
        :param omega_matrix: TensorFlow tensor containing the omega matrix as
        computed by the Gaussian process;
        :param b_matrix: TensorFlow tensor containing the B matrix as computed
        by the Gaussian process;
        :param t_matrix: TensorFlow tensor containing the T matrix as computed
        by the Gaussian process;
        :param s_inv_matrix: TensorFlow tensor containing the S^{-1} matrix as
        computed by the Gaussian process.
        """
        self.t = t
        self.dimensionality = dimensionality
        self.n_points = n_points
        self._build_covariance_matrix(omega_matrix, c_phi_matrix, b_matrix,
                                      t_matrix, s_inv_matrix)
        return

    def _build_covariance_matrix(self,
                                 omega_matrix: tf.Tensor,
                                 c_phi_matrix: tf.Tensor,
                                 b_matrix: tf. Tensor,
                                 t_matrix: tf.Tensor,
                                 s_inv_matrix: tf.Tensor) -> None:
        """
        Build the OU covariance matrix (and its Cholesky decomposition) which
        we'll use to sample the OU process.
        :param omega_matrix: TensorFlow tensor containing the omega matrix as
        computed by the Gaussian process;
        :param c_phi_matrix: TensorFlow tensor containing the C_phi matrices
        as computed by the Gaussian process;
        :param b_matrix: TensorFlow tensor containing the B matrix as computed
        by the Gaussian process;
        :param t_matrix: TensorFlow tensor containing the T matrix as computed
        by the Gaussian process;
        :param s_inv_matrix: TensorFlow tensor containing the S^{-1} matrix as
        computed by the Gaussian process.
        """
        self.omega_tilde = tf.matmul(tf.matmul(s_inv_matrix,
                                     tf.matmul(b_matrix,
                                               tf.matmul(omega_matrix, b_matrix,
                                                         transpose_b=True))),
                                     s_inv_matrix)
        t_tilde = tf.matmul(tf.matmul(s_inv_matrix, t_matrix), s_inv_matrix)
        self.aux_matrix = self.omega_tilde + c_phi_matrix + t_tilde
        cov_matrix = self.omega_tilde\
            - tf.matmul(self.omega_tilde, tf.linalg.solve(self.aux_matrix,
                                                          self.omega_tilde))\
            + 1e-4 * tf.diag(tf.ones(omega_matrix.shape[0],
                                     dtype=tf.float64))
        self.cov_matrix_chol = tf.cholesky(cov_matrix)
        return

    def _compute_mean(self, system: tf.Tensor) -> tf.Tensor:
        """
        Compute the mean function of the Ornstein - Uhlenbeck process
        conditioned on y.
        :param system: y values;
        :return: mean tensor.
        """
        y_vector = tf.reshape(system, [-1, 1])
        mu = tf.matmul(self.omega_tilde, tf.linalg.solve(self.aux_matrix,
                                                         y_vector))
        return tf.reshape(tf.transpose(mu), [1, self.dimensionality,
                                             self.n_points])

    def sample_noise_batch(self,
                           system: tf.Tensor,
                           batch_size: tf.Tensor) -> tf.Tensor:
        """
        Returns a batch of OU noise processes.
        :param system: y values;
        :param batch_size: number of elements in the batch;
        :return: the tensor containing the batch, with shape
        [batch_size, dim, n_points_int].
        """
        gaussian_noise = tf.random_normal(shape=[self.dimensionality
                                                 * self.n_points,
                                                 batch_size], dtype=tf.float64)
        noise_batch = tf.matmul(self.cov_matrix_chol, gaussian_noise)
        noise_batch = tf.reshape(tf.transpose(noise_batch),
                                 [batch_size, self.dimensionality,
                                  self.n_points])
        mu = self._compute_mean(system)
        return mu + noise_batch