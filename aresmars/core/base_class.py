"""
Base class for the MARS and ARES classes, with common utilities/methods and
attributes.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from aresmars.utils.trainable_models import TrainableModelSDE
from aresmars.utils.gaussian_processes import GaussianProcessSDE
from aresmars.utils.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess
from aresmars.utils.tensorflow_optimizer import ExtendedScipyOptimizerInterface
import tensorflow as tf
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt


class AresMarsBaseClass(object):
    """
    Base class that encapsulates some of the methods/fields needed by both MMD
    and WGAN-based regression algorithms for SDEs.
    """

    def __init__(self,
                 trainable: TrainableModelSDE,
                 system_data: np.array,
                 t_data: np.array,
                 gp_kernel: str = 'RBF',
                 gp_mean: str = None,
                 initial_gmat_value: float = None,
                 batch_size: int = 128,
                 optimizer: str = 'Adam',
                 starter_learning_rate: float = 1e-2,
                 state_normalization: bool = True,
                 time_normalization: bool = True,
                 diagonal_wiener_process: bool = False,
                 single_variable_wiener_process: bool = False,
                 use_single_gp: bool = False):
        """
        Constructor.
        :param trainable: TrainableModelSDE, as explained and implemented in
        utils.trainable_models;
        :param system_data: numpy array containing the noisy observations of
        the state values of the system, size is [n_states, n_points];
        :param t_data: numpy array containing the time stamps corresponding to
        the observations passed as system_data;
        :param gp_kernel: string indicating which kernel to use in the GP.
        Valid options are 'RBF', 'Matern52', 'Matern32', 'RationalQuadratic',
        'Sigmoid';
        :param gp_mean: string indicating which mean function to use for
        regression. Valid options are 'ExpDecay', if None the zero-mean function
        is used;
        :param initial_gmat_value: initial value for the matrix G multiplicating
        the Wiener process.
        :param batch_size: number of elements in the batch used in the training
        process;
        :param optimizer: string describing the type of optimizer to use, valid
        values are 'RMSProp' or 'Adam';
        :param starter_learning_rate: starting value of the learning rate of
        the optimizer before the eventual decay;
        :param state_normalization: boolean, indicates whether to normalize the
        states values before the optimization (notice the parameter values
        theta won't change);
        :param time_normalization: boolean, indicates whether to normalize the
        time stamps before the optimization (notice the parameter values
        theta won't change);
        :param diagonal_wiener_process: use a diagonal wiener process;
        :param single_variable_wiener_process: use a diagonal wiener process
        with only one variable (e.g. sigma * eye(n_states);
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting);
        """
        # Save arguments
        self.trainable = trainable
        self.system_data = np.copy(system_data)
        self.t_data = np.copy(t_data).reshape(-1, 1)
        self.dim, self.n_p = system_data.shape
        self.gp_kernel = gp_kernel
        self.gp_mean = gp_mean
        self.batch_size_int = batch_size
        self.optimizer = optimizer
        self.starter_learning_rate = starter_learning_rate
        self.use_single_gp = use_single_gp
        self.diagonal_wiener_process = diagonal_wiener_process
        self.single_variable_wiener_process = single_variable_wiener_process
        # Initialize utils
        self._compute_standardization_data(state_normalization,
                                           time_normalization)
        # TensorFlow placeholders and constants
        self._build_tf_data()
        # Supporting classes
        self.data_gp_sampler =\
            GaussianProcessSDE(self.dim,
                               self.n_p,
                               self.gp_kernel,
                               self.gp_mean,
                               use_single_gp,
                               diagonal_wiener_process,
                               single_variable_wiener_process,
                               initial_gmat_value)
        # Initialization of some attributes
        self.init = None
        self.ou_process = None
        self.negative_data_loglikelihood = None
        self.noise_batch = None
        self.model_batch = None
        self.data_batch = None
        return

    def _compute_standardization_data(self, state_normalization: bool,
                                      time_normalization: bool) -> None:
        """
        Compute the means and the standard deviations for data standardization,
        used in the GP regression.
        :param state_normalization: boolean, indicates whether to normalize the
        states values before the optimization (notice the parameter values
        theta won't change);
        :param time_normalization: boolean, indicates whether to normalize the
        time stamps before the optimization (notice the parameter values
        theta won't change).
        """
        # Compute mean and std dev of the state and time values
        if state_normalization:
            self.system_data_means = np.mean(self.system_data,
                                             axis=1).reshape(self.dim, 1)
            self.system_data_std_dev = np.std(self.system_data,
                                              axis=1).reshape(self.dim, 1)
        else:
            self.system_data_means = np.zeros([self.dim, 1])
            self.system_data_std_dev = np.ones([self.dim, 1])
        if time_normalization:
            self.t_data_mean = np.mean(self.t_data)
            self.t_data_std_dev = np.std(self.t_data)
        else:
            self.t_data_mean = 0.0
            self.t_data_std_dev = 1.0
        # For the sigmoid kernel the input time values must be positive, i.e.
        # we only divide by the standard deviation
        if self.gp_kernel == 'Sigmoid':
            self.t_data_mean = 0.0
        # Normalize states and time
        self.normalized_states = (self.system_data - self.system_data_means) / \
            self.system_data_std_dev
        self.normalized_t_data = (self.t_data - self.t_data_mean) / \
            self.t_data_std_dev
        return

    def _build_tf_data(self) -> None:
        """
        Initialize all the TensorFlow constants needed by the pipeline.
        """
        self.system = tf.constant(self.normalized_states, dtype=tf.float64)
        self.t = tf.constant(self.normalized_t_data, dtype=tf.float64)
        self.system_means = tf.constant(self.system_data_means,
                                        dtype=tf.float64,
                                        shape=[1, self.dim, 1])
        self.system_std_dev = tf.constant(self.system_data_std_dev,
                                          dtype=tf.float64,
                                          shape=[1, self.dim, 1])
        self.t_mean = tf.constant(self.t_data_mean, dtype=tf.float64)
        self.t_std_dev = tf.constant(self.t_data_std_dev, dtype=tf.float64)
        self.n_points = tf.constant(self.n_p, dtype=tf.int32)
        self.dimensionality = tf.constant(self.dim, dtype=tf.int32)
        self.batch_size = tf.constant(self.batch_size_int, dtype=tf.int32)
        return

    def _build_gmat_bounds_gp(self) -> Tuple:
        """
        Build the bounds for the G matrix multiplying the Wiener Process for the
        Gaussian Process fit.
        :return: a tuple containing the lower and upper bounds for G.
        """
        # G Wiener Process
        n_elements = self.data_gp_sampler.g_vector.shape[0]
        gmat_bounds_min = np.zeros(n_elements)
        gmat_bounds_max = np.zeros(n_elements)
        if not self.single_variable_wiener_process:
            gmat_bounds_min[0:self.dim] = np.log(1e-4)
            gmat_bounds_max[0:self.dim] = np.reshape(
                np.log(self.system_data_std_dev * 2.0), [-1])
        else:
            gmat_bounds_min[0] = np.log(1e-2)
            gmat_bounds_max[0] = np.max(np.log(self.system_data_std_dev * 2.0))
        # Off-diagonal elements
        if not (self.diagonal_wiener_process or
                self.single_variable_wiener_process):
            gmat_bounds_min[self.dim:n_elements] = -10.0
            gmat_bounds_max[self.dim:n_elements] = 10.0
        gmat_bounds = (gmat_bounds_min, gmat_bounds_max)
        return gmat_bounds

    def _build_var_to_bounds_gp(self) -> dict:
        """
        Builds the dictionary containing the bounds that will be applied to the
        variable in the Gaussian Process model.
        :return: the dictionary variables to bounds.
        """
        # Extract TF variables and select the GP ones
        t_vars = tf.trainable_variables()
        gp_vars = [var for var in t_vars if 'gaussian_process' in var.name]
        # GP hyper-parameters
        gp_kern_lengthscale_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_variance_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_likelihood_bounds = (np.log(1e-6), np.log(100.0))
        # G Wiener Process
        gmat_bounds = self._build_gmat_bounds_gp()
        var_to_bounds = {gp_vars[0]: gp_kern_lengthscale_bounds,
                         gp_vars[1]: gp_kern_variance_bounds,
                         gp_vars[2]: gp_kern_likelihood_bounds,
                         gp_vars[3]: gmat_bounds}
        return var_to_bounds

    def _build_var_to_bounds_gp_sigmoid(self) -> dict:
        """
        Builds the dictionary containing the bounds that will be applied to the
        variable in the Gaussian Process model (specific for a sigmoid kernel).
        :return: the dictionary variables to bounds.
        """
        # Extract TF variables and select the GP ones
        t_vars = tf.trainable_variables()
        gp_vars = [var for var in t_vars if 'gaussian_process' in var.name]
        # GP hyper-parameters
        gp_kern_a_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_b_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_variance_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_likelihood_bounds = (np.log(1e-6), np.log(100.0))
        # Sigma Wiener Process
        gmat_bounds = self._build_gmat_bounds_gp()
        var_to_bounds = {gp_vars[0]: gp_kern_a_bounds,
                         gp_vars[1]: gp_kern_b_bounds,
                         gp_vars[2]: gp_kern_variance_bounds,
                         gp_vars[3]: gp_kern_likelihood_bounds,
                         gp_vars[4]: gmat_bounds}
        return var_to_bounds

    def _train_data_based_gp(self, session: tf.Session()) -> None:
        """
        Perform the GP regression on the data of the system. For each
        state of the system we train a different GP by maximum likelihood to fix
        the kernel hyper-parameters and the matrix G.
        :param session: TensorFlow session used during the optimization.
        """
        t_vars = tf.trainable_variables()
        gp_vars = [var for var in t_vars if 'gaussian_process' in var.name]
        if self.gp_kernel == 'Sigmoid':
            var_to_bounds = self._build_var_to_bounds_gp_sigmoid()
        else:
            var_to_bounds = self._build_var_to_bounds_gp()
        self.data_gp_optimizer = ExtendedScipyOptimizerInterface(
            self.negative_data_loglikelihood, method="L-BFGS-B",
            var_list=gp_vars, var_to_bounds=var_to_bounds)
        self.data_gp_optimizer.basinhopping(session, n_iter=50, stepsize=0.05)
        return

    def _sample_next_data_batch(self) -> tf.Tensor:
        """
        Get the next batch of derivative posterior draws from the data-based
        Gaussian Process.
        :return: TensorFlow tensor containing the batch, with shape
        [batch_size, dimensionality, n_points_int].
        """
        posterior_derivative_samples = \
            self.data_gp_sampler.sample_from_derivative_posterior(
                self.system, self.batch_size)
        return posterior_derivative_samples

    def _sample_next_model_batch(self) -> tf.Tensor:
        """
        Get the next batch of posterior draws from the ODE-based Gaussian
        Process.
        :return: TensorFlow tensor containing the batch, with shape
        [batch_size, dimensionality, n_points_int].
        """
        # Compute the unnormalized noisy batch
        noisy_observations = tf.expand_dims(self.system, 0)\
            - self.noise_batch
        samples = self.data_gp_sampler.sample_from_model_posterior(
            noisy_observations) * self.system_std_dev + self.system_means
        # Compute the gradients and return the normalized gradients
        gradients = self.trainable.compute_gradients(
            samples + self.noise_batch * self.system_std_dev)\
            + self.noise_batch * self.system_std_dev
        return gradients / self.system_std_dev * self.t_std_dev

    def build_model(self) -> None:
        """
        Builds Some common part of the computational graph for the optimization.
        """
        # Data GP Sampler
        self.data_gp_sampler.build_supporting_covariance_matrices(
            self.t, self.system_std_dev,
            tf.squeeze(self.t_mean), tf.squeeze(self.t_std_dev))
        self.negative_data_loglikelihood =\
            - self.data_gp_sampler.compute_average_log_likelihood(self.system)
        # Ornstein - Uhlenbeck Process
        self.ou_process = OrnsteinUhlenbeckProcess(
            self.t * self.t_std_dev + self.t_mean, self.dimensionality,
            self.n_points, self.data_gp_sampler.total_c_phi,
            self.data_gp_sampler.omega_matrix,
            self.data_gp_sampler.b_matrix, self.data_gp_sampler.t_matrix,
            self.data_gp_sampler.s_matrix_inv)
        # TensorFlow Operations
        self.noise_batch = self.ou_process.sample_noise_batch(self.system,
                                                              self.batch_size)
        self.data_batch = self._sample_next_data_batch()
        self.model_batch = self._sample_next_model_batch()
        return

    def _initialize_variables(self) -> None:
        """
        Initialize all the variables and placeholders in the graph.
        """
        self.init = tf.global_variables_initializer()
        return

    def _visualize_batches(self, session: tf.Session) -> None:
        """
        Visualize the samples from the model and the data during training.
        :param session: TensorFlow session used during training.
        """
        data_batch = session.run(self.data_batch)
        model_batch = session.run(self.model_batch)
        n_samples = min(self.batch_size_int - 1, 32 - 1)
        # Latex fonts
        plt.close()
        plt.figure()
        plt.plot(self.t_data, data_batch[n_samples - 1, 0, :], 'C0',
                 label=r'$\dot{\mathbf{x}}(t)$ Data-based')
        plt.plot(self.t_data, model_batch[n_samples - 1, 0, :], 'C1',
                 label=r'$\dot{\mathbf{x}}(t)$ Model-based')
        for i_state in range(self.dim):
            for j in range(n_samples):
                plt.plot(self.t_data, data_batch[j, i_state, :], 'C0')
                plt.plot(self.t_data, model_batch[j, i_state, :], 'C1')
        plt.tight_layout()
        plt.show(block=False)
        return
