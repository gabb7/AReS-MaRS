"""
MArS (MMD-minimizing Regression for SDEs) implementation.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from aresmars.core.base_class import AresMarsBaseClass
from aresmars.utils.trainable_models import TrainableModelSDE
from aresmars.utils.mmd import *
import tensorflow as tf
from typing import Tuple
import numpy as np
import sys


class MARS(AresMarsBaseClass):
    """
    Maximum Mean Discrepancy-minimizing regression and parameter inference for
    systems of Stochastic Differential Equations.
    """

    def __init__(self,
                 trainable: TrainableModelSDE,
                 system_data: np.array,
                 t_data: np.array,
                 gp_kernel: str = 'RBF',
                 gp_mean: str = None,
                 initial_gmat_value: float = None,
                 mmd_kernel: str = 'RBF',
                 optimizer: str = 'Adam',
                 batch_size: int = 64,
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
        the Wiener process;
        :param mmd_kernel: kernel used for the computation of the MMD estimator.
        Valid options are 'RBF', 'RBFMixture', 'RationalQuadratic', 'Linear';
        :param optimizer: string describing the type of optimizer to use, valid
        values are 'RMSProp' or 'Adam';
        :param batch_size: number of elements in the batch used in the training
        process;
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
        super(MARS,
              self).__init__(trainable, system_data, t_data, gp_kernel, gp_mean,
                             initial_gmat_value, batch_size, optimizer,
                             starter_learning_rate, state_normalization,
                             time_normalization, diagonal_wiener_process,
                             single_variable_wiener_process, use_single_gp)
        self.mmd_kernel = mmd_kernel
        # Supporting classes
        self.mmd_estimator = MMD(self._initialize_mmd_kernel())
        # Attributes initialization
        self.mmd_estimator = None
        self.optimizer_tf = None
        return

    def _initialize_mmd_kernel(self) -> GenericKernel:
        """
        Initialize the selected kernel for the Maximum Mean Discrepancy
        estimator.
        :return: the aresmars.utils.mmd.GenericKernel object.
        """
        if self.mmd_kernel == 'RBF':
            return RBFKernel()
        elif self.mmd_kernel == 'RBFMixture':
            return RBFMixtureKernel()
        elif self.mmd_kernel == 'RationalQuadratic':
            return RationalQuadraticKernel()
        elif self.mmd_kernel == 'Linear':
            return LinearKernel()
        else:
            sys.exit("Kernel for MMD Estimator not valid")

    def _compute_mmd(self) -> tf.Tensor:
        """
        Compute the Maximum Mean Discrepancy between the data-based and the
         model-based batches.
        :return: the TensorFlow tensor containing the MMD estimator.
        """
        mmd = self.mmd_estimator.compute_mmd_estimator_gretton(
            self.data_batch, self.model_batch)
        return tf.sqrt(mmd)

    def _build_objective(self) -> None:
        """
        Builds the TensorFlow node containing the mmd estimator.
        """
        self.mmd_estimator = self._compute_mmd()
        return

    def _build_adam_optimizer(self) -> None:
        """
        Build Adam optimizer for training.
        """
        # Extract the variables to train (theta)
        t_vars = tf.trainable_variables()
        mmd_vars = [var for var in t_vars if 'system_parameters' in var.name]
        # Initialize Adam
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            self.starter_learning_rate, global_step, 200, 0.96,
            staircase=True)
        self.optimizer_tf = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=0.9,
            beta2=0.999, epsilon=1e-8, use_locking=False).minimize(
            self.mmd_estimator, var_list=mmd_vars, global_step=global_step)
        return

    def _build_rmsprop_optimizer(self) -> None:
        """
        Build RMSProp optimizers for training.
        """
        # Extract the variables to train (theta)
        t_vars = tf.trainable_variables()
        mmd_vars = [var for var in t_vars if 'system_parameters' in var.name]
        # Initialize RMSProp
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            self.starter_learning_rate, global_step, 200, 0.96,
            staircase=True)
        self.optimizer_tf = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate).minimize(
            self.mmd_estimator, var_list=mmd_vars, global_step=global_step)
        return

    def _build_optimizer(self) -> None:
        """
        Build the TensorFlow optimizer.
        """
        if self.optimizer == 'Adam':
            self._build_adam_optimizer()
        elif self.optimizer == 'RMSProp':
            self._build_rmsprop_optimizer()
        else:
            sys.exit("Wrong optimizer name\n")
        return

    def build_model(self) -> None:
        """
        Build the computational graph for the optimization.
        """
        super(MARS, self).build_model()
        self._build_objective()
        self._build_optimizer()
        return

    def train(self, n_batches: int = 10000) -> np.array:
        """
        Trains the model by iteratively minimizing the MMD in a stochastic
        optimization fashion.
        :param n_batches: number of batches to be iterated on;
        :return the numpy array containing the found parameters.
        """
        self._initialize_variables()
        session = tf.Session()
        with session:
            session.run(self.init)
            self._train_data_based_gp(session)
            n = 0
            while n < n_batches:
                session.run(self.optimizer_tf)
                n += 1
            theta = session.run(self.trainable.theta)
            sigma = session.run(self.data_gp_sampler.sigma_matrix)
        tf.reset_default_graph()
        return theta, sigma

    def test(self, n_batches=10000, visualization_frequency: int = 0,
             plot: bool = False) -> Tuple[np.array, np.array]:
        """
        Trains the model by iteratively minimizing the chosen metric in a
        stochastic optimization fashion.
        :param n_batches: number of batches to be iterated on;
        :param visualization_frequency: frequency at which samples from the
        data and the model gets plotted;
        :param plot: boolean, whether to plot or not the batches.
        """
        self._initialize_variables()
        session = tf.Session()
        with session:
            session.run(self.init)
            self._train_data_based_gp(session)
            n = 0
            while n < n_batches:
                if visualization_frequency and n % visualization_frequency == 0:
                    print("Theta:\n", session.run(self.trainable.theta))
                    print("MMD Estimator:\n", session.run(self.mmd_estimator))
                    if plot:
                        self._visualize_batches(session)
                session.run(self.optimizer_tf)
                n += 1
            theta = session.run(self.trainable.theta)
            sigma = session.run(self.data_gp_sampler.sigma_matrix)
        tf.reset_default_graph()
        return theta, sigma
