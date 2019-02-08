"""
AReS (Adversarial Regression for SDEs) implementation.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from aresmars.core.base_class import AresMarsBaseClass
from aresmars.utils.trainable_models import TrainableModelSDE
from typing import Tuple
import tensorflow as tf
import numpy as np


class ARESCritic(object):
    """
    Multilayer perceptron that act as the critic in the WGAN of ARES.
    """

    def __init__(self,
                 n_states: int,
                 n_points: int,
                 architecture: list = (128, 64),
                 batch_size: int = 128):
        """
        Constructor.
        :param n_states: dimensionality of the system (number of states);
        :param n_points: number of points for each sample;
        :param architecture: list of number of nodes nodes, one for each layer
        of the network;
        :param batch_size: number of samples in one batch.
        """
        # Architecture details
        self.dimensionality = n_states
        self.n_points = n_points
        self.architecture = architecture
        self.n_layers = len(architecture)
        self.batch_size = batch_size
        # Weights and biases of the network
        self._build_weights_dictionary()
        self._build_biases_dictionary()
        return

    def _build_weights_dictionary(self) -> None:
        """
        Build the dictionary that will contain all the weights in the network.
        """
        with tf.variable_scope('critic'):
            self.weights = {
                'wfc1': tf.get_variable(
                    'wfc1', initializer=tf.random_normal([self.dimensionality *
                                                          self.n_points,
                                                          self.architecture[0]],
                                                         stddev=0.01,
                                                         dtype=tf.float64)),
                'z_out': tf.get_variable(
                    'z_out', initializer=tf.random_normal(
                        [self.architecture[-1], 1], stddev=0.01,
                        dtype=tf.float64))
            }
            if self.n_layers > 1:
                for n in range(1, self.n_layers):
                    layer_name = 'wfc' + str(n+1)
                    self.weights[layer_name] = tf.get_variable(
                        layer_name, initializer=tf.random_normal(
                            [self.architecture[n-1], self.architecture[n]],
                            stddev=0.01, dtype=tf.float64))
            self.weights_keys = sorted(self.weights.keys())
        return

    def _build_biases_dictionary(self) -> None:
        """
        Build the dictionary of the biases of the network.
        """
        with tf.variable_scope('critic'):
            self.biases = {}
            for n in range(self.n_layers):
                layer_name = 'bfc' + str(n + 1)
                self.biases[layer_name] = tf.Variable(tf.zeros(
                    self.architecture[n], dtype=tf.float64))
            self.biases['z_out'] = tf.Variable(tf.zeros(1, dtype=tf.float64))
            self.biases_keys = sorted(self.biases.keys())
        return

    def compute_batch_logits(self, batch: tf.Tensor) -> tf.Tensor:
        """
        Compute the logits from the data in the batch.
        :param batch: batch tensor of data.
        :return: the logits tensor.
        """
        x = tf.reshape(batch, shape=[self.batch_size,
                                     self.dimensionality * self.n_points])
        for n_l in range(self.n_layers):
            x = tf.matmul(x, self.weights[self.weights_keys[n_l]]) +\
                self.biases[self.biases_keys[n_l]]
            x = tf.nn.relu(x)
        out = tf.matmul(x, self.weights['z_out']) + self.biases['z_out']
        return out


class ARES(AresMarsBaseClass):
    """
    Adversarial regression and parameter inference for systems of Stochastic
    Differential Equations.
    """

    def __init__(self,
                 trainable: TrainableModelSDE,
                 system_data: np.array,
                 t_data: np.array,
                 gp_kernel: str = 'RBF',
                 gp_mean: str = None,
                 initial_gmat_value: float = None,
                 critic_architecture: list = (256, 128),
                 critic_steps: int = 2,
                 gen_steps: int = 1,
                 c_clipping: float = 0.01,
                 optimizer: str = 'Adam',
                 batch_size: int = 128,
                 starter_learning_rate: float = 1e-3,
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
        :param critic_architecture: list of integers, describing the
        architecture of the (multilayer) perceptron that act as a critic in the
        WGAN framework;
        :param critic_steps: number of SGD iterations to apply to the critic
        before going to the generator;
        :param gen_steps: number of SGD iterations to apply to the generator
        before going to the critic;
        :param c_clipping: constant for clipping the weights of the critic and
        ensure Lipschitz-continuity;
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
        super(ARES,
              self).__init__(trainable, system_data, t_data, gp_kernel, gp_mean,
                             initial_gmat_value, batch_size, optimizer,
                             starter_learning_rate, state_normalization,
                             time_normalization, diagonal_wiener_process,
                             single_variable_wiener_process, use_single_gp)
        # Save arguments
        self.c_architecture = critic_architecture
        self.c_steps = critic_steps
        self.g_steps = gen_steps
        self.c_clipping = tf.constant(c_clipping, dtype=tf.float64)
        # Initialize some fields
        self.clipping_ops = None
        self.global_step = tf.Variable(0, trainable=False)
        return

    def _build_critic(self) -> None:
        """
        Build the Neural network that will act as a critic.
        """
        self.critic = ARESCritic(n_states=self.dim, n_points=self.n_p,
                                 batch_size=self.batch_size,
                                 architecture=self.c_architecture)
        return

    def _compute_logits(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the logits for the next batches of data-based and model-based
        posterior samples.
        :return: the logit tensors.
        """
        logits_data = self.critic.compute_batch_logits(self.data_batch)
        logits_model = self.critic.compute_batch_logits(self.model_batch)
        return logits_data, logits_model

    def _build_losses(self) -> None:
        """
        Build the losses of the critic and the generator that will be
        minimized by TensorFlow during training.
        """
        # Compute the predictions with the critic Neural Network
        self.logits_data, self.logits_model = self._compute_logits()
        # critic loss
        self.c_loss = tf.reduce_mean(self.logits_data) - \
            tf.reduce_mean(self.logits_model)
        # Generator loss
        self.g_loss = tf.reduce_mean(self.logits_model)
        return

    def build_clipping_critic_weights_ops(self) -> None:
        """
        Build the necessary TensorFlow nodes (ops) in the computational graph
        to clip the weights of the critic and ensure Lipshitz continuity.
        """
        t_vars = tf.trainable_variables()
        c_vars = [var for var in t_vars if 'critic' in var.name]
        self.clipping_ops = []
        for var in c_vars:
            self.clipping_ops.append(
                tf.assign(var, tf.clip_by_value(var, -self.c_clipping,
                                                self.c_clipping)))
        return

    def _build_adam_optimizers(self, c_vars: list, g_vars: list) -> None:
        """
        Build Adam optimizer for training.
        :param c_vars: list containing the variables of the critic;
        :param g_vars: list containing the variables of the generator.
        """
        self.c_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.starter_learning_rate, beta1=0.9, beta2=0.99,
            epsilon=1e-8, use_locking=False, name='d_opt').minimize(
            self.c_loss, var_list=c_vars, global_step=self.global_step)
        self.g_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.starter_learning_rate, beta1=0.9, beta2=0.99,
            epsilon=1e-8, use_locking=False, name='g_opt').minimize(
            self.g_loss, var_list=g_vars, global_step=self.global_step)
        return

    def _build_rmsprop_optimizers(self, c_vars: list, g_vars: list) -> None:
        """
        Build RMSProp optimizers for training.
        :param c_vars: list containing the variables of the critic;
        :param g_vars: list containing the variables of the generator.
        """
        batch = tf.Variable(0, trainable=False, dtype=tf.int64)
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate,
                                                   batch,
                                                   1000000,
                                                   0.96)
        self.c_optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate).minimize(self.c_loss, var_list=c_vars)
        self.g_optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate).minimize(self.g_loss,
                                                  var_list=g_vars)
        return

    @staticmethod
    def _build_variable_lists() -> Tuple[list, list]:
        """
        Build two different lists that collect the TensorFlow variables in the
        critic and the generator.
        :return: the two lists of tf.Variables.
        """
        t_vars = tf.trainable_variables()
        c_vars = [var for var in t_vars if 'critic' in var.name]
        g_vars = [var for var in t_vars if 'system_parameters' in var.name]
        return c_vars, g_vars

    def _build_optimizers(self) -> None:
        """
        Builds the TensorFlow optimizers for training respectively (and
        separately) critic and generator.
        """
        c_vars, g_vars = self._build_variable_lists()
        if self.optimizer == 'Adam':
            self._build_adam_optimizers(c_vars, g_vars)
        elif self.optimizer == 'RMSProp':
            self._build_rmsprop_optimizers(c_vars, g_vars)
        return

    def build_model(self) -> None:
        """
        Build the actual model for the GAN-like posterior distance minimizer.
        """
        super(ARES, self).build_model()
        # Build the TensorFlow infrastructure
        self._build_critic()
        self._build_losses()
        self._build_optimizers()
        self.build_clipping_critic_weights_ops()
        return

    def train(self, n_batches: int = 10000) -> Tuple[np.array, np.array]:
        """
        Trains the model by iteratively minimizing the MMD in a stochastic
        optimization fashion.
        :param n_batches: number of batches to be iterated on;
        :return the numpy arrays containing the found parameters.
        """
        self._initialize_variables()
        session = tf.Session()
        with session:
            session.run(self.init)
            self._train_data_based_gp(session)
            n = 0
            while n < n_batches:
                for _ in range(self.c_steps):
                    session.run(self.c_optimizer)
                    session.run(self.clipping_ops)
                for _ in range(self.g_steps):
                    session.run(self.g_optimizer)
                n += 1
                if session.run(self.c_loss) == 0.0:
                    break
            theta = session.run(self.trainable.theta)
            sigma = session.run(self.data_gp_sampler.sigma_matrix)
        tf.reset_default_graph()
        return theta, sigma

    def test(self, n_batches: int = 10000,
             visualization_frequency: int = 100,
             plot: bool = False) -> Tuple[np.array, np.array]:
        """
        Trains the model by iteratively minimizing the MMD in a stochastic
        optimization fashion.
        :param n_batches: number of batches to be iterated on.
        :param visualization_frequency: frequency at which samples from the
        data and the model gets plotted;
        :param plot: boolean, whether to plot or not the batches;
        :return the numpy arrays containing the found parameters.
        """
        self._initialize_variables()
        session = tf.Session()
        with session:
            session.run(self.init)
            self._train_data_based_gp(session)
            print("Estimated Sigma matrix:\n",
                  session.run(self.data_gp_sampler.sigma_matrix))
            n = 0
            while n < n_batches:
                if visualization_frequency and n % visualization_frequency == 0:
                    print("Theta\n", session.run(self.trainable.theta))
                    print("C Loss", session.run(self.c_loss))
                    print("G Loss", session.run(self.g_loss))
                    if plot:
                        self._visualize_batches(session)
                for _ in range(self.c_steps):
                    session.run(self.c_optimizer)
                    session.run(self.clipping_ops)
                for _ in range(self.g_steps):
                    session.run(self.g_optimizer)
                if session.run(self.c_loss) == 0.0:
                    break
                n += 1
            theta = session.run(self.trainable.theta)
            sigma = session.run(self.data_gp_sampler.sigma_matrix)
        tf.reset_default_graph()
        return theta, sigma
