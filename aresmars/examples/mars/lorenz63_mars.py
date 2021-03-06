"""
Example script that runs the MARS regression on the classic setting for the
Lorenz63 attractor.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Import libraries
import numpy as np
import tensorflow as tf
from aresmars import Lorenz63SDE
from aresmars import TrainableLorenz63
from aresmars import MARS


# Fix the random seeds for reproducibility
seed = 58718
np.random.seed(seed)
tf.set_random_seed(seed)


# 1) Use the provided utilities class to simulate some noisy observations of
#    the Lorenz '63 model

g_matrix = np.sqrt(10.0) * np.eye(3)

lorenz63_system = Lorenz63SDE(
    true_param=[10.0, 28.0, 2.66667],
    obs_variance=1.0,
    wiener_std_dev=g_matrix)

system_obs, t_obs = \
    lorenz63_system.observe(initial_state=-7. * np.ones(3),
                            final_time=5.0,
                            t_delta_integration=1e-4,
                            t_delta_observation=0.025,
                            n_rollouts=1)
system_obs = np.squeeze(system_obs)


# 2) Initialize the provided TrainableLorenz63 class

trainable_l63 = TrainableLorenz63()


# 3) Run the actual MARS regression by initializing the optimizer, building the
#    model and calling the train() function

mars_solver = MARS(trainable_l63,
                   system_obs,
                   t_obs,
                   gp_kernel='RBF',
                   gp_mean=None,
                   mmd_kernel='RationalQuadratic',
                   optimizer='Adam',
                   starter_learning_rate=1e-1,
                   batch_size=128,
                   use_single_gp=True,
                   single_variable_wiener_process=True,
                   state_normalization=True,
                   time_normalization=True)

mars_solver.build_model()

# Train the model
final_theta, final_g = mars_solver.train(n_batches=2000)
