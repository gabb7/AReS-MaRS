"""
Example script that runs the MARS regression on the Ornstein - Uhlenbeck model.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Import libraries
import numpy as np
import tensorflow as tf
from aresmars import OrnsteinUhlenbeckSDE
from aresmars import TrainableOrnsteinUhlenbeck
from aresmars import MARS


# Fix the random seeds for reproducibility
seed = 58718
np.random.seed(seed)
tf.set_random_seed(seed)


# 1) Use the provided utilities class to simulate some noisy observations of
#    the Ornstein - Uhlenbeck model

ornstein_uhlenbeck_simulator = OrnsteinUhlenbeckSDE(true_param=(0.5, 1.0),
                                                    wiener_std_dev=0.5,
                                                    obs_variance=0.0 ** 2)
system_obs, t_obs =\
    ornstein_uhlenbeck_simulator.observe(initial_state=10.0,
                                         initial_time=0.0,
                                         final_time=20.0,
                                         t_delta_integration=1e-4,
                                         t_delta_observation=20.0/50.0)
system_obs = system_obs.reshape(1, -1)


# 2) Initialize the provided TrainableOrnsteinUhlenbeck class

trainable_ou = TrainableOrnsteinUhlenbeck()


# 3) Run the actual MARS regression by initializing the optimizer, building the
#    model and calling the train() function

mars_solver = MARS(trainable_ou,
                   system_obs,
                   t_obs,
                   gp_kernel='Sigmoid',
                   gp_mean=None,
                   mmd_kernel='RationalQuadratic',
                   optimizer='Adam',
                   starter_learning_rate=1e-2,
                   batch_size=128,
                   use_single_gp=False,
                   diagonal_wiener_process=False,
                   state_normalization=True,
                   time_normalization=True)

mars_solver.build_model()

# Train the model
final_theta, final_g = mars_solver.train(n_batches=2000)
