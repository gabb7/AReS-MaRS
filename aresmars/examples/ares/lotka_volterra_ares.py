"""
Example script that runs the ARES regression on the classic setting for the
Lotka - Volterra predator-prey model.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Import libraries
import numpy as np
import tensorflow as tf
from aresmars import LotkaVolterraSDE
from aresmars import TrainableLotkaVolterra
from aresmars import ARES


# Fix the random seeds for reproducibility
seed = 58718
np.random.seed(seed)
tf.set_random_seed(seed)


# 1) Use the provided utilities class to simulate some noisy observations of
#    the Lotka - Volterra model

g_matrix = np.array([[0.2, 0.0], [0.1, 0.3]])

lotka_volterra_simulator = LotkaVolterraSDE(
    true_param=(2.0, 1.0, 4.0, 1.0),
    obs_variance=0.1**2,
    wiener_std_dev=g_matrix)

system_obs, t_obs = \
    lotka_volterra_simulator.observe(initial_state=[5.0, 3.0],
                                     final_time=2.0,
                                     t_delta_integration=1e-4,
                                     t_delta_observation=0.04,
                                     n_rollouts=1)
system_obs = np.squeeze(system_obs)


# 2) Initialize the provided TrainableLotkaVolterra class

trainable_lv = TrainableLotkaVolterra()


# 3) Run the actual ARES regression by initializing the optimizer, building the
#    model and calling the train() function

ares_solver = ARES(trainable_lv,
                   system_obs,
                   t_obs,
                   gp_kernel='RBF',
                   gp_mean=None,
                   batch_size=128,
                   critic_architecture=[128, 128],
                   critic_steps=8,
                   gen_steps=1,
                   c_clipping=0.01,
                   optimizer='RMSProp',
                   starter_learning_rate=1e-2,
                   state_normalization=True,
                   time_normalization=True,
                   diagonal_wiener_process=False,
                   use_single_gp=False)
ares_solver.build_model()

# Train the model
final_theta, final_g = ares_solver.train(n_batches=5000)
