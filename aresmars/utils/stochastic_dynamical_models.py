"""
Stochastic dynamical systems needed to create the data for the experiments.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Callable, Tuple
import sdeint


class StochasticDynamicalSystem(ABC):
    """
    Abstract class for stochastic dynamical systems. Uses the sdeint library
    to integrate the SDEs.
    """

    def __init__(self,
                 dimensionality: int,
                 true_param: Union[list, np.array],
                 wiener_std_dev: Union[list, np.array],
                 obs_variance: float):
        """
        Constructor.
        :param dimensionality: dimension of the state of the system;
        :param true_param: true parameters of the system;
        :param wiener_std_dev: matrix G that multiplies the Wiener process;
        :param obs_variance: variance of the observation noise;
        """
        self.dim = int(dimensionality)
        self.theta = np.array(true_param)
        self.wiener_std_dev = wiener_std_dev
        self.obs_variance = obs_variance
        self.f = self._build_f()
        self.g = self._build_g()
        return

    @abstractmethod
    def _build_f(self):
        def f():
            return []
        return f

    @abstractmethod
    def _build_g(self):
        def g():
            return []
        return g

    def simulate(self,
                 initial_state: Union[float, list, np.array],
                 initial_time: float,
                 final_time: float,
                 t_delta_integration: float) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an sdeint built-in SDE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array with dimensions [n_times, self.dim].
        """
        f = self._build_f()
        g = self._build_g()
        t_int = np.arange(initial_time, final_time, t_delta_integration)
        system_int = sdeint.itoint(f, g, initial_state, t_int)
        return system_int.T, t_int.reshape(-1, 1)

    def _observe(self,
                 initial_state: Union[float, list, np.array],
                 initial_time: float, final_time: float,
                 t_delta_integration: float,
                 t_delta_observation: float) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an sdeint built-in SDE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals.
        :return: a numpy array with dimensions [n_times_obs, self.dim].
        """
        system, t = self.simulate(initial_state,
                                  initial_time,
                                  final_time,
                                  t_delta_integration)
        t_obs = np.arange(initial_time, final_time + t_delta_observation,
                          t_delta_observation)[:]
        system_obs = np.zeros([self.dim, t_obs.shape[0]])
        for n in range(self.dim):
            system_obs[n, :] = np.interp(t_obs, t[:, 0], system[n, :])
        if self.obs_variance != 0.0:
            system_obs += np.random.normal(loc=0.0,
                                           scale=np.sqrt(self.obs_variance),
                                           size=system_obs.shape)
        system_obs = np.expand_dims(system_obs, 0)
        return system_obs, t_obs.reshape(-1, 1)

    def observe(self, initial_state: Union[float, list, np.array],
                initial_time: float, final_time: float,
                t_delta_integration: float,
                t_delta_observation: float, n_rollouts: int = 1)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an sdeint built-in SDE solver and extract the
        noisy observations, n_rollouts times.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals;
        :param n_rollouts: number of rollouts to observe;
        :return: a numpy array with dimensions [n_times_obs, self.dim].
        """
        total_system_obs, t_obs = self._observe(initial_state,
                                                initial_time,
                                                final_time,
                                                t_delta_integration,
                                                t_delta_observation)
        for n_roll in range(1, n_rollouts):
            system_obs, t_obs = self._observe(initial_state, initial_time,
                                              final_time, t_delta_integration,
                                              t_delta_observation)
            total_system_obs = np.concatenate((total_system_obs, system_obs),
                                              axis=0)
        return total_system_obs, t_obs.reshape(-1, 1)


class OrnsteinUhlenbeckSDE(StochasticDynamicalSystem):
    """
    1D Ornstein-Uhlenbeck process.
    """

    def __init__(self,
                 true_param: Union[list, np.array] = (0.5, 0.0),
                 wiener_std_dev: Union[float, list, np.array] = 1.0,
                 obs_variance: float = 0.0):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param wiener_std_dev: matrix G that multiplies the Wiener process;
        :param obs_variance: variance of the observation noise;
        """
        super(OrnsteinUhlenbeckSDE, self).__init__(1,
                                                   true_param,
                                                   wiener_std_dev,
                                                   obs_variance)
        return

    def _build_f(self) -> Callable:
        """
        Compute the function f that describes the overall evolution of the
        system in the form:
                dy / dt = f(y, theta) + G dW
        Needed by sdeint.
        :return: the f function so built.
        """
        def f(y, t):
            return self.theta[0] * (self.theta[1] - y)
        return f

    def _build_g(self) -> Callable:
        """
        Compute the function f that describes the the Wiener process G of the
        system in the form:
                dy / dt = f(y, theta) + G dW
        Needed by sdeint.
        :return: the g function so built.
        """
        def g(y, t):
            return self.wiener_std_dev
        return g

    def simulate(self,
                 initial_state: Union[float, list, np.array] = 0.0,
                 initial_time: float = 0.0,
                 final_time: float = 25.0,
                 t_delta_integration: float = 0.01)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an sdeint built-in SDE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array with dimensions [n_times, self.dim].
        """
        system, t = super(OrnsteinUhlenbeckSDE,
                          self).simulate(initial_state,
                                         initial_time,
                                         final_time,
                                         t_delta_integration)
        return system, t.reshape(-1, 1)

    def observe(self, initial_state: Union[list, np.array] = 0.0,
                initial_time: float = 0.0, final_time: float = 25.0,
                t_delta_integration: float = 0.01,
                t_delta_observation: float = 1.0,
                n_rollouts: int = 1) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an sdeint built-in SDE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals;
        :param n_rollouts: number of rollouts to observe;
        :return: a numpy array with dimensions [n_times_obs, self.dim].
        """
        system_obs, t_obs = super(OrnsteinUhlenbeckSDE,
                                  self).observe(initial_state,
                                                initial_time,
                                                final_time,
                                                t_delta_integration,
                                                t_delta_observation,
                                                n_rollouts)
        return system_obs, t_obs.reshape(-1, 1)


class DoubleWellPotentialSDE(StochasticDynamicalSystem):
    """
    1D Double-Well Potential (Ginzburg - Landau) SDE.
    """

    def __init__(self,
                 true_param: Union[list, np.array] = (0.1, 4.0),
                 wiener_std_dev: Union[float, list, np.array] = 0.5,
                 obs_variance: float = 0.0):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param wiener_std_dev: matrix G that multiplies the Wiener process;
        :param obs_variance: variance of the observation noise;
        """
        super(DoubleWellPotentialSDE, self).__init__(1,
                                                     true_param,
                                                     wiener_std_dev,
                                                     obs_variance)
        return

    def _build_f(self) -> Callable:
        """
        Compute the function f that describes the overall evolution of the
        system in the form:
                dy / dt = f(y, theta) + G dW
        Needed by sdeint.
        :return: the f function so built.
        """
        def f(y, t):
            return self.theta[0] * y * (self.theta[1] - y ** 2)
        return f

    def _build_g(self) -> Callable:
        """
        Compute the function f that describes the the Wiener process G of the
        system in the form:
                dy / dt = f(y, theta) + G dW
        Needed by sdeint.
        :return: the g function so built.
        """
        def g(y, t):
            return self.wiener_std_dev
        return g

    def simulate(self,
                 initial_state: Union[float, list, np.array] = 0.0,
                 initial_time: float = 0.0,
                 final_time: float = 25.0,
                 t_delta_integration: float = 0.01)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an sdeint built-in SDE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array with dimensions [n_times, self.dim].
        """
        system, t = super(DoubleWellPotentialSDE,
                          self).simulate(initial_state,
                                         initial_time,
                                         final_time,
                                         t_delta_integration)
        return system, t.reshape(-1, 1)

    def observe(self,
                initial_state: Union[list, np.array] = 0.0,
                initial_time: float = 0.0, final_time: float = 25.0,
                t_delta_integration: float = 0.01,
                t_delta_observation: float = 1.0,
                n_rollouts: int = 1) -> Tuple[np.array, np.array]:
        """®
        Integrate the system using an sdeint built-in SDE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals;
        :param n_rollouts: number of rollouts to observe;
        :return: a numpy array with dimensions [n_times_obs, self.dim].
        """
        system_obs, t_obs = super(DoubleWellPotentialSDE,
                                  self).observe(initial_state,
                                                initial_time,
                                                final_time,
                                                t_delta_integration,
                                                t_delta_observation,
                                                n_rollouts)
        return system_obs, t_obs.reshape(-1, 1)


class LotkaVolterraSDE(StochasticDynamicalSystem):
    """
    2D Lotka-Volterra SDE.
    """

    def __init__(self,
                 true_param: Union[list, np.array] = (2.0, 1.0, 4.0, 1.0),
                 wiener_std_dev: Union[float, list, np.array] = (1.0, 1.0),
                 obs_variance: float = 0.1):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param wiener_std_dev: matrix G that multiplies the Wiener process;
        :param obs_variance: variance of the observation noise;
        """
        super(LotkaVolterraSDE, self).__init__(2,
                                               true_param,
                                               wiener_std_dev,
                                               obs_variance)
        assert self.theta.shape[0] == 4,\
            "Error: length of true_param should be 4"
        return

    def _build_f(self) -> Callable:
        """
        Compute the function f that describes the overall evolution of the
        system in the form:
                dy / dt = f(y, theta) + G dW
        Needed by sdeint.
        :return: the f function so built.
        """
        def f(y, t):
            f_vector = [self.theta[0] * y[0] - self.theta[1] * y[0] * y[1],
                        - self.theta[2] * y[1] + self.theta[3] * y[0] * y[1]]
            return np.array(f_vector)
        return f

    def _build_g(self) -> Callable:
        """
        Compute the function f that describes the the Wiener process G of the
        system in the form:
                dy / dt = f(y, theta) + G dW
        Needed by sdeint.
        :return: the g function so built.
        """
        def g(y, t):
            return self.wiener_std_dev
        return g

    def simulate(self,
                 initial_state: Union[list, np.array] = (5.0, 3.0),
                 initial_time: float = 0.0,
                 final_time: float = 2.0,
                 t_delta_integration: float = 0.01)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an sdeint built-in SDE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array with dimensions [n_times, self.dim].
        """
        system, t = super(LotkaVolterraSDE, self).simulate(
            initial_state, initial_time, final_time, t_delta_integration)
        return system, t.reshape(-1, 1)

    def observe(self,
                initial_state: Union[list, np.array] = (5.0, 3.0),
                initial_time: float = 0.0, final_time: float = 2.0,
                t_delta_integration: float = 0.01,
                t_delta_observation: float = 0.1,
                n_rollouts: int = 1) \
            -> Tuple[np.array, np.array]:
        """®
        Integrate the system using an sdeint built-in SDE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals;
        :param n_rollouts: number of rollouts to simulate and observe;
        :return: a numpy array with dimensions [n_times_obs, self.dim].
        """
        system_obs, t_obs = super(LotkaVolterraSDE,
                                  self).observe(initial_state,
                                                initial_time,
                                                final_time,
                                                t_delta_integration,
                                                t_delta_observation,
                                                n_rollouts)
        return system_obs, t_obs.reshape(-1, 1)


class Lorenz63SDE(StochasticDynamicalSystem):
    """
    3D Lorenz '63 Stochastic System.
    """

    def __init__(self,
                 true_param: Union[list, np.array] = (10.0, 28.0, 2.66667),
                 wiener_std_dev: Union[float, list, np.array] =
                 np.diag([6.0, 6.0, 6.0]), obs_variance: float = 1.0):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param wiener_std_dev: matrix G that multiplies the Wiener process;
        :param obs_variance: variance of the observation noise;
        """
        super(Lorenz63SDE, self).__init__(3,
                                          true_param,
                                          wiener_std_dev,
                                          obs_variance)
        assert self.theta.shape[0] == 3,\
            "Error: length of true_param should be 3"
        return

    def _build_f(self) -> Callable:
        """
        Compute the function f that describes the overall evolution of the
        system in the form:
                dy / dt = f(y, theta) + G dW
        Needed by sdeint.
        :return: the f function so built.
        """
        def f(y, t):
            f_vector = [self.theta[0] * (y[1] - y[0]),
                        self.theta[1] * y[0] - y[1] - y[0] * y[2],
                        y[0] * y[1] - self.theta[2] * y[2]]
            return np.array(f_vector)
        return f

    def _build_g(self) -> Callable:
        """
        Compute the function f that describes the the Wiener process G of the
        system in the form:
                dy / dt = f(y, theta) + G dW
        Needed by sdeint.
        :return: the g function so built.
        """
        def g(y, t):
            return self.wiener_std_dev
        return g

    def simulate(self,
                 initial_state: Union[list, np.array] = (-7.0, -7.0, -7.0),
                 initial_time: float = 0.0,
                 final_time: float = 20.0,
                 t_delta_integration: float = 0.01)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an sdeint built-in SDE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array with dimensions [n_times, self.dim].
        """
        system, t = super(Lorenz63SDE, self).simulate(initial_state,
                                                      initial_time,
                                                      final_time,
                                                      t_delta_integration)
        return system, t.reshape(-1, 1)

    def observe(self,
                initial_state: Union[list, np.array] = -7. * np.ones(3),
                initial_time: float = 0.0,
                final_time: float = 20.0,
                t_delta_integration: float = 0.01,
                t_delta_observation: float = 0.1,
                n_rollouts: int = 1) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an sdeint built-in SDE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals;
        :param n_rollouts: number of rollouts to observe;
        :return: a numpy array with dimensions [n_times_obs, self.dim].
        """
        system_obs, t_obs = super(Lorenz63SDE,
                                  self).observe(initial_state,
                                                initial_time,
                                                final_time,
                                                t_delta_integration,
                                                t_delta_observation,
                                                n_rollouts)
        return system_obs, t_obs.reshape(-1, 1)
