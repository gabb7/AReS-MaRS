"""
The file presents an alternative implementation of the ScipyOptimizerInterface
which is present in TensorFlow. In particular, a wrapper to the basinhopping
routine present in Scipy was added.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from tensorflow.contrib.opt import ScipyOptimizerInterface
from tensorflow.python.framework import ops
from scipy.optimize._basinhopping import RandomDisplacement
import numpy as np


class BoundedRandomDisplacement(RandomDisplacement):
    """
    Class that overwrites the RandomDisplacement one, default for scipy
    basinhopping, and adds the bounds clipping.
    """

    def __init__(self, stepsize: float = 1.0, bounds: list = None):
        """
        Constructor.
        :param stepsize: maximum length of step;
        :param bounds: bounds for the search of the optimum x.
        """
        super(BoundedRandomDisplacement, self).__init__(stepsize)
        if bounds is None:
            self.bounds_min = - np.infty
            self.bounds_max = np.infty
        else:
            bounds_np = np.array(bounds)
            self.bounds_min = bounds_np[:, 0]
            self.bounds_max = bounds_np[:, 1]
        return

    def __call__(self, x):
        """
        Compute the new x and clips it to the bounds.
        :param x: old starting point.
        :return: new starting point.
        """
        new_x = super(BoundedRandomDisplacement, self).__call__(x)
        clipped_x = np.clip(new_x, self.bounds_min, self.bounds_max)
        if np.sum(np.abs(clipped_x - new_x)) > 1e-6:
            self.stepsize = self.stepsize / 2.0
        return clipped_x


class ExtendedScipyOptimizerInterface(ScipyOptimizerInterface):
    """
    Extension of the ScipyOptimizerInterface class which is present in
    TensorFlow. This one allows for calling the basinhopping routine in Scipy.
    """

    def basinhopping(self,
                     session=None,
                     n_iter=10,
                     temperature=1.0,
                     stepsize=0.05,
                     feed_dict=None,
                     fetches=None,
                     step_callback=None,
                     loss_callback=None,
                     **run_kwargs):
        """
        Minimize a scalar `Tensor` through the Scipy routine basinhopping.
        Variables subject to optimization are updated in-place at the end of
        optimization.
        Note that this method does *not* just return a minimization `Op`, unlike
        `Optimizer.minimize()`; instead it actually performs minimization by
        executing commands to control a `Session`.
        :param session: A `Session` instance.
        :param n_iter: The number of basin-hopping iterations.
        :param temperature: The “temperature” parameter for the accept or reject
        criterion. Higher “temperatures” mean that larger jumps in function
        value will be accepted. For best results T should be comparable to the
        separation (in function value) between local minima.
        :param stepsize: Maximum step size for use in the random displacement.
        :param feed_dict: A feed dict to be passed to calls to `session.run`.
        :param fetches: A list of `Tensor`s to fetch and supply to
        `loss_callback` as positional arguments.
        :param step_callback: A function to be called at each optimization step;
        arguments are the current values of all optimization variables
        flattened into a single vector.
        :param loss_callback: A function to be called every time the loss and
        gradients are computed, with evaluated fetches supplied as positional
        arguments.
        :param run_kwargs: kwargs to pass to `session.run`.
        """
        self._DEFAULT_METHOD = 'L-BFGS-B'
        # Set arguments as input or empty elements
        session = session or ops.get_default_session()
        feed_dict = feed_dict or {}
        fetches = fetches or []
        loss_callback = loss_callback or (lambda *fetches: None)
        step_callback = step_callback or (lambda xk: None)

        # Construct loss function and associated gradient.
        loss_grad_func = self._make_eval_func([self._loss,
                                               self._packed_loss_grad], session,
                                              feed_dict, fetches, loss_callback)

        # Construct equality constraint functions and associated gradients.
        equality_funcs = self._make_eval_funcs(self._equalities, session,
                                               feed_dict, fetches)
        equality_grad_funcs = self._make_eval_funcs(self._packed_equality_grads,
                                                    session, feed_dict, fetches)

        # Construct inequality constraint functions and associated gradients.
        inequality_funcs = self._make_eval_funcs(self._inequalities, session,
                                                 feed_dict, fetches)
        inequality_grad_funcs = self._make_eval_funcs(
            self._packed_inequality_grads,
            session, feed_dict, fetches)

        # Get initial value from TF session.
        initial_packed_var_val = session.run(self._packed_var)

        # Perform minimization.
        packed_var_val = self._basinhopping(
            initial_val=initial_packed_var_val,
            loss_grad_func=loss_grad_func,
            n_iter=n_iter,
            temperature=temperature,
            stepsize=stepsize,
            equality_funcs=equality_funcs,
            equality_grad_funcs=equality_grad_funcs,
            inequality_funcs=inequality_funcs,
            inequality_grad_funcs=inequality_grad_funcs,
            packed_bounds=self._packed_bounds,
            step_callback=step_callback,
            optimizer_kwargs=self.optimizer_kwargs)
        var_vals = [
            packed_var_val[packing_slice] for packing_slice in
            self._packing_slices
        ]

        # Set optimization variables to their new values.
        session.run(
            self._var_updates,
            feed_dict=dict(zip(self._update_placeholders, var_vals)),
            **run_kwargs)

        return

    def _basinhopping(self,
                      initial_val,
                      loss_grad_func,
                      n_iter,
                      temperature,
                      stepsize,
                      equality_funcs,
                      equality_grad_funcs,
                      inequality_funcs,
                      inequality_grad_funcs,
                      packed_bounds,
                      step_callback,
                      optimizer_kwargs):
        """
        Wrapper to the actual Scipy routine.
        :param initial_val: A NumPy vector of initial values.
        :param loss_grad_func: A function accepting a NumPy packed variable
        vector and returning two outputs, a loss value and the gradient of that
        loss with respect to the packed variable vector.
        :param n_iter: The number of basin-hopping iterations.
        :param temperature: The “temperature” parameter for the accept or reject
        criterion. Higher “temperatures” mean that larger jumps in function
        value will be accepted. For best results T should be comparable to the
        separation (in function value) between local minima.
        :param stepsize: Maximum step size for use in the random displacement.
        :param equality_funcs: A list of functions each of which specifies a
        scalar quantity that an optimizer should hold exactly zero.
        :param equality_grad_funcs: A list of gradients of equality_funcs.
        :param inequality_funcs: A list of functions each of which specifies a
        scalar quantity that an optimizer should hold >= 0.
        :param inequality_grad_funcs: A list of gradients of inequality_funcs.
        :param packed_bounds: A list of bounds for each index, or `None`.
        :param step_callback: A callback function to execute at each
        optimization step, supplied with the current value of the packed
        variable vector.
        :param optimizer_kwargs: Other key-value arguments available to the
        optimizer.
        """
        def loss_grad_func_wrapper(x):
            # SciPy's L-BFGS-B Fortran implementation requires gradients as
            # doubles.
            loss, gradient = loss_grad_func(x)
            return loss, gradient.astype('float64')

        optimizer_kwargs = dict(optimizer_kwargs.items())
        method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)

        constraints = []
        for func, grad_func in zip(equality_funcs, equality_grad_funcs):
            constraints.append({'type': 'eq', 'fun': func, 'jac': grad_func})
        for func, grad_func in zip(inequality_funcs, inequality_grad_funcs):
            constraints.append({'type': 'ineq', 'fun': func, 'jac': grad_func})

        minimizer_kwargs = {
            'jac': True,
            'callback': step_callback,
            'method': method,
            'constraints': constraints,
            'bounds': packed_bounds,
        }

        for kwarg in minimizer_kwargs:
            if kwarg in optimizer_kwargs:
                if kwarg == 'bounds':
                    # Special handling for 'bounds' kwarg since ability to
                    # specify bounds was added after this module was already
                    # publicly released.
                    raise ValueError(
                        'Bounds must be set using the var_to_bounds argument')
                raise ValueError(
                    'Optimizer keyword arg \'{}\' is set '
                    'automatically and cannot be injected manually'.format(
                        kwarg))

        minimizer_kwargs.update(optimizer_kwargs)

        import scipy.optimize  # pylint: disable=g-import-not-at-top
        result = scipy.optimize.basinhopping(
            loss_grad_func_wrapper, initial_val, niter=n_iter, T=temperature,
            stepsize=stepsize,
            take_step=BoundedRandomDisplacement(bounds=self._packed_bounds),
            minimizer_kwargs=minimizer_kwargs)

        return result['x']
