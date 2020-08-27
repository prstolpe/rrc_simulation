import gym
import numpy as np
import abc
import inspect
from attempt.utilities.utils import flatten_goal_observation

from typing import Union, Iterable, List, Tuple

EPSILON = 1e-6
NP_FLOAT_PREC = np.float64
NUMPY_INTEGER_PRECISION = np.int64


class StateNormalizationWrapper:
    """Wrapper for state normalization using running mean and variance estimations."""

    def __init__(self, state_shapes: Union[Iterable[Iterable[int]], Iterable[int], int], observation_names):


        # parse input types into normed shape format
        self.shapes: Iterable[Iterable[int]]
        if isinstance(state_shapes, Iterable) and all(isinstance(x, Iterable) for x in state_shapes):
            self.shapes = state_shapes
        elif isinstance(state_shapes, int):
            self.shapes = ((state_shapes,),)
        elif isinstance(state_shapes, Iterable) and all(isinstance(x, int) for x in state_shapes):
            self.shapes = (state_shapes,)
        else:
            raise ValueError("Cannot understand shape format.")

        self.mean = [np.zeros(i_shape, NP_FLOAT_PREC) for i_shape in self.shapes if len(i_shape) == 1]
        self.variance = [np.ones(i_shape, NP_FLOAT_PREC) for i_shape in self.shapes if len(i_shape) == 1]

        assert len(self.mean) > 0 and len(self.variance) > 0, "Initialized StateNormalizationWrapper got no vector " \
                                                              "states."

        self.n = 1e-4
        self.obs_names = observation_names

    def __add__(self, other) -> "BaseRunningMeanWrapper":
        # needs_shape = len(inspect.signature(self.__class__).parameters) > 0
        nw = self.__class__(tuple(m.shape for m in self.mean), self.obs_names)
        nw.n = self.n + other.n

        for i in range(len(self.mean)):
            nw.mean[i] = (self.n / nw.n) * self.mean[i] + (other.n / nw.n) * other.mean[i]
            nw.variance[i] = (self.n * (self.variance[i] + (self.mean[i] - nw.mean[i]) ** 2)
                              + other.n * (other.variance[i] + (other.mean[i] - nw.mean[i]) ** 2)) / nw.n

        return nw

    def modulate(self, step_result: Tuple, update=True) -> Tuple:
        """Normalize a given batch of 1D tensors and update running mean and std."""
        try:
            o, r, done, info = step_result
        except ValueError:
            raise ValueError("Wrapping did not receive a valid input.")

        if update:
            self.update(o)

        # normalize
        if not isinstance(o, Tuple):
            o = (o,)


        normed_o = []
        for i, op in enumerate(filter(lambda a: len(a.shape) == 1, o)):
            normed_o.append(np.clip((op - self.mean[i]) / (np.sqrt(self.variance[i] + EPSILON)), -5., 5.))

        normed_o = normed_o[0] if len(normed_o) == 1 else tuple(normed_o)
        return normed_o, r, done, info

    def warmup(self, env: gym.Env, observations=10):
        """Warmup the wrapper by sampling the observation space."""
        for i in range(observations):
            self.update(flatten_goal_observation(env.observation_space.sample(), self.obs_names))

    def update(self, observation: Union[Tuple[np.ndarray], np.ndarray]) -> None:
        """Update the mean(s) and variance(s) of the tracked statistic based on the new sample.
        Simplification of https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.
        The method can handle multi input states where the observation is a tuple of numpy arrays. For each element
        a separate mean/variance is tracked. Any non-vector input will be skipped as they are assumed to be images
        and should be handled seperatly.
        """
        self.n += 1

        if not isinstance(observation, Tuple):
            observation = (observation,)

        for i, obs in enumerate(filter(lambda o: isinstance(o, (int, float)) or len(o.shape) in [0, 1], observation)):
            delta = obs - self.mean[i]
            m_a = self.variance[i] * (self.n - 1)

            self.mean[i] = np.array(self.mean[i] + delta * (1 / self.n), dtype=NP_FLOAT_PREC)
            self.variance[i] = np.array((m_a + np.square(delta) * (self.n - 1) / self.n) / self.n, dtype=NP_FLOAT_PREC)