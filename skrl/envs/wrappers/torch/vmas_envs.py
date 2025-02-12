from typing import Any, Tuple

import gymnasium

import torch

from skrl import logger
from skrl.envs.wrappers.torch.base import Wrapper, MultiAgentEnvWrapper
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)
from vmas.simulator.environment.skrl.skrl_single_agent import SKRLSingleAgentWrapper
from vmas.simulator.environment.skrl.skrl import SKRLWrapper
from vmas.simulator.environment.skrl.skrl_single_agent_vec import SKRLSingleAgentVectorizedWrapper
from vmas.simulator.environment.skrl.skrl_vec import SKRLVectorizedWrapper


class VmasWrapper(Wrapper):
    """Vmas environment wrapper"""

    def __init__(self, env: Any) -> None:
        """Vmas environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Vmas environment
        """
        super().__init__(env)

        assert isinstance(env, SKRLSingleAgentWrapper) or isinstance(
            env, SKRLSingleAgentVectorizedWrapper
        ), f"Unsupported environment type: {type(env)}"
        self._vectorized = isinstance(env, SKRLSingleAgentVectorizedWrapper)
        self._env = env
        self._unwrapped = env.unwrapped
        if self._vectorized:
            self._reset_once = True
            self._observation = None
            self._info = None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        if self._vectorized:
            return self._env.single_observation_space
        return self._env.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        if self._vectorized:
            return self._env.single_action_space
        return self._env.action_space

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        observation, reward, terminated, truncated, info = self._env.step(actions)

        # save the observation and info for vectorized environments
        if self._vectorized:
            self._observation = observation
            self._info = info

        return observation, reward, terminated, truncated, info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """

        if self._vectorized:
            if self._reset_once:
                observation, self._info = self._env.reset()
                self._observation = observation
                self._reset_once = False
            return self._observation, self._info

        observation, info = self._env.reset()
        return observation, info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        frame = self._env.render(mode="rgb_array", visualize_when_rgb=True, **kwargs)

        return frame

    def close(self) -> None:
        """Close the environment"""
        self._env.close()


class VmasMultiAgentWrapper(MultiAgentEnvWrapper):
    """Vmas multi-agent environment wrapper"""

    def __init__(self, env: Any) -> None:
        """Vmas multi-agent environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Vmas environment
        """
        super().__init__(env)
        assert isinstance(env, SKRLVectorizedWrapper) or isinstance(
            env, SKRLWrapper
        ), f"Unsupported environment type: {type(env)}"
        self._env = env
        self._unwrapped = env.unwrapped

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        return flatten_tensorized_space(self._env.observation_space)

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        return flatten_tensorized_space(self._env.action_space)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        observation, reward, terminated, info = self._env.step(actions)
        truncated = torch.zeros_like(terminated)
        return observation, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        observation, info = self._env.reset()
        return observation, info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        frame = self._env.render(mode="rgb_array", visualize_when_rgb=True, **kwargs)
        return frame
