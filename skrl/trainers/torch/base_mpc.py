from typing import List, Optional, Union

import atexit
import sys
import tqdm

import torch

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer


class TrainerMPC(Trainer):
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Base class for trainers

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``)
        :type cfg: dict, optional
        """
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=cfg)

    def train(self) -> None:
        """Train the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def eval(self) -> None:
        """Evaluate the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def single_agent_train(self) -> None:
        """Train agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # get MPC states
                mpc_states = self.env.mpc_state()

                # compute actions
                actions = self.agents.act(states, mpc_states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    mpc_states=mpc_states,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def single_agent_eval(self) -> None:
        """Evaluate agent

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # get MPC states
                mpc_states = self.env.mpc_state()

                # compute actions
                outputs = self.agents.act(states, mpc_states, timestep=timestep, timesteps=self.timesteps)
                actions = outputs[0] if self.stochastic_evaluation else outputs[-1].get("mean_actions", outputs[0])

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    mpc_states=mpc_states,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def multi_agent_train(self) -> None:
        """Train multi-agents

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                mpc_states = self.env.mpc_state()

                # compute actions
                actions = self.agents.act(states, mpc_states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    mpc_states=mpc_states,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states

    def multi_agent_eval(self) -> None:
        """Evaluate multi-agents

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # get MPC states
                mpc_states = self.env.mpc_state()
                # compute actions
                outputs = self.agents.act(states, mpc_states, timestep=timestep, timesteps=self.timesteps)
                actions = (
                    outputs[0]
                    if self.stochastic_evaluation
                    else {k: outputs[-1][k].get("mean_actions", outputs[0][k]) for k in outputs[-1]}
                )

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    mpc_states=mpc_states,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states
