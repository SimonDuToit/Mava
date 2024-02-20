# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.lbf import LevelBasedForaging
from jumanji.environments.routing.robot_warehouse import RobotWarehouse
from jumanji.environments.routing.connector import MaConnector
from jumanji.environments.routing.cleaner import Cleaner
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import Observation, State, ObservationGlobalState


class MultiAgentWrapper(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)
        self._num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for `step` and `reset`."""
        pass

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        return state, self.modify_timestep(timestep)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        return state, self.modify_timestep(timestep)

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self._num_agents,),
            jnp.int32,
            [0] * self._num_agents,
            [self.time_limit] * self._num_agents,
            "step_count",
        )
        return self._env.observation_spec().replace(step_count=step_count)


class RwareWrapper(MultiAgentWrapper):
    """Multi-agent wrapper for the Robotic Warehouse environment."""

    def __init__(self, env: RobotWarehouse):
        super().__init__(env)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for the Robotic Warehouse environment."""
        observation = Observation(
            agents_view=timestep.observation.agents_view,
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self._num_agents),
        )
        print(observation.agents_view.size)
        reward = jnp.repeat(timestep.reward, self._num_agents)
        discount = jnp.repeat(timestep.discount, self._num_agents)
        return timestep.replace(observation=observation, reward=reward, discount=discount)


class LbfWrapper(MultiAgentWrapper):
    """
     Multi-agent wrapper for the Level-Based Foraging environment.

    Args:
        env (Environment): The base environment.
        use_individual_rewards (bool): If true each agent gets a separate reward,
        sum reward otherwise.
    """

    def __init__(self, env: LevelBasedForaging, use_individual_rewards: bool = False):
        super().__init__(env)
        self._env: LevelBasedForaging
        self._use_individual_rewards = use_individual_rewards

    def aggregate_rewards(
        self, timestep: TimeStep, observation: Observation
    ) -> TimeStep[Observation]:
        """Aggregate individual rewards across agents."""
        team_reward = jnp.sum(timestep.reward)

        # Repeat the aggregated reward for each agent.
        reward = jnp.repeat(team_reward, self._num_agents)
        return timestep.replace(observation=observation, reward=reward)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for Level-Based Foraging environment and update
        the reward based on the specified reward handling strategy."""

        # Create a new observation with adjusted step count
        modified_observation = Observation(
            agents_view=timestep.observation.agents_view,
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self._num_agents),
        )
        
        if self._use_individual_rewards:
            # The environment returns a list of individual rewards and these are used as is.
            return timestep.replace(observation=modified_observation)

        # Aggregate the list of individual rewards and use a single team_reward.
        return self.aggregate_rewards(timestep, modified_observation)


class ConnectorWrapper(MultiAgentWrapper):
    """Multi-agent wrapper for the MA Connector environment."""

    def __init__(self, env: MaConnector):
        super().__init__(env)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for the Connector environment."""

        def convert_obs(grid):
            positions = jnp.where(grid % 3 == 2, True, False)   
            targets = jnp.where((grid % 3 == 0) & (grid != 0), True, False)
            paths = jnp.where(grid % 3 == 1, True, False)
            my_position = jnp.where(grid == 2, True, False)
            my_target = jnp.where(grid == 3, True, False)
            agents_view = jnp.stack((positions, targets, paths, my_position, my_target), -1)
            return agents_view
        
        def convert_global(grid):
            positions = jnp.where(grid % 3 == 2, True, False)   
            targets = jnp.where((grid % 3 == 0) & (grid[0] != 0), True, False)
            paths = jnp.where(grid % 3 == 1, True, False)
            global_state = jnp.stack((positions, targets, paths), -1)
            return global_state
        
        observation = ObservationGlobalState(
            global_state = convert_global(timestep.observation.grid),
            agents_view= convert_obs(timestep.observation.grid),
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self._num_agents),
        ) 
        return timestep.replace(observation=observation)
    
    
    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self._num_agents,),
            jnp.int32,
            [0] * self._num_agents,
            [self.time_limit] * self._num_agents,
            "step_count",
        )

        agents_view = specs.BoundedArray(
            shape=(self._env.num_agents, self._env.grid_size, self._env.grid_size, 5),
            dtype=jnp.bool,
            name="agents_view",
            minimum=False,
            maximum=True,
        )

        global_state = specs.BoundedArray(
            shape=(self._env.num_agents, self._env.grid_size, self._env.grid_size, 3),
            dtype=jnp.bool,
            name="global_state",
            minimum=False,
            maximum=True,
        )

        spec = specs.Spec(
            ObservationGlobalState,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=self._env.observation_spec().action_mask,
            global_state=global_state,
            step_count=step_count,
        )
        return spec
    

class CleanerWrapper(MultiAgentWrapper):
    """Multi-agent wrapper for the Cleaner environment."""

    def __init__(self, env: Cleaner):
        super().__init__(env)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for the Cleaner environment."""

        def convert_obs(obs):
            DIRTY = 0
            WALL = 2
            agents_locations = obs.agents_locations
            num_agents = agents_locations.shape[0]
            grid = obs.grid
            dirty_channel = jnp.tile(jnp.where(grid == DIRTY, 1, 0), (num_agents, 1, 1))
            wall_channel = jnp.tile(jnp.where(grid == WALL, 1, 0), (num_agents, 1, 1))
            xs, ys = agents_locations[:, 0], agents_locations[:, 1] 
            pos_per_agent = jnp.repeat(jnp.zeros_like(grid)[None, :, :], num_agents, axis=0)
            pos_per_agent = pos_per_agent.at[jnp.arange(num_agents), xs, ys].set(1)
            #agents_pos = jnp.tile(pos_per_agent.T, (num_agents, 1, 1, 1))
            agents_pos = jnp.tile(jnp.sum(pos_per_agent, axis=0), (num_agents, 1, 1))

            transformed = jnp.stack(
                [dirty_channel, wall_channel, agents_pos, pos_per_agent], axis=-1
                #[dirty_channel, wall_channel, pos_per_agent], axis=-1
            )
            return transformed
            #return jnp.concatenate((transformed, agents_pos), -1)
        
        def convert_global(obs):
            DIRTY = 0
            WALL = 2
            agents_locations = obs.agents_locations
            num_agents = agents_locations.shape[0]
            grid = obs.grid
            dirty_channel = jnp.tile(jnp.where(grid == DIRTY, 1, 0), (num_agents, 1, 1))
            wall_channel = jnp.tile(jnp.where(grid == WALL, 1, 0), (num_agents, 1, 1))

            
            xs, ys = agents_locations[:, 0], agents_locations[:, 1] 
            pos_per_agent = jnp.repeat(jnp.zeros_like(grid)[None, :, :], num_agents, axis=0)
            pos_per_agent = pos_per_agent.at[jnp.arange(num_agents), xs, ys].set(1)
            agents_pos = jnp.tile(jnp.sum(pos_per_agent, axis=0), (num_agents, 1, 1))

            return jnp.stack(
                (dirty_channel, wall_channel, agents_pos), axis=-1
            )
        
        agents_view = convert_obs(timestep.observation)
        observation = ObservationGlobalState(
            global_state = convert_global(timestep.observation),
            agents_view=agents_view,
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self._num_agents),
        ) 
        return timestep.replace(observation=observation, reward = jnp.repeat(timestep.reward, self.num_agents))
    
    
    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self._num_agents,),
            jnp.int32,
            [0] * self._num_agents,
            [self.time_limit] * self._num_agents,
            "step_count",
        )

        agents_view = specs.BoundedArray(
            shape=(self._env.num_agents, self._env.num_rows, self._env.num_cols, 4),
            #shape=(self._env.num_agents, self._env.num_rows, self._env.num_cols, self._env.num_agents + 3),
            dtype=jnp.bool,
            name="agents_view",
            minimum=0,
            maximum=self._env.num_agents,
        )

        global_state = specs.BoundedArray(
            shape=(self._env.num_agents, self._env.num_rows, self._env.num_cols, 3),
            dtype=jnp.bool,
            name="agents_view",
            minimum=0,
            maximum=self._env.num_agents,
        )

        spec = specs.Spec(
            ObservationGlobalState,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=self._env.observation_spec().action_mask,
            global_state=global_state,
            step_count=step_count,
        )
        return spec