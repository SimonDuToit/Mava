from typing import Tuple

import chex
import jax
from jumanji.env import State
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from functools import cached_property
from mava.types import Observation, ObservationGlobalState

class CentralizedWrapper(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)
        #self.num_agents = self._env.num_agents
        #self.time_limit = self._env._env.time_limit

        # https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
        def cartesian_product(*arrays):
            la = len(arrays)
            dtype = jnp.result_type(*arrays)
            arr = jnp.empty([len(a) for a in arrays] + [la], dtype=dtype)
            for i, a in enumerate(jnp.ix_(*arrays)):
                arr = arr.at[...,i].set(a)
            return arr.reshape(-1, la)

        #self.num_actions = self.observation_spec().action_mask.shape[-1]
        self.num_actions = self._env.action_spec().num_values[0]
        num_agents = self.num_agents
        action_space = [jnp.arange(self.num_actions)]*num_agents
        self.combined_action_space = cartesian_product(*action_space)

    def _modify_action_mask(self, action_mask: chex.Array) -> chex.Array:
        jax.vmap(lambda x: jnp.all(action_mask[jnp.arange(len(x)), x]))(self.combined_action_space)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        #print(self._env.get_global_state(timestep.observation.agents_view).shape)
        agents_view = timestep.observation.global_state[jnp.newaxis, 0]        
        #agents_view = self._env.get_global_state(timestep.observation)[jnp.newaxis, 0]
        action_mask = jax.vmap(lambda x: jnp.all(timestep.observation.action_mask[jnp.arange(len(x)), x]))(self.combined_action_space)
        action_mask = action_mask[jnp.newaxis, :]
        step_count = timestep.observation.step_count[jnp.newaxis, 0]
        obs = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

        reward = timestep.reward[jnp.newaxis, 0]
        discount = timestep.discount[jnp.newaxis, 0]
        return timestep.replace(observation = obs, reward=reward, discount=discount)

    
    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        #print(timestep.observation.global_state.shape)
        timestep = self.modify_timestep(timestep)
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        action = self.combined_action_space[action][0]
        #print(action.shape)
        state, timestep = self._env.step(state, action)
        timestep = self.modify_timestep(timestep)
        return state, timestep

    
    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (1,),
            int,
            jnp.array(0),#(1, dtype=int),
            jnp.array(self.time_limit),
            "step_count"
        )

        action_mask = specs.BoundedArray(
            shape=(1, self.num_actions ** self.num_agents),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )

        obs_spec = self._env.observation_spec()
        return specs.Spec(
                Observation,
                "ObservationSpec",
                agents_view=obs_spec["global_state"],
                #action_mask=obs_spec["action_mask"],
                action_mask = action_mask,
                step_count=step_count,
        )
    
    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec for the Connector environment.

        5 actions: [0,1,2,3,4] -> [No Op, Up, Right, Down, Left]. Since this is an environment with
        a multi-dimensional action space, it expects an array of actions of shape (num_agents,).

        Returns:
            observation_spec: `MultiDiscreteArray` of shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array(self.num_actions ** self.num_agents),
            dtype=jnp.int32,
            name="action",
        )
    
    @cached_property
    def action_dim(self) -> chex.Array:
        "Get the actions dim for each agent."
        return int(self.num_actions ** self.num_agents)
