import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state  # Useful dataclass to keep train state
from typing import Any, Dict, Tuple
from flax import struct
import optax
import jumanji
import jumanji.wrappers
import matplotlib.pyplot as plt
import flashbax as fbx
import flax
import collections
from functools import partial
import matplotlib.pyplot as plt
import chex
import neptune
from jumanji.environments.routing.connector.generator import Generator, RandomWalkGenerator
import hydra
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardValueNet as Critic
from jumanji.env import Environment
from mava.systems.ppo.types import LearnerState, OptStates, Params, PPOTransition
from omegaconf import DictConfig, OmegaConf
from mava.utils.training import make_learning_rate
from mava.utils import make_env as environments
from colorama import Fore, Style
import copy
import pprint
import time
from optax import OptState
from mava.evaluator import make_eval_fns
from mava.types import ActorApply, LearnerFn
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax import merge_leading_dims, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.total_timestep_checker import check_total_timesteps
from mava.wrappers.episode_metrics import get_final_step_metrics
from hydra import initialize, compose
from omegaconf import OmegaConf

def load_teacher(key, config):
    #config = compose(config_name='default_ff_ippo.yaml')
    #print(OmegaConf.to_yaml(cfg))
    #OmegaConf.set_struct(config, False)

    env, eval_env = environments.make(config)

    teacher_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    teacher_action_head = hydra.utils.instantiate(
        config.network.action_head, action_dim=env.action_dim
    )
    actor_network = Actor(torso=teacher_torso, action_head=teacher_action_head)

    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_network = Critic(torso=critic_torso)

    teacher_policy = actor_network.apply

    obs = env.observation_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], obs)

    key, actor_net_key, critic_net_key = jax.random.split(key, 3)

    # Initialise actor params
    actor_params = actor_network.init(actor_net_key, init_x)
    critic_params = critic_network.init(critic_net_key, init_x)
    params = Params(actor_params, critic_params)

    loaded_checkpoint = Checkpointer(
    model_name=config.logger.system_name,
    **config.logger.checkpointing.load_args,  # Other checkpoint args
    )
    # Restore the learner state from the checkpoint
    restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
    # Update the paramsp
    #teacher_params = restored_params
    teacher_params = restored_params.actor_params

    # EVALUATE TEACHER

    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size,) + x.shape)
    replicate_params = jax.tree_map(broadcast, teacher_params)

    # Duplicate learner across devices.
    replicate_params = flax.jax_utils.replicate(replicate_params, devices=jax.devices())

    evaluator, absolute_metric_evaluator = make_eval_fns(eval_env, actor_network, config)
    n_devices = len(jax.devices())

    logger = MavaLogger(config)

    start_time = time.time()

    trained_params = unreplicate_batch_dim(replicate_params)
    #trained_params = teacher_params
    key_e, *eval_keys = jax.random.split(key, n_devices + 1)
    eval_keys = jnp.stack(eval_keys)
    eval_keys = eval_keys.reshape(n_devices, -1)

    # Evaluate.
    evaluator_output = evaluator(trained_params, eval_keys)
    jax.block_until_ready(evaluator_output)

    # Log the results of the evaluation.
    elapsed_time = time.time() - start_time
    episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

    steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
    evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
    logger.log(evaluator_output.episode_metrics, 1,1, LogEvent.EVAL)
    
@hydra.main(config_path="./configs", config_name="default_ff_ippo.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    #eval_performance = run_experiment(cfg, buffer_state)
    key = jax.random.PRNGKey(cfg.system.seed)
    eval_performance = load_teacher(key, cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Behavior Cloning experiment completed{Style.RESET_ALL}")
    return eval_performance


hydra_entry_point()