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

    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size,) + x.shape)
    replicate_params = jax.tree_map(broadcast, teacher_params)
    replicate_params = flax.jax_utils.replicate(replicate_params, devices=jax.devices())

    # Initialize Env info

    n_devices = len(jax.devices())

    key, env_key = jax.random.split(key)
    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
        #key, n_devices * config.system.update_batch_size + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
        #(n_devices, config.system.update_batch_size) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_map(reshape_states, env_states)
    timesteps = jax.tree_map(reshape_states, timesteps)

    # Define policy function

    def get_action_and_logits(params, obs):
        output = teacher_policy(params, obs)
        logits = output.distribution.logits
        action = output.sample(seed=key)
        return action, logits

    batched_get_action_and_logits = jax.pmap(jax.vmap(get_action_and_logits))

    # Construct dataset

    NUM_STEPS = 100_000

    # Flashbax buffer
    max_length = NUM_STEPS
    min_length = 1
    sample_batch_size = 1

    add_sequences = False
    add_batch_size = None

    # Instantiate the flat buffer, which is a Dataclass of pure functions.
    buffer = fbx.make_item_buffer(max_length, min_length, sample_batch_size, False, False)

    init_logits = batched_get_action_and_logits(replicate_params, timesteps.observation)[1]
    buffer_state = buffer.init((timesteps.observation, init_logits))

    # Generate D with teacher
    @jax.jit
    def env_step(carry):
        """Step the environment."""
        # SELECT ACTION
        i, key, env_state, last_timestep, buffer_state = carry
        i = i + 1
        action, logits = batched_get_action_and_logits(replicate_params, last_timestep.observation)

        # STEP ENVIRONMENT
        env_state, timestep = jax.pmap(jax.vmap(jax.vmap(env.step)))(env_state, action)

        transition = (
            last_timestep.observation, logits
        )
        buffer.add(buffer_state, transition)
        carry = i, key, env_state, timestep, buffer_state
        return carry

    step_init = 0, key, env_states, timesteps, buffer_state
    #jax.lax.scan(env_step, step_init, None, NUM_STEPS)
    #jax.lax.while_loop(lambda x: x[0] < NUM_STEPS, env_step, step_init)
    
    carry = step_init
    for i in range(NUM_STEPS):
        carry = env_step(carry)
    return buffer, buffer_state


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
):
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    config.system.num_agents = env.num_agents
    #config.system.num_agents = 1 

    # PRNG keys.
    key, actor_net_key = keys

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.action_head, action_dim=env.action_dim
    )

    network = Actor(torso=actor_torso, action_head=actor_action_head)

    actor_lr = make_learning_rate(config.system.actor_lr, config)

    optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )

    # Initialise observation with obs of all agents.
    obs = env.observation_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], obs)

    # Initialise actor params and optimiser state.
    params = network.init(actor_net_key, init_x)
    opt_state = optim.init(params)

    # Pack apply and update functions.
    apply_fn = network.apply
    update_fn = optim.update

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fn, update_fn, config)
    learn = jax.pmap(learn, in_axes=(0, 0), axis_name="device")

    '''
    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params
    '''
    # Define params to be replicated across devices and batches.
    key, step_keys = jax.random.split(key)  # noqa: E999
    replicate_learner = (params, opt_state, step_keys)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_state, step_keys = replicate_learner
    init_learner_state = (params, opt_state, step_keys)

    return learn, network, init_learner_state

def get_learner_fn(
    env: Environment,
    apply_fn,
    update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""


    def _update_step(learner_state: Any, batch: Any) -> Tuple[LearnerState, Tuple]:

        # Sample experience batch
        params, opt_state, key = learner_state
        
        #print(batch.experience[1])
        obs, target = batch.experience

        '''
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )
        '''

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_state, key = train_state
                obs, target = batch_info

                '''
                def _loss_fn(
                    params: FrozenDict,
                    actor_opt_state: OptState,
                    obs,
                    outputs: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK
                    policy = apply_fn(params, obs)

                    # CALCULATE ACTOR LOSS
                    loss_actor = optax.softmax_cross_entropy(outputs)
                    # The seed will be used in the TanhTransformedDistribution:
                    entropy = policy.entropy(seed=key).mean()
                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    return total_loss_actor, (loss_actor, entropy)
                '''

                # CALCULATE ACTOR LOSS
                key, entropy_key = jax.random.split(key)
                '''
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params,
                    opt_states.actor_opt_state,
                    traj_batch,
                    advantages,
                    entropy_key,
                )
                '''

                def get_action_and_logits(params, obs):
                    #print(params["params"]["action_head"]["Dense_0"]["kernel"].shape)
                    #print(obs.agents_view.shape)
                    output = apply_fn(params, obs)
                    logits = output.distribution.logits
                    action = output.sample(seed=key)
                    return action, logits
                #batched_get_action_and_logits = jax.pmap(jax.vmap(get_action_and_logits))

                #output = apply_fn(params, obs)
                #print(params["params"]["action_head"]["Dense_0"]["kernel"].shape)
                #print(obs.agents_view.shape)

                def loss_fn(params, obs, target):
                    action, logits = get_action_and_logits(params, obs)
                    return optax.softmax_cross_entropy(logits, target).mean()
                
                loss_info, grads = jax.value_and_grad(loss_fn)(params, obs, target)


                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                grads, loss_info = jax.lax.pmean(
                    (grads, loss_info), axis_name="batch"
                )
                # pmean over devices.
                grads, loss_info = jax.lax.pmean(
                    (grads, loss_info), axis_name="device"
                )

                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                updates, new_opt_state = update_fn(
                    grads, opt_state
                )
                #print(params["params"]["action_head"]["Dense_0"]["kernel"].shape)
                #print(updates["params"]["action_head"]["Dense_0"]["kernel"].shape)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_opt_state, entropy_key), loss_info

            params, opt_state, obs, target, key = update_state
            #key, shuffle_key, entropy_key = jax.random.split(key, 3)

            # (don't) SHUFFLE MINIBATCHES
            batch = (obs, target)
            
            '''
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                batch,
            )
            '''
            # UPDATE MINIBATCHES
            key, entropy_key = jax.random.split(key)
            #(params, opt_state, entropy_key), loss_info = jax.lax.scan(
            #    _update_minibatch, (params, opt_state, entropy_key), batch #minibatches
            #)
            (params, opt_state, entropy_key), loss_info = _update_minibatch((params, opt_state, entropy_key), batch)
            update_state = (params, opt_state, obs, target, key)
            return update_state, loss_info
        
        update_state = (params, opt_state, obs, target, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_state, obs, target, key = update_state
        learner_state =(params, opt_state, key)
        #metric = traj_batch.info #??
        return (learner_state, batch), (loss_info)
        #return learner_state, (metric, loss_info)
    
    def learner_fn(learner_state: Any, batch: Any) -> Tuple:
            """Learner function.

            This function represents the learner, it updates the network parameters
            by iteratively applying the `_update_step` function for a fixed number of
            updates. The `_update_step` function is vectorized over a batch of inputs.

            Args:
                learner_state (NamedTuple):
                    - params (Params): The initial model parameters.
                    - opt_states (OptStates): The initial optimizer state.
                    - key (chex.PRNGKey): The random number generator state.
                    - env_state (LogEnvState): The environment state.
                    - timesteps (TimeStep): The initial timestep in the initial trajectory.
            """

            batched_update_step = jax.vmap(_update_step, in_axes=(0, 0), axis_name="batch")

            def scanned_batched_update_step(carry, _):
                return batched_update_step(carry[0], carry[1])
   
            (learner_state, batch), (loss_info) = jax.lax.scan(
                scanned_batched_update_step, (learner_state, batch), None, config.system.num_updates_per_eval
            )

            print("yippee")
            return (
                learner_state,
                #episode_metrics=episode_info,
                loss_info,
            )

    return learner_fn


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    key = jax.random.PRNGKey(config.system.seed)
    buffer, buffer_state = load_teacher(key, config)

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config)

    # PRNG keys.
    key, key_e, actor_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=3
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key), config
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    evaluator, absolute_metric_evaluator = make_eval_fns(eval_env, actor_network, config)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = MavaLogger(config)
    #cfg: Dict = OmegaConf.to_container(config, resolve=True)
    #cfg["arch"]["devices"] = jax.devices()
    #pprint(cfg)

    # Set up checkpointer
    '''
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )
    '''

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()
        key, batch_key = jax.random.split(key)
        batch = buffer.sample(buffer_state, batch_key)

        def inner(x):
            return x[0]
        batch = jax.tree_map(inner, batch)

        params, opt_state, keys = learner_state
        learner_output = learn(learner_state, batch)
        
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        #episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        #episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        #if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
        #    logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output[1], t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        
        trained_params = unreplicate_batch_dim(params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
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
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        '''
        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )
        '''

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state, loss_info = learner_output

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        t = int(steps_per_rollout * (eval_step + 1))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance
    
    
@hydra.main(config_path="./configs", config_name="default_ff_ippo.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Behavior Cloning experiment completed{Style.RESET_ALL}")
    return eval_performance


hydra_entry_point()