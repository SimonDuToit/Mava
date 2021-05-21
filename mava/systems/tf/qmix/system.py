# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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

"""Defines the QMIX system class."""
import functools
from typing import Any, Callable, Dict, Optional, Type

import dm_env
import reverb
import sonnet as snt
from acme import specs as acme_specs
from acme.utils import counting

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedValueActor
from mava.components.tf.modules import mixing
from mava.components.tf.modules.exploration import LinearExplorationScheduler
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf.madqn.system import MADQN
from mava.systems.tf.qmix import builder, execution, training
from mava.utils.loggers import MavaLogger, logger_utils


# TODO Correct documentation
# TODO Implement recurrent option from MADQN
class QMIX(MADQN):
    """QMIX system.
    This implements a single-process QMIX system.
    Args:
        environment_factory: Callable to instantiate an environment on a compute node.
        network_factory: Callable to instantiate system networks on a compute node.
        logger_factory: Callable to instantiate a system logger on a compute node.
        architecture: system architecture, e.g. decentralised or centralised.
        trainer_fn: training type associated with executor and architecture,
            e.g. centralised training.
        executor_fn: executor type for example feedforward or recurrent.
        num_executors: number of executor processes to run in parallel.
        num_caches: number of trainer node caches.
        environment_spec: description of the actions, observations, etc.
        q_networks: the online Q network (the one being optimized)
        epsilon: probability of taking a random action; ignored if a policy
            network is given.
        trainer_fn: the class used for training the agent and mixing networks.
        shared_weights: boolean determining whether shared weights is used.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        clipping: whether to clip gradients by global norm.
        replay_table_name: string indicating what name to give the replay table.
        max_replay_size: maximum replay size.
        samples_per_insert: number of samples to take from replay for every insert
            that is made.
        prefetch_size: size to prefetch from replay.
        batch_size: batch size for updates.
        n_step: number of steps to squash into a single transition.
        discount: discount to use for TD updates.
        counter: counter object used to keep track of steps.
        checkpoint: boolean indicating whether to checkpoint the learner.
    """

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[DecentralisedValueActor] = DecentralisedValueActor,
        trainer_fn: Type[training.QMIXTrainer] = training.QMIXTrainer,
        executor_fn: Type[core.Executor] = execution.QMIXFeedForwardExecutor,
        mixer: Type[mixing.BaseMixingModule] = mixing.MonotonicMixing,
        exploration_scheduler_fn: Type[
            LinearExplorationScheduler
        ] = LinearExplorationScheduler,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 1e-4,
        num_executors: int = 1,
        num_caches: int = 0,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = False,
        batch_size: int = 256,
        prefetch_size: int = 4,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000,
        samples_per_insert: Optional[float] = 32.0,
        n_step: int = 5,
        sequence_length: int = 20,
        period: int = 20,
        clipping: bool = True,
        discount: float = 0.99,
        optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=1e-4),
        target_update_period: int = 100,
        executor_variable_update_period: int = 1000,
        max_executor_steps: int = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        logger_config: Dict = {},
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
    ):

        self._mixer = mixer

        if not environment_spec:
            environment_spec = mava_specs.MAEnvironmentSpec(
                environment_factory(evaluation=False)  # type:ignore
            )

        # set default logger if no logger provided
        if not logger_factory:
            logger_factory = functools.partial(
                logger_utils.make_logger,
                directory="~/mava",
                to_terminal=True,
                time_delta=10,
            )

        super(QMIX, self).__init__(
            architecture=architecture,
            environment_factory=environment_factory,
            network_factory=network_factory,
            logger_factory=logger_factory,
            environment_spec=environment_spec,
            shared_weights=shared_weights,
            num_executors=num_executors,
            num_caches=num_caches,
            max_executor_steps=max_executor_steps,
            checkpoint_subpath=checkpoint_subpath,
            checkpoint=checkpoint,
            train_loop_fn=train_loop_fn,
            train_loop_fn_kwargs=train_loop_fn_kwargs,
            eval_loop_fn=eval_loop_fn,
            eval_loop_fn_kwargs=eval_loop_fn_kwargs,
        )

        if issubclass(executor_fn, executors.RecurrentExecutor):
            extra_specs = self._get_extra_specs()
        else:
            extra_specs = {}

        self._builder = builder.QMIXBuilder(
            builder.QMIXConfig(
                environment_spec=environment_spec,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
                shared_weights=shared_weights,
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                executor_variable_update_period=executor_variable_update_period,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                n_step=n_step,
                sequence_length=sequence_length,
                period=period,
                clipping=clipping,
                checkpoint=checkpoint,
                optimizer=optimizer,
                checkpoint_subpath=checkpoint_subpath,
            ),
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
            exploration_scheduler_fn=exploration_scheduler_fn,
        )

    def trainer(
        self,
        replay: reverb.Client,
        counter: counting.Counter,
    ) -> mava.core.Trainer:
        """The Trainer part of the system. Train with mixing networks."""
        # Create the networks to optimize (online)
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec
        )

        # Create system architecture
        architecture = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            shared_weights=self._shared_weights,
        )

        agent_networks = architecture.create_actor_variables()

        # Augment network architecture by adding mixing layer network.
        system_networks = self._mixer(
            architecture=architecture,
            environment_spec=self._environment_spec,
            agent_networks=agent_networks,
        ).create_system()

        # create logger
        trainer_logger_config = {}
        if self._logger_config and "trainer" in self._logger_config:
            trainer_logger_config = self._logger_config["trainer"]
        trainer_logger = self._logger_factory(  # type: ignore
            "trainer", **trainer_logger_config
        )

        dataset = self._builder.make_dataset_iterator(replay)
        counter = counting.Counter(counter, "trainer")

        return self._builder.make_trainer(
            networks=system_networks,
            dataset=dataset,
            counter=counter,
            logger=trainer_logger,
        )

    def build(self, name: str = "qmix") -> Any:
        """Build the distributed system topology."""
        return super().build(name=name)
