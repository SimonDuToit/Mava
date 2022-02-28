# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

# TODO (Arnu): remove once we figured out the no attribute error.
# type: ignore

"""Abstract mixin class used to call system component hooks."""

from abc import ABC


class CallbackHookMixin(ABC):

    ######################
    # system builder hooks
    ######################

    # INITIALISATION
    def on_building_init_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_init_start(self)

    def on_building_init(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_init(self)

    def on_building_init_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_init_end(self)

    # DATA SERVER
    def on_building_data_server_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_data_server_start(self)

    def on_building_data_server_adder_signature(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_data_server_adder_signature(self)

    def on_building_data_server_rate_limiter(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_data_server_rate_limiter(self)

    def on_building_data_server(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_data_server(self)

    def on_building_data_server_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_data_server_end(self)

    # PARAMETER SERVER
    def on_building_parameter_server_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_parameter_server_start(self)

    def on_building_parameter_server(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_parameter_server_make_parameter_server(self)

    def on_building_parameter_server_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_parameter_server_end(self)

    # EXECUTOR
    def on_building_executor_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_start(self)

    def on_building_executor_adder_priority(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_adder_priority(self)

    def on_building_executor_adder(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_adder(self)

    def on_building_executor_logger(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_logger(self)

    def on_building_executor_parameter_client(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_parameter_client(self)

    def on_building_executor(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor(self)

    def on_building_executor_environment(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_environment(self)

    def on_building_executor_environment_loop(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_train_loop(self)

    def on_building_executor_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_end(self)

    # TRAINER
    def on_building_trainer_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_start(self)

    def on_building_trainer_logger(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_logger(self)

    def on_building_trainer_dataset(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_dataset(self)

    def on_building_trainer_parameter_client(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_parameter_client(self)

    def on_building_trainer(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer(self)

    def on_building_trainer_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_end(self)

    # BUILD
    def on_building_program_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_start(self)

    def on_building_program_nodes(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_program_nodes(self)

    def on_building_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_end(self)

    # LAUNCH
    def on_building_launch_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_launch_start(self)

    def on_building_launch(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_launch(self)

    def on_building_launch_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_launch_end(self)
