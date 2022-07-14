from types import SimpleNamespace
from mava.components.jax.training.step import DefaultTrainerStep, MAPGWithTrustRegionStep
from mava.systems.jax.trainer import Trainer
import pytest
import time
import jax.numpy as jnp


def step_fn(sample:int):
    """ step_fn to test DefaultTrainerStep component
    
    Args:
        sample
    Returns:
        Dictionary
    """
    return {
        "sample":sample
    }

class MockTrainerLogger:
    """Mock of TrainerLogger to test DefaultTrainerStep component"""
    def __init__(self):
        self.written=None

    def write(self, results):
        self.written=results

class MockParameterClient():
    """Mock of ParameterClient to test DefaultTrainerStep component"""
    def __init__(self)->None:
        self.params={
            "trainer_steps":0,
            "trainer_walltime": -1,
        }
        self.call_set_and_get_async=False
    def add_async(self, params)->None:
        self.params=params
    
    def set_and_get_async(self)->None:
        self.call_set_and_get_async=True

class MockTrainer(Trainer):
    """Mock of Trainer"""
    def __init__(self) -> None:
        trainer_agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2",
        }
        networks = {
            "networks": {
                "network_agent_0": SimpleNamespace(params=jnp.array([0.0, 0.0, 0.0])),
                "network_agent_1": SimpleNamespace(params=jnp.array([1.0, 1.0, 1.0])),
                "network_agent_2": SimpleNamespace(params=jnp.array([2.0, 2.0, 2.0])),
            }
        }

        store=SimpleNamespace(
            dataset_iterator=iter([1,2,3]),
            step_fn=step_fn,
            timestamp=1657703548.5225394, #time.time() format
            trainer_parameter_client=MockParameterClient(),
            trainer_counts={
                "next_sample": 2
            },
            trainer_logger=MockTrainerLogger(),
            sample_batch_size=5,
            sequence_length=20,
            trainer_agent_net_keys=trainer_agent_net_keys,
            networks=networks,
            gae_fn=jnp.add
        )
        self.store=store

@pytest.fixture
def mock_trainer()->MockTrainer:
    """Build fixture from MockTrainer"""
    return MockTrainer()

def test_default_trainer_step_initiator()->None:
    """Test constructor of DefaultTrainerStep component"""
    trainer_step=DefaultTrainerStep()
    assert trainer_step.config.random_key==42

def test_on_training_step_with_timestamp(mock_trainer:Trainer)->None:
    """Test on_training_step method from TrainerStep case of existing timestamp"""
    trainer_step=DefaultTrainerStep()
    old_timestamp=mock_trainer.store.timestamp
    trainer_step.on_training_step(trainer=mock_trainer)

    assert int(mock_trainer.store.timestamp)==int(time.time())

    assert list(mock_trainer.store.trainer_parameter_client.params.keys())==["trainer_steps","trainer_walltime"]
    assert mock_trainer.store.trainer_parameter_client.params["trainer_steps"]==1
    elapsed_time=int(time.time()-old_timestamp)
    assert int(mock_trainer.store.trainer_parameter_client.params["trainer_walltime"])==elapsed_time

    assert mock_trainer.store.trainer_parameter_client.call_set_and_get_async==True

    assert mock_trainer.store.trainer_logger.written=={
        "next_sample": 2,
        "sample":1
    }

def test_on_training_step_without_timestamp(mock_trainer:Trainer)->None:
    """Test on_training_step method from TrainerStep case of no timestamp"""
    trainer_step=DefaultTrainerStep()
    del mock_trainer.store.timestamp
    trainer_step.on_training_step(trainer=mock_trainer)

    assert int(mock_trainer.store.timestamp)==int(time.time())

    assert list(mock_trainer.store.trainer_parameter_client.params.keys())==["trainer_steps","trainer_walltime"]
    assert mock_trainer.store.trainer_parameter_client.params["trainer_steps"]==1
    assert int(mock_trainer.store.trainer_parameter_client.params["trainer_walltime"])==0

    assert mock_trainer.store.trainer_parameter_client.call_set_and_get_async==True

    assert mock_trainer.store.trainer_logger.written=={
        "next_sample": 2,
        "sample":1
    }


def test_mapg_with_trust_region_step_initiator()->None:
    """Test constructor of MAPGWITHTrustRegionStep component"""
    mapg_with_trust_region_step=MAPGWithTrustRegionStep()
    assert mapg_with_trust_region_step.config.discount==0.99

def test_on_training_init_start(mock_trainer:MockTrainer):
    mapg_with_trust_region_step=MAPGWithTrustRegionStep()
    mapg_with_trust_region_step.on_training_init_start(trainer=mock_trainer)

    assert mock_trainer.store.full_batch_size== mock_trainer.store.sample_batch_size * (
            mock_trainer.store.sequence_length - 1
        )

def test_on_training_step_fn(mock_trainer:MockTrainer):
    """Test on_training_init_start method from MAPGWITHTrustRegionStep component"""
    mapg_with_trust_region_step=MAPGWithTrustRegionStep()
    del mock_trainer.store.step_fn
    mapg_with_trust_region_step.on_training_step_fn(trainer=mock_trainer)

    assert callable(mock_trainer.store.step_fn)

"""def test_step(mock_trainer:MockTrainer, mock_sample: MockSample)->None:
    mapg_with_trust_region_step=MAPGWithTrustRegionStep()
    del mock_trainer.store.step_fn
    mapg_with_trust_region_step.on_training_step_fn(trainer=mock_trainer)
"""
