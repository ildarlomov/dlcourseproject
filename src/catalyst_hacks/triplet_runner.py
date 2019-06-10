from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.experiments.experiment import BaseExperiment
from typing import Any, Mapping, Dict, List, Union
from catalyst.dl.callbacks.core import Callback
from torch import FloatTensor
from catalyst.dl.callbacks.base import OptimizerCallback, SchedulerCallback, CheckpointCallback


class TripletLossCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier

    def _add_loss_to_state(self, state, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state, criterion):
        targets = FloatTensor(state.output[self.output_key][0].size()).fill_(1)
        loss = criterion(
            state.output[self.output_key][0],
            state.output[self.output_key][1],
            targets
        )
        return loss

    def on_stage_start(self, state):
        assert state.criterion is not None

    def on_batch_end(self, state):
        criterion = state.get_key(
            key="criterion", inner_key=self.criterion_key
        )

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class TripletRunExperiment(BaseExperiment):

    def get_callbacks(self, stage: str) -> "List[Callback]":
        callbacks = self._callbacks
        if not stage.startswith("infer"):
            default_callbacks = [
                (self._criterion, TripletLossCallback),
                (self._optimizer, OptimizerCallback),
                (self._scheduler, SchedulerCallback),
                ("_default_saver", CheckpointCallback),
            ]

            for key, value in default_callbacks:
                is_already_present = any(
                    isinstance(x, value) for x in callbacks)
                if key is not None and not is_already_present:
                    callbacks.append(value())
        return callbacks


class TripletRunner(SupervisedRunner):
    _default_experiment = TripletRunExperiment
    def __init__(self, achor_key, pos_key, neg_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.achor_key = achor_key
        self.pos_key = pos_key
        self.neg_key = neg_key

    def predict_batch(self, batch: Mapping[str, Any]):
        output = self.model(batch[self.achor_key], batch[self.pos_key], batch[self.neg_key])
        if isinstance(output, dict):
            pass
        elif isinstance(output, (list, tuple)) \
                and isinstance(self.output_key, list):
            output = dict(
                (key, value) for key, value in zip(self.output_key, output))
        else:
            output = {self.output_key: output}
        return output

    def _batch2device(self, batch: Mapping[str, Any], device):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 2
            batch = {self.input_key: batch[0], self.target_key: batch[1]}
        batch = super()._batch2device(batch, device)
        return batch
