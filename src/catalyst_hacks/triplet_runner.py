from catalyst.dl.experiments import SupervisedRunner
from typing import Any, Mapping, Dict, List, Union

class TripletRunner(SupervisedRunner):

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