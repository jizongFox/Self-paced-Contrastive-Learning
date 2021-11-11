from torch import nn

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.meters import AverageValueMeter, MeterInterface
from .utils import meter_focus


class ConsistencyTrainerHook(TrainerHook):

    def __init__(self, name: str, weight: float):
        super().__init__(name)
        self._weight = weight
        self._criterion = nn.MSELoss()

    def __call__(self):
        return _ConsistencyEpocherHook(name=self._hook_name, weight=self._weight, criterion=self._criterion)


class _ConsistencyEpocherHook(EpocherHook):
    def __init__(self, name: str, weight: float, criterion) -> None:
        super().__init__(name)
        self._weight = weight
        self._criterion = criterion

    @meter_focus
    def configure_meters(self, meters: MeterInterface):
        self.meters.register_meter("loss", AverageValueMeter())

    @meter_focus
    def __call__(self, *, unlabeled_tf_logits, unlabeled_logits_tf, seed, affine_transformer, **kwargs):
        unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
        unlabeled_prob_tf = unlabeled_logits_tf.softmax(1)
        loss = self._criterion(unlabeled_prob_tf.detach(), unlabeled_tf_prob)
        self.meters["loss"].add(loss.item())
        return self._weight * loss


if __name__ == '__main__':
    hook = ConsistencyTrainerHook(name="1", weight=10)
    print(list(hook.parameters()))
