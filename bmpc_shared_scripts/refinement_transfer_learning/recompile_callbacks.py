from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

@dataclass(frozen=True)
class RecompileCallback:
    epoch: int
    callback: Callable[[], None]

@dataclass(frozen=True)
class TrainingPart:
    num_epochs: int
    callbacks: list[Callable[[], None]]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for cb in self.callbacks:
            cb()

def get_training_parts(recompile_callbacks : list[RecompileCallback], total_epochs) -> list[TrainingPart]:
    assert not any([x.epoch > total_epochs for x in recompile_callbacks])
    rcb_dict = {}
    for rcb in recompile_callbacks:
        rcb_dict.setdefault(rcb.epoch, [])
        rcb_dict[rcb.epoch].append(rcb.callback)
    rcb_keys = sorted([int(x) for x in rcb_dict])
    current_epoch = 0
    training_parts : list[TrainingPart] = []
    for epoch_key in rcb_keys:
        training_parts.append(TrainingPart(
            num_epochs=epoch_key - current_epoch,
            callbacks=rcb_dict[epoch_key]
        ))
        current_epoch = epoch_key

    return training_parts