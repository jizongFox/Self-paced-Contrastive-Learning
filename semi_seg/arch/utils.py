from typing import Union

from torch.nn import Parameter, Module


def get_requires_grad(input_: Union[Parameter, Module]) -> bool:
    """this just check the first element of the module, thus causing errors if inconsistency occurs"""
    assert isinstance(input_, (Parameter, Module)), type(input_)
    if isinstance(input_, Module):
        return next(input_.parameters()).requires_grad
    return input_.requires_grad


def get_bn_track(input_: Module) -> bool:
    """this just check the first bn submodule of the module, thus causing errors if inconsistency occurs"""
    if hasattr(input_, "track_running_stats"):
        return input_.track_running_stats
    for m in input_.modules():
        if hasattr(m, "track_running_stats"):
            return m.track_running_stats
    raise RuntimeError(f"BN module not found in {input_}")
