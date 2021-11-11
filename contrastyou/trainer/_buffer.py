from collections import OrderedDict, namedtuple
from copy import deepcopy
from typing import Union
from warnings import warn

import numpy as np
import torch
from torch import Tensor

from ..types import typeNumeric, is_numeric, typePath, is_path


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


class _BufferMixin:
    """
    The buffer in Trainer is for automatic loading and saving.
    """

    def __init__(self) -> None:
        self._buffers: OrderedDict = OrderedDict()

    def _register_buffer(self, name: str, value: Union[typePath, typeNumeric] = None):
        r"""Adds a persistent buffer to the module.
        """
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(
                "buffer name should be a string. " "Got {}".format(torch.typename(name))
            )
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        else:
            if is_path(value):
                value = str(value)
            elif is_numeric(value):
                value = value
            elif value is None:
                value = None
            else:
                raise TypeError(value)

            self._buffers[name] = value

    def __getattr__(self, name):
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        buffers = self.__dict__.get("_buffers")
        if buffers is not None and name in buffers:
            buffers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._buffers:
            del self._buffers[name]
        else:
            object.__delattr__(self, name)

    def _save_to_state_dict(self, destination, prefix):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, buf in self._buffers.items():
            value = buf
            if isinstance(buf, Tensor):
                value = buf.detach()
            if isinstance(buf, np.ndarray):
                value = deepcopy(buf)
            if buf is None:
                value = None
            destination[prefix + name] = value

    def state_dict(self, destination=None, prefix=""):
        r"""Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys() # noqa
            ['bias', 'weight']

        """
        if destination is None:
            destination = OrderedDict()
        self._save_to_state_dict(destination, prefix)
        return destination

    def _load_from_state_dict(
        self, state_dict, prefix, strict, missing_keys, unexpected_keys, error_msgs
    ):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module. # noqa
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """

        local_name_params = self._buffers.items()
        # local_state = {k: v for k, v in local_name_params if v is not None}
        local_state = {k: v for k, v in local_name_params}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if isinstance(param, torch.Tensor):
                    if len(param.shape) == 0 and len(input_param.shape) == 1:
                        input_param = input_param[0]

                    if input_param.shape != param.shape:
                        # local shape should match the one in checkpoint
                        error_msgs.append(
                            "size mismatch for {}: copying a param with shape {} from checkpoint, "
                            "the shape in current model is {}.".format(
                                key, input_param.shape, param.shape
                            )
                        )
                        continue

                try:
                    with torch.no_grad():
                        if isinstance(input_param, Tensor):
                            param.copy_(input_param)
                        else:
                            self._buffers[name] = input_param
                except Exception as ex:
                    error_msgs.append(
                        'While copying the parameter named "{}", '
                        "whose dimensions in the model are {} and "
                        "whose dimensions in the checkpoint are {}, "
                        "an exception occured : {}.".format(
                            key, param.size(), input_param.size(), ex.args
                        )
                    )
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split(".", 1)[
                        0
                    ]  # get the name of param/buffer/child
                    if input_name not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(self, state_dict, strict=True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        state_dict = state_dict.copy()

        def load(module, prefix=""):
            module._load_from_state_dict(  # noqa
                state_dict, prefix, True, missing_keys, unexpected_keys, error_msgs
            )

        load(self)
        load = None  # break load->load reference cycle

        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0,
                "Unexpected key(s) in state_dict: {}. ".format(
                    ", ".join('"{}"'.format(k) for k in unexpected_keys)
                ),
            )
        if len(missing_keys) > 0:
            error_msgs.insert(
                0,
                "Missing key(s) in state_dict: {}. ".format(
                    ", ".join('"{}"'.format(k) for k in missing_keys)
                ),
            )

        if len(error_msgs) > 0:
            if strict:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        self.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
            else:
                warn(
                    RuntimeWarning(
                        "Error(s) in loading state_dict for {}:\n\t{}".format(
                            self.__class__.__name__, "\n\t".join(error_msgs)
                        )
                    )
                )
        return _IncompatibleKeys(missing_keys, unexpected_keys)
