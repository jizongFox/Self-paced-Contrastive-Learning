import atexit
import math

from torch.utils.tensorboard import SummaryWriter as _SummaryWriter

__tensorboard_queue__ = []

from contrastyou.utils import flatten_dict


def prune_dict(dictionary: dict, ignore="_"):
    for k, v in dictionary.copy().items():
        if isinstance(v, dict):
            prune_dict(v, ignore)
        else:
            if k.startswith(ignore):
                del dictionary[k]


class SummaryWriter(_SummaryWriter):

    def __init__(self, log_dir: str):
        super().__init__(log_dir)
        atexit.register(self.close)

    def add_scalar_with_tag(
        self, tag, tag_scalar_dict, global_step: int, walltime=None
    ):
        """
        Add one-level dictionary {A:1,B:2} with tag
        :param tag: main tag like `train` or `val`
        :param tag_scalar_dict: dictionary like {A:1,B:2}
        :param global_step: epoch
        :param walltime: None
        :return:
        """
        assert global_step is not None
        prune_dict(tag_scalar_dict)
        tag_scalar_dict = flatten_dict(tag_scalar_dict, sep=".")

        for k, v in tag_scalar_dict.items():
            if math.isnan(v):
                continue
            self.add_scalar(tag=f"{tag}   |   {k}", scalar_value=v, global_step=global_step, walltime=walltime)

    def add_scalars_from_meter_interface(self, *, epoch: int, **kwargs):
        for g, group_dictionary in kwargs.items():
            for k, v in group_dictionary.items():
                self.add_scalar_with_tag(g + "/" + k, v, global_step=epoch)

    def __enter__(self):
        __tensorboard_queue__.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        try:
            _self = __tensorboard_queue__.pop()
            assert id(_self) == id(self)
        except Exception:
            pass
        super(SummaryWriter, self).close()


def get_tb_writer() -> SummaryWriter:
    if len(__tensorboard_queue__) == 0:
        raise RuntimeError(
            "`get_tb_writer` must be call after with statement of a writer"
        )
    return __tensorboard_queue__[-1]
