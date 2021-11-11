import os

import matplotlib.pyplot as plt
import torch
from deepclustering2.configparser import ConfigManger

from contrastyou.losses.contrast_loss2 import SelfPacedSupConLoss


def load_patient(projector_folder: str, anchor_name: str, positve_name: str):
    return torch.load(os.path.join(projector_folder, anchor_name)).cpu(), \
           torch.load(os.path.join(projector_folder, positve_name)).cpu()


class Gamma:

    def __init__(self, begin_value: float, end_value: float, max_epoch: int) -> None:
        super().__init__()
        self._begin_value = begin_value
        self._end_value = end_value
        self._max_epoch = max_epoch

    def __call__(self, cur_epoch):
        if cur_epoch > self._max_epoch:
            raise RuntimeError()
        return (self._end_value - self._begin_value) / self._max_epoch * cur_epoch + self._begin_value


def main(root_name, anchor_name, postive_name):
    config_manager = ConfigManger(base_path=os.path.join(root_name, "pre", "config.yaml"))
    config = config_manager.config
    with config_manager(scope="base"):
        projection_folder = os.path.join(root_name, "projections")
        folders = sorted([x for x in os.listdir(projection_folder)], key=lambda x: int(x))
        try:
            criterion = SelfPacedSupConLoss(weight_update=config["ProjectorParams"]["LossParams"]["weight_update"][0])
            gamma_evaluator = Gamma(config["ProjectorParams"]["LossParams"]["begin_value"][0],
                                    config["ProjectorParams"]["LossParams"]["end_value"][0],
                                    config["Trainer"]["max_epoch"])
        except Exception as e:
            criterion = SelfPacedSupConLoss()
            gamma_evaluator = Gamma(1e6, 1e6,
                                    config["Trainer"]["max_epoch"])

        for cur_epoch in folders[::]:
            criterion.set_gamma(gamma_evaluator(int(cur_epoch)))
            proj1, proj2 = load_patient(projector_folder=os.path.join(projection_folder, cur_epoch),
                                        anchor_name=anchor_name,
                                        positve_name=postive_name)
            loss = criterion(proj1, proj2)
            plt.figure()
            plt.imshow(criterion.weight.cpu(), cmap="gray")
            plt.colorbar()
            # if int(cur_epoch) > 100:
            #     break

    plt.show()


if __name__ == '__main__':
    main(
        "/home/jizong/Workspace/Contrast-You/semi_seg/runs/0411/demo/soften_case1/type_inversesquare/"

        , "patient004_00", "patient006_01")
