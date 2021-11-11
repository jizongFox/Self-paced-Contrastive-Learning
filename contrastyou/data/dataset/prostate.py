from pathlib import Path

from ._ioutils import downloading
from .base import DatasetBase
from ...augment import SequentialWrapper


class ProstateDataset(DatasetBase):
    download_link = "https://drive.google.com/uc?id=1hZISuvq2OGk6MZDhZ-p5ebV0q0IXAlaf"
    zip_name = "PROSTATE.zip"
    folder_name = "PROSTATE"

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None, ) -> None:
        path = Path(root_dir, self.folder_name)
        sub_folders = ["img", "gt"]
        sub_folder_types = ["image", "gt"]
        group_re = r"Case\d+"

        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode=mode, sub_folders=sub_folders,
                         sub_folder_types=sub_folder_types,
                         transforms=transforms, group_re=group_re)


class ProstateMDDataset(DatasetBase):
    folder_name = "PROSTATE_md"
    zip_name = "PROSTATE_md.zip"
    download_link = "https://drive.google.com/uc?id=1MngFjFmbO8lBHC0G6sbW7_kjjijQqSsu"

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None, ) -> None:
        path = Path(root_dir, self.folder_name)
        sub_folders = ["t2",  "gt"]
        sub_folder_types = ["image",  "gt"]
        group_re = r"prostate_\d+"

        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode=mode, sub_folders=sub_folders,
                         sub_folder_types=sub_folder_types,
                         transforms=transforms, group_re=group_re)
