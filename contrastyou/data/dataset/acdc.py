from pathlib import Path

from ._ioutils import downloading
from .base import DatasetBase
from ...augment import SequentialWrapper


class ACDCDataset(DatasetBase):
    download_link = "https://drive.google.com/uc?id=147xICU__T23aOYkdjGSA5Hh8W1SK-y9p"
    zip_name = "ACDC-all.zip"
    folder_name = "ACDC-all"

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        sub_folders = ["img", "gt"]
        sub_folder_types = ["image", "gt"]
        group_re = r"patient\d+_\d+"
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)

        super().__init__(root_dir=str(path), mode=mode, sub_folders=sub_folders, sub_folder_types=sub_folder_types,
                         transforms=transforms, group_re=group_re)
