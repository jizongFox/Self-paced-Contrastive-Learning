from pathlib import Path

from ._ioutils import downloading
from .base import DatasetBase
from ...augment import SequentialWrapper


class mmWHSCTDataset(DatasetBase):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "MMWHS.zip"
    folder_name = "MMWHS"

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        sub_folders = ["img", "gt"]
        sub_folder_types = ["image", "gt"]
        group_re = r"\d+"
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="ct_" + mode, sub_folders=sub_folders,
                         sub_folder_types=sub_folder_types,
                         transforms=transforms, group_re=group_re)


class mmWHSMRDataset(DatasetBase):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "MMWHS.zip"
    folder_name = "MMWHS"

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        sub_folders = ["img", "gt"]
        sub_folder_types = ["image", "gt"]
        group_re = r"\d+"
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="mr_" + mode, sub_folders=sub_folders,
                         sub_folder_types=sub_folder_types,
                         transforms=transforms, group_re=group_re)
