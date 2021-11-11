from .synchronize import SequentialWrapperTwice, SequentialWrapper

"""
class ACDCTransforms:
    pretrain = SequentialWrapperTwice(
        com_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(30),  # interpolation to be nearest
        ]),
        image_transform=pil_augment.Compose([
            transforms.ColorJitter(brightness=[0.8, 1.3], contrast=[0.8, 1.3], saturation=[0.8, 1.3]),
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        com_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(30),  # interpolation to be nearest
        ]),
        image_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(
        com_transform=pil_augment.CenterCrop(224)
    )


class ACDCStrongTransforms:
    pretrain = SequentialWrapperTwice(
        com_transform=pil_augment.Compose([
            pil_augment.RandomRotation(30),
            pil_augment.RandomVerticalFlip(),
            pil_augment.RandomHorizontalFlip(),
            pil_augment.RandomCrop(224),

        ]),
        image_transform=pil_augment.Compose([
            transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5]),
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        com_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(30),
        ]),
        image_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(
        com_transform=pil_augment.CenterCrop(224)
    )
    val_double = SequentialWrapperTwice(
        com_transform=pil_augment.CenterCrop(224)
    )


transform_dict = {"strong": ACDCStrongTransforms, "simple": ACDCTransforms}
"""
