from io import BytesIO
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from torchvision import transforms

from contrastyou.augment.synchronize import SequentialWrapper, SequentialWrapperTwice
from deepclustering2.augment import pil_augment
from deepclustering2.decorator import FixRandomSeed

url = "https://www.sciencemag.org/sites/default/files/styles/article_main_image_-_1280w__no_aspect_/public/dogs_1280p_0.jpg?itok=6jQzdNB8"
response = requests.get(url)


class TestTransformationWrapper(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._img1 = Image.open(BytesIO(response.content))
        self._img2 = Image.fromarray((255.0 - np.asarray(self._img1)).astype(np.uint8))
        self._target1 = Image.fromarray((np.asarray(self._img1) < 128).astype(np.uint8)).convert("L")
        self._target2 = Image.fromarray((np.asarray(self._img1) >= 128).astype(np.uint8)).convert("L")

    def test_sqeuential_wrapper(self):
        comm_transform = pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(23),
        ])
        img_transform = pil_augment.ToTensor()
        target_transform = pil_augment.ToLabel()
        wrapper = SequentialWrapper(comm_transform=comm_transform,
                                    img_transform=img_transform,
                                    target_transform=target_transform)

        imgs, targets = wrapper(imgs=[self._img1, self._img2], targets=[self._target1, self._target2])

        plt.imshow(imgs[0].numpy().transpose(1, 2, 0))
        plt.show()
        plt.imshow(imgs[1].numpy().transpose(1, 2, 0))
        plt.show()
        plt.imshow(targets[0].numpy().squeeze())
        plt.show()

        plt.imshow(targets[1].numpy().squeeze())
        plt.show()

    def test_on_double(self):
        comm_transform = pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(23),
        ])
        img_transform = pil_augment.Compose([
            transforms.ColorJitter(brightness=[0.5, 1.1]),
            pil_augment.ToTensor()
        ])
        target_transform = pil_augment.ToLabel()
        wrapper = SequentialWrapper(comm_transform=comm_transform,
                                    img_transform=img_transform,
                                    target_transform=target_transform)
        with FixRandomSeed(2):
            imgs1, targets1 = wrapper(imgs=[self._img1, self._img2], targets=[self._target1, self._target2])
        with FixRandomSeed(2):
            imgs2, targets2 = wrapper(imgs=[self._img1, self._img2], targets=[self._target1, self._target2], img_seed=3)
        plt.imshow(imgs1[0].numpy().transpose(1, 2, 0))
        plt.show()
        plt.imshow(targets1[0].numpy().squeeze())
        plt.show()
        plt.imshow(imgs2[0].numpy().transpose(1, 2, 0))
        plt.show()
        plt.imshow(targets2[0].numpy().squeeze())
        plt.show()

    def test_on_img(self):
        comm_transform = pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(23),
        ])
        img_transform = pil_augment.Compose([
            transforms.ColorJitter(brightness=[0.5, 1.1]),
            pil_augment.ToTensor()
        ])
        target_transform = pil_augment.ToLabel()
        wrapper = SequentialWrapper(comm_transform=comm_transform,
                                    img_transform=img_transform,
                                    target_transform=target_transform)
        imgs1, targets1 = wrapper(imgs=[self._img1, self._img2])

    def test_on_twice_wrapper(self):
        comm_transform = pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(23),
        ])
        img_transform = pil_augment.Compose([
            transforms.ColorJitter(brightness=[0, 1.1], contrast=[0, 1.5]),
            pil_augment.ToTensor()
        ])
        target_transform = pil_augment.ToLabel()
        wrapper = SequentialWrapperTwice(comm_transform=comm_transform,
                                         img_transform=img_transform,
                                         target_transform=target_transform,
                                         total_freedom=True)
        (imgs1, targets1), (imgs2, targets2) = wrapper(imgs=[self._img1, self._img2], targets=[self._target1])
        plt.imshow(imgs1[0].numpy().transpose(1, 2, 0))
        plt.show()
        plt.imshow(targets1[0].numpy().squeeze())
        plt.show()
        plt.imshow(imgs2[0].numpy().transpose(1, 2, 0))
        plt.show()
        plt.imshow(targets2[0].numpy().squeeze())
        plt.show()

        wrapper = SequentialWrapperTwice(comm_transform=comm_transform,
                                         img_transform=img_transform,
                                         target_transform=target_transform,
                                         total_freedom=False)
        (imgs1, targets1), (imgs2, targets2) = wrapper(imgs=[self._img1, self._img2], targets=[self._target1])
        plt.imshow(imgs1[0].numpy().transpose(1, 2, 0))
        plt.show()
        plt.imshow(targets1[0].numpy().squeeze())
        plt.show()
        plt.imshow(imgs2[0].numpy().transpose(1, 2, 0))
        plt.show()
        plt.imshow(targets2[0].numpy().squeeze())
        plt.show()
