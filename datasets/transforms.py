import torch
from torchvision import transforms

import numpy as np
import torch
from PIL import Image
import random


class Transforms:
    """
    Handles different input transformations for various depth estimation models.
    """

    @staticmethod
    def get_transforms(model_name: str, val: bool):
        """
        Returns the appropriate transformations based on the model name.
        """
        if model_name.lower() == "guidedepth" or model_name.lower() == "guidedepth-t" or model_name.lower() == "hybrid":
            if val:
                return transforms.Compose(
                    [Resize((240, 320)),
                     ToTensor(test=True, maxDepth=10.0)])
            else:
                return transforms.Compose([
                    Resize((240, 320)),
                    RandomHorizontalFlip(),
                    RandomChannelSwap(0.5),
                    ToTensor(test=False, maxDepth=10.0)
                ])
            # if val:
            #     return transforms.Compose(
            #         [Resize((480, 640)),
            #          ToTensor(test=True, maxDepth=10.0)])
            # else:
            #     return transforms.Compose([
            #         Resize((480, 640)),
            #         RandomHorizontalFlip(),
            #         RandomChannelSwap(0.5),
            #         ToTensor(test=False, maxDepth=10.0)
            #     ])

        elif model_name.lower() == "repmono-s":
            if val:
                return transforms.Compose([
                    transforms.ToTensor(
                    ),  # Converts NumPy array to (1, H, W) tensor and scales if uint8
                    transforms.Resize((240, 320)),  # Resize depth map
                    transforms.Lambda(lambda x: (x - x.min()) / (x.max(
                    ) - x.min() + 1e-8)),  # Min-Max Normalization
                ])
            else:
                return transforms.Compose([
                    transforms.Lambda(
                        lambda x: torch.tensor(x, dtype=torch.float32).permute(
                            2, 0, 1)),  # Convert to tensor (C, H, W)
                    transforms.Resize((240, 320)),  # Resize to (480, 640)
                    transforms.Lambda(
                        lambda x: x.unsqueeze(0))  # Add batch dimension
                ])
            # if val:
            #     return transforms.Compose([
            #         transforms.ToTensor(
            #         ),  # Converts NumPy array to (1, H, W) tensor and scales if uint8
            #         transforms.Resize((480, 640)),  # Resize depth map
            #         transforms.Lambda(lambda x: (x - x.min()) / (x.max(
            #         ) - x.min() + 1e-8)),  # Min-Max Normalization
            #     ])
            # else:
            #     return transforms.Compose([
            #         transforms.Lambda(
            #             lambda x: torch.tensor(x, dtype=torch.float32).permute(
            #                 2, 0, 1)),  # Convert to tensor (C, H, W)
            #         transforms.Resize((480, 640)),  # Resize to (480, 640)
            #         transforms.Lambda(
            #             lambda x: x.unsqueeze(0))  # Add batch dimension
            #     ])
        elif model_name.lower() == "repmono-u":
            return transforms.Resize(240,320)
        else:
            raise ValueError(f"Unknown model: {model_name}.")


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(
                type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(
                type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class RandomVerticalFlip(object):

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(
                type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(
                type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            depth = depth.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        return {'image': image, 'depth': depth}


class RandomChannelSwap(object):

    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(
                type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(
                type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(
                image[...,
                      list(self.indices[random.randint(0,
                                                       len(self.indices) -
                                                       1)])])
        return {'image': image, 'depth': depth}


class ToTensor(object):

    def __init__(self, test=False, maxDepth=1000.0):
        self.test = test
        self.maxDepth = maxDepth

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        transformation = transforms.ToTensor()

        if self.test:
            """
            If test, move image to [0,1] and depth to [0, 1]
            """
            image = np.array(image).astype(np.float32) / 255.0
            depth = np.array(depth).astype(
                np.float32) * 0.001  #/ self.maxDepth #Why / maxDepth?
            image, depth = transformation(image), transformation(depth)
        else:
            #Fix for PLI=8.3.0
            image = np.array(image).astype(np.float32) / 255.0 * 10.0
            depth = np.array(depth).astype(np.float32)

            #For train use DepthNorm
            zero_mask = depth == 0.0
            image, depth = transformation(image), transformation(depth)
            depth = torch.clamp(depth, self.maxDepth / 100.0, self.maxDepth)
            depth = self.maxDepth / depth
            depth[:, zero_mask] = 0.0

        #print('Depth after, min: {} max: {}'.format(depth.min(), depth.max()))
        #print('Image, min: {} max: {}'.format(image.min(), image.max()))

        image = torch.clamp(image, 0.0, 1.0)
        return {'image': image, 'depth': depth}


class CenterCrop(object):
    """
    Wrap torch's CenterCrop
    """

    def __init__(self, output_resolution):
        self.crop = transforms.CenterCrop(output_resolution)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image))
        if isinstance(depth, np.ndarray):
            depth = Image.fromarray(depth)
        image = self.crop(image)
        depth = self.crop(depth)

        return {'image': image, 'depth': depth}


class Resize(object):
    """
    Wrap torch's Resize
    """

    def __init__(self, output_resolution):
        self.resize = transforms.Resize(output_resolution)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image))
        if isinstance(depth, np.ndarray):
            depth = Image.fromarray(depth)

        image = self.resize(image)
        depth = self.resize(depth)

        return {'image': image, 'depth': depth}


class RandomRotation(object):
    """
    Wrap torch's Random Rotation
    """

    def __init__(self, degrees):
        self.angle = degrees

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        angle = random.uniform(-self.angle, self.angle)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image))
        if isinstance(depth, np.ndarray):
            depth = Image.fromarray(depth)

        image = transforms.functional.rotate(image, angle)
        depth = transforms.functional.rotate(depth, angle)

        return {'image': image, 'depth': depth}


def DepthNorm(depth, maxDepth=1000.0):
    return maxDepth / depth
