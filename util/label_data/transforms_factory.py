""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2020 Ross Wightman
"""
import random
import traceback
import numpy as np

import paddle.vision.transforms.functional as F
from paddle.vision import transforms

from .auto_augment import rand_augment_transform, RandAugment
from ..data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ..data.transforms import _pil_interp, _RANDOM_INTERPOLATION, RandomResizedCropAndInterpolation
from ..data.random_erasing import RandomErasing


class ComposeWithLabel(transforms.Compose):

    def __call__(self, image, label):
        for f in self.transforms:
            try:
                if isinstance(f, (
                    RandomHorizontalFlipWithLabel, RandomVerticalFlipWithLabel,
                    RandAugment, RandomResizedCropAndInterpolationWithCoords)):
                    image, label = f((image, label))
                else:
                    image = f(image)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform transform [{}] with error: "
                      "{} and stack:\n{}".format(f, e, str(stack_info)))
                raise e
        return image, label


class RandomHorizontalFlipWithLabel(transforms.RandomHorizontalFlip):

    def _apply_label(self, label):
        return label[:, :, :, ::-1]


class RandomVerticalFlipWithLabel(transforms.RandomVerticalFlip):

    def _apply_label(self, label):
        return label[:, :, ::-1]


class RandomResizedCropAndInterpolationWithCoords(RandomResizedCropAndInterpolation):

    def _get_params(self, inputs):
        image, label = inputs
        return self._get_param(image), transforms.transforms._get_image_size(image)

    def _apply_image(self, img):
        interpolation = self.interpolation
        if self.interpolation == 'random':
            interpolation = random.choice(_RANDOM_INTERPOLATION)

        (i, j, h, w), (width, height) = self.params

        cropped_img = F.crop(img, i, j, h, w)
        return F.resize(cropped_img, self.size, interpolation)

    def _apply_label(self, label):
        (i, j, h, w), (width, height) = self.params

        coords = (i / height, j / width, h / height, w / width)
        coords_map = np.zeros_like(label[0:1])
        # trick to store coords_map is label
        coords_map[0, 0, 0, 0], coords_map[0, 0, 0, 1], \
            coords_map[0, 0, 0, 2], coords_map[0, 0, 0, 3] = coords
        return np.concatenate([label, coords_map])


def transforms_imagenet_train(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range

    primary_tfl = []
    if hflip > 0.:
        primary_tfl += [RandomHorizontalFlipWithLabel(prob=hflip, keys=("image", "label"))]
    if vflip > 0.:
        primary_tfl += [RandomVerticalFlipWithLabel(prob=vflip, keys=("image", "label"))]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif color_jitter is not None:
        raise NotImplementedError

    final_tfl = [RandomResizedCropAndInterpolationWithCoords(
        img_size, scale=scale, ratio=ratio, interpolation=interpolation, keys=("image", "label"))]
    if use_prefetcher:
        raise NotImplementedError
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std)
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits))

    if separate:
        raise NotImplementedError
    else:
        return ComposeWithLabel(primary_tfl + secondary_tfl + final_tfl)


def create_transform(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False):

    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        raise NotImplementedError
    else:
        if is_training and no_aug:
            assert not separate, "Cannot perform split augmentation with no_aug"
            raise NotImplementedError
        elif is_training:
            transform = transforms_imagenet_train(
                img_size,
                scale=scale,
                ratio=ratio,
                hflip=hflip,
                vflip=vflip,
                color_jitter=color_jitter,
                auto_augment=auto_augment,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
                re_num_splits=re_num_splits,
                separate=separate)
        else:
            assert not separate, "Separate transforms not supported for validation preprocessing"
            raise NotImplementedError

    return transform
