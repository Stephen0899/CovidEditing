import numpy as np
import cv2
from torchvision import transforms, utils
import torch

trans = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

def border_pad(image, cfg):
    h, w, c = image.shape

    if cfg.border_pad == 'zero':
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode='constant',
                       constant_values=0.0)
    elif cfg.border_pad == 'pixel_mean':
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode='constant',
                       constant_values=cfg.pixel_mean)
    else:
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode=cfg.border_pad)

    return image


def fix_ratio(image, cfg):
    h, w, c = image.shape

    if h >= w:
        ratio = h * 1.0 / w
        h_ = cfg.long_side
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = cfg.long_side
        h_ = round(w_ / ratio)

    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
    image = border_pad(image, cfg)

    return image


def transform(image, cfg):
    height = cfg.height
    width = cfg.width
    assert image.ndim == 2, "image must be gray image"
    if cfg.use_equalizeHist:
        image = cv2.equalizeHist(image)

    if cfg.gaussian_blur > 0:
        image = cv2.GaussianBlur(
            image,
            (cfg.gaussian_blur, cfg.gaussian_blur), 0)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = trans(image)

    # image = fix_ratio(image, cfg)
    # # augmentation for train or co_train

    # # normalization
    # image = image.astype(np.float32) - cfg.pixel_mean
    # # vgg and resnet do not use pixel_std, densenet and inception use.
    # if cfg.pixel_std:
    #     image /= cfg.pixel_std
    
    # print(np.min(image),np.max(image))
    # # normal image tensor :  H x W x C
    # # torch image tensor :   C X H X W
    # image = image.transpose((2, 0, 1))

    return image
