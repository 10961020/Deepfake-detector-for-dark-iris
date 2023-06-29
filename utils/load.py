import os
import cv2
import random

import torch
from PIL import Image
import numpy as np


def get_ids(dir_img):
    return (f[:-4] for f in os.listdir(dir_img))


def split_train_val(dataset, batch_size, val_percent=0.1):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    if n % batch_size:
        n = n - n % batch_size
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def get_imgs_and_masks(ids, dir_img, suffix):
    """Return all the couples (img, mask)"""
    imgs = to_cropped_imgs(ids, dir_img, suffix)
    imgs_switched = map(hwc_to_chw, imgs)  # need to transform from HWC to CHW
    return imgs_switched


def to_cropped_imgs(ids, dir_img, suffix):  # params suffix: is data format
    """From a list of tuples, returns the correct cropped img"""
    for id_img in ids:
        imname = os.path.join(dir_img, id_img+suffix)
        im = resize_and_crop(Image.open(imname))
        yield im


def hwc_to_chw(img):
    return np.transpose(img, axes=[0, 3, 1, 2])


def normalize(x):
    return x / 255


def resize_and_crop(img):
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []
    if len(b) > 0:
        yield b


# data_type = 0 为 img, data_type = 1 为 mask
def tuple_transform_np(imgs, batch_size, data_type):
    if data_type == 0:
        img_np = np.array([]).reshape((0, 3, 800, 800))
    else:
        img_np = np.array([]).reshape((0, 800, 800))
    for img in imgs:
        img_np = np.concatenate([img_np, img[data_type]], axis=0)
    return img_np.astype(np.uint8)


def cv_canny(masks, batch_size):
    batch_size *= 5
    img_edge_mask = np.array([]).reshape([0, 128, 128])
    for i in range(batch_size):
        input_mask = masks[i].numpy().astype('uint8')
        input_mask = cv2.resize(input_mask, (int(input_mask.shape[0] / 4), int(input_mask.shape[0] / 4)))
        input_mask = np.expand_dims(cv2.Canny(input_mask, 25, 255), axis=0)
        img_edge_mask = np.concatenate((img_edge_mask, input_mask), axis=0)
    return torch.from_numpy(img_edge_mask.astype('float32'))
