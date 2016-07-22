import os
import json
import scipy.misc
import numpy as np
import tensorflow as tf


def get_image(image_path, image_size, is_crop=True):
    img = transform(imread(image_path), image_size, is_crop)
    # box_size = 10
    # x = np.random.randint(0, 64 - 10)
    # y = np.random.randint(0, 64 - 10)
    # img[x:x + box_size, y:y + box_size, :] = 0
    return img


def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_w, resize_w])


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


class DataPath:
    base = json.loads(open("../config.json").read()).get("path", "")
    celeb_path = "/home/bgavran3/petnica/src/model/downloaded_examples/DCGAN_tensorflow/data/celebA"
    pass


class Data:
    def next_batch(self, batch_size):
        raise NotImplementedError()


class CelebDataset(Data):
    image_size = 108

    def __init__(self):
        self.f = []
        for (dirpath, dirnames, filenames) in os.walk(DataPath.celeb_path):
            self.f.extend(filenames)
        self.total_size = len(self.f)
        self.curr_batch_index = 0

    def next_batch(self, batch_size):
        if self.curr_batch_index * batch_size >= self.total_size - batch_size:
            self.curr_batch_index = 0
        img_names = self.f[self.curr_batch_index * batch_size:(self.curr_batch_index + 1) * batch_size]
        self.curr_batch_index += 1
        return [get_image(os.path.join(DataPath.celeb_path, img_name), CelebDataset.image_size) for img_name in
                img_names], [-1 for _ in img_names]
