import json
import numpy as np
import os
from sklearn.cross_validation import train_test_split

from src.model import utils


class DataPath:
    json_file = json.loads(open("../config.json").read())
    base = json_file["path"]
    celeb_path = json_file["celeb_path"]


class Data:
    def next_batch(self, batch_size):
        raise NotImplementedError()


class CelebDataset(Data):
    image_size = 108

    def __init__(self, data):
        self.data = data
        self.curr_batch_index = 0

    def next_batch(self, batch_size):
        if self.curr_batch_index * batch_size >= len(self.data) - batch_size:
            self.curr_batch_index = 0
        batch = self.data[self.curr_batch_index * batch_size:(self.curr_batch_index + 1) * batch_size]
        batch = [CelebDataset.load_image(img_name) for img_name in batch]

        self.curr_batch_index += 1
        return np.array(batch)

    @staticmethod
    def load_image(img_name, celeb_path=True, is_crop=True):
        if celeb_path:
            img_name = os.path.join(DataPath.celeb_path, img_name)
        return utils.get_image(img_name, CelebDataset.image_size, is_crop=is_crop)

    @staticmethod
    def load_data():
        img_names = []
        for (dirpath, dirnames, filenames) in os.walk(DataPath.celeb_path):
            img_names.extend(filenames)
        data = img_names
        # data = [CelebDataset.load_image(img) for img in img_names]
        np.random.shuffle(data)
        return data


class DataSets:
    def __init__(self, data_feed, test_ratio=0.3):
        data = data_feed.load_data()
        data_train, data_test = train_test_split(data, test_size=test_ratio)
        self.train = CelebDataset(data_train)
        self.test = CelebDataset(data_test)
