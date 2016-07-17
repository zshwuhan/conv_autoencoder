import os
import json


class DataPath:
    base = json.loads(open("../config.json").read()).get("path", "")
    pass


class Data:
    def next_batch(self, batch_size, review_size):
        raise NotImplementedError()
