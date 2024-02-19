from typing import Optional, Dict, List

import numpy as np
import tensorflow as tf
from fedless.datasets.dataset_loaders import DatasetLoader, DatasetNotLoadedError
from fedless.common.cache import cache
from pydantic import BaseModel, Field

class SleepApneaConfig(BaseModel):
    """Configuration parameters for Sleep Apnea dataset"""

    type: str = Field("sleepapnea", const=True)
    indices: Optional[List[int]] = None
    split: str = "train"
    proxies: Optional[Dict] = None
    location: str = "fedless/datasets/sleepapnea/dataset/sleepapnea.npz"

class SleepApnea(DatasetLoader):
    def __init__(
        self,
        indices: Optional[List[int]] = None,
        split: str = "train",
        proxies: Optional[Dict] = None,
        location: str = "fedless/datasets/sleepapnea/dataset/sleepapnea.npz",
    ):
        self.split = split
        self.indices = indices
        self.proxies = proxies or {}
        self.location = location

    @cache
    def load(self) -> tf.data.Dataset:
        file_path = self.location

        try:
            with np.load(file_path, allow_pickle=True) as f:
                x_train, y_train = f["x_train"], f["y_train"]
                x_test, y_test = f["x_test"], f["y_test"]
        except FileNotFoundError:
            raise DatasetNotLoadedError(f"Sleep Apnea dataset not found at {file_path}")

        if self.split.lower() == "train":
            features, labels = x_train, y_train
        elif self.split.lower() == "test":
            features, labels = x_test, y_test
        else:
            raise DatasetNotLoadedError(f"Split {self.split} does not exist")

        if self.indices:
            features, labels = features[self.indices], labels[self.indices]

        def _scale_features(features, label):
            return tf.cast(features, tf.float32), tf.cast(label, tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((features, labels))

        return ds.map(_scale_features)
