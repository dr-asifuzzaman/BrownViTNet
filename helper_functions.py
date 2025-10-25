import os
import random
import warnings

import cv2
import gdown
from functools import partial

warnings.simplefilter(action="ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import pandas as pd
from matplotlib import cm
from numpy.random import rand
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.optimizer.set_jit(True)
    keras.mixed_precision.set_global_policy("mixed_float16")
except:
    pass

seed = 1337
tf.random.set_seed(seed)





learning_rate = 0.001
weight_decay = 0.0001
batch_size = 64
num_epochs = 75  # For real training, use num_epochs=100. 10 is a test value
image_size = 128  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [
    2048,
    1024,
]  # Size of the dense layers of the final classifier




def get_model_weight(model_id):
    """Get the trained weights."""
    if not os.path.exists("model.h5"):
        model_weight = gdown.download(id=model_id, quiet=False)
    else:
        model_weight = "model.h5"
    return model_weight


def get_model_history(history_id):
    """Get the history / log file."""
    if not os.path.exists("history.csv"):
        history_file = gdown.download(id=history_id, quiet=False)
    else:
        history_file = "history.csv"
    return history_file


def make_plot(tfdata, take_batch=1, title=True, figsize=(20, 20)):
    """ref: https://gist.github.com/innat/4dc4080cfdf5cf20ef0fc93d3623ca9b"""

    font = {
        "family": "serif",
        "color": "darkred",
        "weight": "normal",
        "size": 15,
    }

    for images, labels in tfdata.take(take_batch):
        plt.figure(figsize=figsize)
        xy = int(np.ceil(images.shape[0] * 0.5))

        for i in range(images.shape[0]):
            plt.subplot(xy, xy, i + 1)
            plt.imshow(tf.cast(images[i], dtype=tf.uint8))
            if title:
                plt.title(tcls_names[tf.argmax(labels[i], axis=-1)], fontdict=font)
            plt.axis("off")

    plt.tight_layout()
    plt.show()