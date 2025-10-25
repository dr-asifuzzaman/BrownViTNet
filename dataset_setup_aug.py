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


import jax
from jax import jit
from jax import random
from jax import numpy as jnp
from jax.experimental import jax2tf
import helper_functions

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.optimizer.set_jit(True)
    keras.mixed_precision.set_global_policy("mixed_float16")
except:
    pass

seed = 1337
tf.random.set_seed(seed)


import pathlib
import numpy as np
data_dir = "dataset"
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.png')))


class Parameters:
    # data level
    image_size = 128
    batch_size = 8
    num_grad_accumulation = 8
    label_smooth=0.05
    class_number = 3
    val_split = 0.2 
    verbosity = 1
    autotune = tf.data.AUTOTUNE
    
    # hparams
    epochs = 125
    lr_sched = 'cosine_restart' # [or, exponential, cosine, linear, constant]
    lr_base  = 0.016
    lr_min   = 0
    lr_decay_epoch  = 2.4
    lr_warmup_epoch = 5
    lr_decay_factor = 0.97
    
    scaled_lr = lr_base * (batch_size / 256.0)
    scaled_lr_min = lr_min * (batch_size / 256.0)
    num_validation_sample = int(image_count * val_split)
    num_training_sample = image_count - num_validation_sample
    train_step = int(np.ceil(num_training_sample / float(batch_size)))
    total_steps = train_step * epochs

params = Parameters()








class RandomApply(layers.Layer):
    """RandomApply will randomly apply the transformation layer
    based on the given probability.
    
    Ref. https://stackoverflow.com/a/72558994/9215780
    """

    def __init__(self, layer, probability, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.probability = probability

    def call(self, inputs, training=True):
        apply_layer = tf.random.uniform([]) < self.probability
        outputs = tf.cond(
            pred=tf.logical_and(apply_layer, training),
            true_fn=lambda: self.layer(inputs),
            false_fn=lambda: inputs,
        )
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer": layers.serialize(self.layer),
                "probability": self.probability,
            }
        )
        return config
        
class RandomGrayscale(layers.Layer):
    """Grayscale is a preprocessing layer that transforms
    RGB images to Grayscale images.

    Ref. https://gist.github.com/innat/4e89725ccdcd763e0a6ba19216fd60bf
    """

    def __init__(self, output_channel=1, prob=1, **kwargs):
        super().__init__(**kwargs)
        self.output_channel = self._check_input_params(output_channel)

    def _check_input_params(self, output_channels):
        if output_channels not in [1, 3]:
            raise ValueError(
                "Received invalid argument output_channels. "
                f"output_channels must be in 1 or 3. Got {output_channels}"
            )
        return output_channels

    @partial(jit, static_argnums=0)
    def _jax_gray_scale(self, images):
        rgb_weights = jnp.array([0.2989, 0.5870, 0.1140], dtype=images.dtype)
        grayscale = (rgb_weights * images).sum(axis=-1)

        if self.output_channel == 1:
            grayscale = jnp.expand_dims(grayscale, axis=-1)
            return grayscale
        elif self.output_channel == 3:
            return jnp.stack([grayscale] * 3, axis=-1)
        else:
            raise ValueError("Unsupported value for `output_channels`.")

    def call(self, images, training=True):
        if training:
            return jax2tf.convert(
                self._jax_gray_scale, polymorphic_shapes=("batch, ...")
            )(images)
        else:
            return images

    def get_config(self):
        config = {
            "output_channel": self.output_channel,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomChannelShuffle(layers.Layer):
    """Shuffle channels of an input image.

    Ref. https://gist.github.com/innat/35ab35329e2ca890a17556384056334b
    """
    def __init__(self, groups=3, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

    @partial(jit, static_argnums=0)
    def _jax_channel_shuffling(self, images):
        batch_size, height, width, num_channels = images.shape

        if not num_channels % self.groups == 0:
            raise ValueError(
                "The number of input channels should be "
                "divisible by the number of groups."
                f"Received: channels={num_channels}, groups={self.groups}"
            )

        channels_per_group = num_channels // self.groups

        images = images.reshape(-1, height, width, self.groups, channels_per_group)
        images = images.transpose([3, 1, 2, 4, 0])
        key = random.PRNGKey(np.random.randint(50))
        images = random.permutation(key=key, x=images, axis=0)
        images = images.transpose([4, 1, 2, 3, 0])
        images = images.reshape(-1, height, width, num_channels)
        return images

    def call(self, images, training=True):
        if training:
            return jax2tf.convert(
                self._jax_channel_shuffling, polymorphic_shapes=("batch, ...")
            )(images)
        else:
            return images
    
    def get_config(self):
        config = {
            "groups": self.groups,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))




train_set = keras.utils.image_dataset_from_directory(
    # train_dir,
    data_dir,
    validation_split=params.val_split,
    subset="training",
    label_mode='categorical',
    seed=params.image_size,
    image_size=(params.image_size, params.image_size),
    batch_size=params.batch_size,
)

val_set = keras.utils.image_dataset_from_directory(
    # test_dir,
    data_dir,
    validation_split=params.val_split,
    subset="validation",
    label_mode='categorical',
    seed=params.image_size,
    image_size=(params.image_size, params.image_size),
    batch_size=params.batch_size,
)











temp_ds = train_set.map(
    lambda x, y: (RandomChannelShuffle()(x), y), num_parallel_calls=params.autotune
)

# make_plot(temp_ds, take_batch=1, title=False)

temp_ds = train_set.map(
    lambda x, y: (RandomGrayscale(output_channel=3)(x), y), num_parallel_calls=params.autotune
)

# make_plot(temp_ds, take_batch=1, title=False) 


jax_to_keras_augment = keras.Sequential(
    [
        RandomApply(RandomGrayscale(output_channel=3), probability=0.2),
        RandomApply(RandomChannelShuffle(), probability=0.5),
    ],
    name="jax2keras_augment",
)


tf_to_keras_augment = keras.Sequential(
    [
        # RandomApply(layers.RandomFlip("horizontal"), probability=0.5),
        RandomApply(layers.RandomZoom(0.2, 0.3), probability=0.2),
        RandomApply(
            layers.RandomRotation((0.02, 0.03), fill_mode="reflect"), probability=0.8
        ),
    ],
    name="tf2keras_augment",
)

# for train set : augmentation
keras_aug = keras.Sequential(
    [
        layers.Resizing(height=params.image_size, width=params.image_size),
        jax_to_keras_augment,
        tf_to_keras_augment,
    ],
    name="keras_augment",
)

train_ds = train_set.shuffle(10 * params.batch_size)
train_ds = train_ds.map(
    lambda x, y: (keras_aug(x), y), num_parallel_calls=params.autotune
)
# train_ds = train_ds.map(
#     lambda x, y: RandomMixUpCutMix()([x, y]), num_parallel_calls=params.autotune
# )

train_ds = train_ds.prefetch(buffer_size=params.autotune)
val_ds = val_set.prefetch(buffer_size=params.autotune)
