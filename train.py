from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
import tensorflow as tf

from dataset_setup_aug import *
from model import *

from helper_functions import *



import pathlib
import numpy as np
data_dir = "../20240417 ICDEC conference BrownField/MargeResizeAndSuperImage"
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






model = BrownViTNet()

model.summary()


class WarmupLearningRateSchedule(optimizers.schedules.LearningRateSchedule):
    """WarmupLearningRateSchedule a variety of learning rate
    decay schedules with warm up.
    
    Ref. https://gist.github.com/innat/69e8f3500c2418c69b150a0a651f31dc
    """

    def __init__(
        self,
        initial_lr,
        steps_per_epoch=None,
        lr_decay_type="cosine_restart",
        decay_factor=0.97,
        decay_epochs=2.4,
        total_steps=None,
        warmup_epochs=5,
        minimal_lr=0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.initial_lr = initial_lr
        self.steps_per_epoch = steps_per_epoch
        self.lr_decay_type = lr_decay_type
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        self.total_steps = total_steps
        self.warmup_epochs = warmup_epochs
        self.minimal_lr = minimal_lr

    def __call__(self, step):
        if self.lr_decay_type == "exponential":
            assert self.steps_per_epoch is not None
            decay_steps = self.steps_per_epoch * self.decay_epochs
            lr = schedules.ExponentialDecay(
                self.initial_lr, decay_steps, self.decay_factor, staircase=True
            )(step)
            
        elif self.lr_decay_type == "cosine":
            assert self.total_steps is not None
            lr = (
                0.5
                * self.initial_lr
                * (1 + tf.cos(np.pi * tf.cast(step, tf.float32) / self.total_steps))
            )

        elif self.lr_decay_type == "linear":
            assert self.total_steps is not None
            lr = (1.0 - tf.cast(step, tf.float32) / self.total_steps) * self.initial_lr

        elif self.lr_decay_type == "constant":
            lr = self.initial_lr

        elif self.lr_decay_type == "cosine_restart":
            decay_steps = self.steps_per_epoch * self.decay_epochs
            lr = tf.keras.experimental.CosineDecayRestarts(
                self.initial_lr, decay_steps
            )(step)
        else:
            assert False, "Unknown lr_decay_type : %s" % self.lr_decay_type

        if self.minimal_lr:
            lr = tf.math.maximum(lr, self.minimal_lr)

        if self.warmup_epochs:
            warmup_steps = int(self.warmup_epochs * self.steps_per_epoch)
            warmup_lr = (
                self.initial_lr
                * tf.cast(step, tf.float32)
                / tf.cast(warmup_steps, tf.float32)
            )
            lr = tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: lr)

        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "lr_decay_type": self.lr_decay_type,
            "decay_factor": self.decay_factor,
            "decay_epochs": self.decay_epochs,
            "total_steps": self.total_steps,
            "warmup_epochs": self.warmup_epochs,
            "minimal_lr": self.minimal_lr,
        }



additional_metrics = [
    "accuracy",
    metrics.AUC(name='auc'),  # Area Under the Receiver Operating Characteristic Curve
    metrics.TruePositives(name='tp'),
    metrics.TrueNegatives(name='tn'),
    metrics.FalsePositives(name='fp'),
    metrics.FalseNegatives(name='fn'),
    metrics.Precision(name='precision'),
    metrics.Recall(name='recall'),
    metrics.AUC(curve='PR', name='prc')  # Precision-Recall Curve
]



model.compile(
    loss=losses.CategoricalCrossentropy(
        label_smoothing=params.label_smooth, from_logits=True
    ),
    optimizer=optimizers.Adam(learning_rate, amsgrad=True),
    metrics=additional_metrics,
)


import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

model_path = 'CustomModel_3Chnl_mergeWithSuperImage.h5'

# Callback to save the model with the highest validation accuracy
checkpoint_callback = ModelCheckpoint(
    model_path, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

# Callback to log the epoch results to a CSV file
csv_logger = CSVLogger('training_log.csv')

# Callback to reduce learning rate when the validation accuracy has not improved after 5 epochs
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.1,  # Reduce learning rate by a factor of 0.1
    patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
    verbose=1
)

# Use these callbacks in the `fit` method
history = model.fit(
    train_ds,
    epochs=params.epochs,
    callbacks=[checkpoint_callback, csv_logger, reduce_lr],
    validation_data=val_ds,
    verbose=params.verbosity,
).history
