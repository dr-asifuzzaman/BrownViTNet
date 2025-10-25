import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import helper_functions
import dataset_setup_aug
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Softmax


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



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


import tensorflow as tf
from tensorflow.keras import layers

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        # Getting dynamic shapes
        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        
        # Calculating the number of patches
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        
        # Preparing sizes and strides for tf.image.extract_patches
        sizes = [1, self.patch_size, self.patch_size, 1]
        strides = [1, self.patch_size, self.patch_size, 1]
        rates = [1, 1, 1, 1]  # No skipping, standard behavior
        
        # Extracting patches using tf.image.extract_patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=sizes,
            strides=strides,
            rates=rates,
            padding='VALID'
        )
        print("Exracted patches shape : ", patches.shape)
        
        # Reshaping the extracted patches
        patches = tf.reshape(
            patches,
            [batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels],
        )
        print("Before return after reshape: ", patches.shape)
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config



class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        # Use tf.range and tf.expand_dims instead of ops
        positions = tf.expand_dims(
            tf.range(start=0, limit=self.num_patches, delta=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection.units})
        return config




import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization


def depth_block(x, strides=(1,1)):
    x = DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.nn.gelu(x)
    return x

def single_conv_block(x, filters):
    x = Conv2D(filters,  3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.nn.gelu(x)
    return x

def combo_layer(x, filters, strides):
    shortcut = x

    x = depth_block(x, strides)
    x = single_conv_block(x, filters)

    # This is the addition for the residual connection
    shortcut = Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
    x = layers.Add()([x, shortcut])

    return x


num_classes = 3
input_shape = (128, 128, 3)

def BrownViTNet():
    inputs = keras.Input(shape=input_shape)
    # Augment data.
    # augmented = data_augmentation(inputs)

    x=combo_layer(inputs, 32, 2)
    # print("shape of x1", x.shape)
    x=combo_layer(x, 32, 1)
    # print("shape of x1", x.shape)
    x=combo_layer(x, 64, 1)
    # print("shape of x2", x.shape)
    x=combo_layer(x, 64, 2)
    # print("shape of x2", x.shape)

    # x=combo_layer(x, 128, 1)
    # print("shape of x2", x.shape)

    img = x.shape
    image_size = img[1]
    print("Image size: ",image_size)

    num_patches = (image_size // 16) ** 2
    # Create patches.
    patches = Patches(patch_size)(x)
    # Encode patches.

    print("Patches Layer Shape: ",patches.shape)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    print("Encoded_patches Layer Shape: ", encoded_patches.shape)


    
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model

