import tensorflow as tf
from preproc.tf_dataset import prepare_dataset, configure_for_performance
from settings.config import Home
from loss.loss import WCE_from_logits, FocalLoss
import numpy as np
import pandas as pd
from models.model import NMS_keynet, NMS_superpoint
from models.unet import unet_mobilenet

CONFIG = Home


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.

  Conv2DTranspose => Batchnorm => Dropout => Relu

  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer

  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def unet_model(output_channels):
  # inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  inputs = tf.keras.layers.Input(shape=[CONFIG["H"], CONFIG["W"], CONFIG["C"]])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


base_model = tf.keras.applications.MobileNetV2(input_shape=[CONFIG["H"], CONFIG["W"], CONFIG["C"]], include_top=False, weights=None)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = True

up_stack = [
    upsample(512, 3),  # 4x4 -> 8x8
    upsample(256, 3),  # 8x8 -> 16x16
    upsample(128, 3),  # 16x16 -> 32x32
    upsample(64, 3),   # 32x32 -> 64x64
]

OUTPUT_CHANNELS = 1
#model = unet_model(OUTPUT_CHANNELS)
model = unet_mobilenet(use_pretrained_encoder=False)


model.build(input_shape=(CONFIG["BATCH_SIZE"], CONFIG["H"], CONFIG["W"], CONFIG["C"]))

low = 100 / (320 * 480)
high = 1 / low

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=WCE_from_logits(pos_weight=high))

#----------------------------------------------------
ds_paths = tf.data.Dataset.list_files("data/input/train/*", shuffle=True)

dataset = ds_paths.map(lambda path: tf.py_function(func=prepare_dataset,
                                                   inp=[path],
                                                   Tout=[tf.float32, tf.float32]),
                                                   num_parallel_calls=tf.data.AUTOTUNE)

# train, test, val
ds_train = dataset.skip(CONFIG["VAL_SIZE"])
ds_val = dataset.take(CONFIG["VAL_SIZE"])
# batch, prefetch, cache
ds_train = configure_for_performance(ds_train, CONFIG["BATCH_SIZE"])
ds_val = configure_for_performance(ds_val, CONFIG["BATCH_SIZE"])

"""model.load_weights("models/checkpoints/unet/noweights")
model.evaluate(ds_val, verbose=2)

for x, y in ds_train:
    logits = model.call(x)

    logits = tf.reshape(logits, (CONFIG["H"], CONFIG["W"]))

    classifications = tf.round(tf.nn.sigmoid(logits)).numpy()
    score_map = tf.nn.softmax(logits).numpy()

    classific = NMS_keynet(score_map, 15)
    classific2 = NMS_superpoint(score_map, 15).numpy()

    indexes = np.argwhere(classific > 0)
    indexes2 = np.argwhere(classific2 > 0)


    print("ENDE")"""



history = model.fit(ds_train,
                    validation_data=ds_val,
                    epochs=CONFIG["NUM_EPOCHS"],
                    )

model.save_weights(f"models/checkpoints/unet/weighted")
pd.DataFrame.from_dict(history.history).to_csv(f'models/checkpoints/unet/weighted.csv', index=False)