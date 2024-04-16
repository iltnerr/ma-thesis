import tensorflow as tf

msg = tf.constant('Hello, TensorFlow!')
tf.print(msg)

print("Tensorflow Version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))