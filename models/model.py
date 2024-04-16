import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import platform

from settings.config import Home, GETLab

op_sys = platform.system()
CONFIG = Home if op_sys == "Windows" else GETLab


def create_derivatives_kernel():
    # Sobel derivative 3x3 X
    kernel_filter_dx_3 = np.float32(np.asarray([[-1, 0, 1],
                                                [-2, 0, 2],
                                                [-1, 0, 1]]))
    kernel_filter_dx_3 = kernel_filter_dx_3[..., np.newaxis]
    kernel_filter_dx_3 = kernel_filter_dx_3[..., np.newaxis]

    # Sobel derivative 3x3 Y
    kernel_filter_dy_3 = np.float32(np.asarray([[-1, -2, -1],
                                                [0, 0, 0],
                                                [1, 2, 1]]))
    kernel_filter_dy_3 = kernel_filter_dy_3[..., np.newaxis]
    kernel_filter_dy_3 = kernel_filter_dy_3[..., np.newaxis]
    return kernel_filter_dx_3, kernel_filter_dy_3

class Handcrafted(tf.keras.layers.Layer):
    """
    Custom Layer to compute gradient-based features of 1st and 2nd order of a greyscale image.
    See Key.Net-Paper.
    """
    def __init__(self):
        super(Handcrafted, self).__init__()

        # Sobel derivatives
        kernel_x, kernel_y = create_derivatives_kernel()
        self.kernel_filter_dx = tf.constant(kernel_x, name='kernel_filter_dx')
        self.kernel_filter_dy = tf.constant(kernel_y, name='kernel_filter_dy')

    def call(self, inputs):

        # Sobel_conv_derivativeX
        dx = tf.nn.conv2d(inputs, self.kernel_filter_dx, strides=[1, 1, 1, 1], padding='SAME')
        dxx = tf.nn.conv2d(dx, self.kernel_filter_dx, strides=[1, 1, 1, 1], padding='SAME')
        dx2 = tf.multiply(dx, dx)

        # Sobel_conv_derivativeY
        dy = tf.nn.conv2d(inputs, self.kernel_filter_dy, strides=[1, 1, 1, 1], padding='SAME')
        dyy = tf.nn.conv2d(dy, self.kernel_filter_dy, strides=[1, 1, 1, 1], padding='SAME')
        dy2 = tf.multiply(dy, dy)

        dxy = tf.nn.conv2d(dx, self.kernel_filter_dy, strides=[1, 1, 1, 1], padding='SAME')

        dxdy = tf.multiply(dx, dy)
        dxxdyy = tf.multiply(dxx, dyy)
        dxy2 = tf.multiply(dxy, dxy)

        features_t = tf.concat([dx, dx2, dxx, dy, dy2, dyy, dxdy, dxxdyy, dxy, dxy2], axis=3)

        # Rescaled
        dx = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/(tf.reduce_max(tf.abs(dx))))(dx)
        dxx = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/(tf.reduce_max(tf.abs(dxx))))(dxx)
        dx2 = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/(tf.reduce_max(tf.abs(dx2))))(dx2)
        dy = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/(tf.reduce_max(tf.abs(dy))))(dy)
        dyy = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/(tf.reduce_max(tf.abs(dyy))))(dyy)
        dy2 = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/(tf.reduce_max(tf.abs(dy2))))(dy2)
        dxy = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/(tf.reduce_max(tf.abs(dxy))))(dxy)
        dxdy = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/(tf.reduce_max(tf.abs(dxdy))))(dxdy)
        dxxdyy = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/(tf.reduce_max(tf.abs(dxxdyy))))(dxxdyy)
        dxy2 = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/(tf.reduce_max(tf.abs(dxy2))))(dxy2)
        features_tr = tf.concat([dx, dx2, dxx, dy, dy2, dyy, dxdy, dxxdyy, dxy, dxy2], axis=3)

        return features_tr

class VGG_Block(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, kernel_regularizer=None, use_relu=True, use_bn=True, use_bias=True):
        super(VGG_Block, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding="same",
                                           kernel_regularizer=kernel_regularizer, use_bias=use_bias)
        self.relu = tf.keras.layers.ReLU()
        self.bn = tf.keras.layers.BatchNormalization()
        self.use_relu = use_relu
        self.use_bn = use_bn

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.use_relu:
            x = self.relu(x)
        return x

class Upsample_Block(tf.keras.layers.Layer):
    """Upsamples an input.

    Conv2DTranspose => Batchnorm => Dropout => Relu

    Args:
        num_filters: number of filters
        kernel_size: filter size
        use_dropout: If True, adds the dropout layer
    """
    def __init__(self, num_filters, kernel_size, use_dropout=False, kernel_regularizer=None, use_bias=True):
        super(Upsample_Block, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.upconv = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size, strides=2, padding='same',
                                                      kernel_initializer=initializer,
                                                      kernel_regularizer=kernel_regularizer, use_bias=use_bias)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=None):
        x = self.upconv(inputs)
        x = self.bn(x, training=training)

        if self.use_dropout:
            x = self.dropout(x, training=training)

        x = self.relu(x)
        return x

class KeyNet(tf.keras.Model):
    """
    Inspired by the keyNet architecture:

    -handcrafted (gradient-based) filters of 1st and 2nd order
    -3 learnable blocks of conv(NxN)+bn+relu layers
    -final bn+conv(1x1) layer
    """
    def __init__(self, concat_handcrafted=False, use_gradients=True, num_filters=8, filter_size=5,
                 kernel_regularizer=None):
        super(KeyNet, self).__init__()

        self.concat_handcrafted = concat_handcrafted
        self.use_gradients = use_gradients

        # Architecture Design
        if isinstance(num_filters, tuple) and isinstance(filter_size, tuple):
            assert len(num_filters) == 3 and len(filter_size) == 4, "Bad args provided to construct conv layers! " \
                                                                    "len(num_filters) == 3 and len(filter_size) == 4"

            self.conv1 = tf.keras.layers.Conv2D(filters=num_filters[0], kernel_size=filter_size[0], padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv_1")
            self.conv2 = tf.keras.layers.Conv2D(filters=num_filters[1], kernel_size=filter_size[1], padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv_2")
            self.conv3 = tf.keras.layers.Conv2D(filters=num_filters[2], kernel_size=filter_size[2], padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv_3")
            self.conv_final = tf.keras.layers.Conv2D(1, kernel_size=filter_size[3], padding="same",
                                                     kernel_regularizer=kernel_regularizer, name="conv_final")
        else:
            self.conv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=filter_size, padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv_1")
            self.conv2 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=filter_size, padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv_2")
            self.conv3 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=filter_size, padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv_3")
            self.conv_final = tf.keras.layers.Conv2D(1, kernel_size=filter_size, padding="same",
                                                     kernel_regularizer=kernel_regularizer, name="conv_final")

        if self.use_gradients:
            self.handcrafted = Handcrafted()

        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
        self.relu1 = tf.keras.layers.ReLU(name="relu1")

        self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
        self.relu2 = tf.keras.layers.ReLU(name="relu2")

        self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")
        self.relu3 = tf.keras.layers.ReLU(name="relu3")

        self.bn_final = tf.keras.layers.BatchNormalization(name="bn_final")

    def call(self, inputs, training=None):

        # ensure greyscale image
        if inputs.shape[-1] == 3:
            inputs = tf.image.rgb_to_grayscale(inputs)

        if self.use_gradients:
            x = self.handcrafted(inputs)
        else:
            x = inputs

        if self.concat_handcrafted and self.use_gradients:
            x = tf.concat([inputs, x], axis=3)

        # 3 learnable blocks
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.relu3(x)

        x = self.bn_final(x, training=training)
        return self.conv_final(x)

class TinyKeyNet(tf.keras.Model):
    """
    Inspired by the keyNet architecture:

    -handcrafted (gradient-based) filters of 1st and 2nd order
    -1 learnable block of conv(NxN)+bn+relu layers
    -final bn+conv(1x1) layer
    """
    def __init__(self, concat_handcrafted=False, use_gradients=True, num_filters=8, filter_size=5,
                 kernel_regularizer=None):
        super(TinyKeyNet, self).__init__()

        self.concat_handcrafted = concat_handcrafted
        self.use_gradients = use_gradients

        # Architecture Design
        if isinstance(filter_size, tuple):
            assert len(filter_size) == 2, "Bad args provided to construct conv layers! len(filter_size) == 2"

            self.conv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=filter_size[0], padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv_1")
            self.conv_final = tf.keras.layers.Conv2D(1, kernel_size=filter_size[1], padding="same",
                                                     kernel_regularizer=kernel_regularizer, name="conv_final")
        else:
            self.conv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=filter_size, padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv_1")
            self.conv_final = tf.keras.layers.Conv2D(1, kernel_size=filter_size, padding="same",
                                                     kernel_regularizer=kernel_regularizer, name="conv_final")

        if self.use_gradients:
            self.handcrafted = Handcrafted()

        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
        self.relu1 = tf.keras.layers.ReLU(name="relu1")

        self.bn_final = tf.keras.layers.BatchNormalization(name="bn_final")

    def call(self, inputs, training=None):

        # ensure greyscale image
        if inputs.shape[-1] == 3:
            inputs = tf.image.rgb_to_grayscale(inputs)

        if self.use_gradients:
            x = self.handcrafted(inputs)
        else:
            x = inputs

        if self.concat_handcrafted and self.use_gradients:
            x = tf.concat([inputs, x], axis=3)

        # 1 learnable block
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.bn_final(x, training=training)
        return self.conv_final(x)

class DeepKeyNet(tf.keras.Model):
    """
    Inspired by the keyNet architecture:

    -handcrafted (gradient-based) filters of 1st and 2nd order
    -5 learnable blocks of conv(NxN)+bn+relu layers
    -final bn+conv(1x1) layer
    """
    def __init__(self, concat_handcrafted=False, use_gradients=True, use_dropout=False, num_filters=8, filter_size=5,
                 kernel_regularizer=None):
        super(DeepKeyNet, self).__init__()

        self.concat_handcrafted = concat_handcrafted
        self.use_gradients = use_gradients
        self.use_dropout = use_dropout

        # Architecture Design
        if isinstance(num_filters, tuple) and isinstance(filter_size, tuple):
            assert len(num_filters) == 5 and len(filter_size) == 6, "Bad args provided to construct conv layers! " \
                                                                    "len(num_filters) == 5 and len(filter_size) == 6"

            self.conv1 = tf.keras.layers.Conv2D(filters=num_filters[0], kernel_size=filter_size[0], padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv1")
            self.conv2 = tf.keras.layers.Conv2D(filters=num_filters[1], kernel_size=filter_size[1], padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv2")
            self.conv3 = tf.keras.layers.Conv2D(filters=num_filters[2], kernel_size=filter_size[2], padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv3")
            self.conv4 = tf.keras.layers.Conv2D(filters=num_filters[3], kernel_size=filter_size[3], padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv4")
            self.conv5 = tf.keras.layers.Conv2D(filters=num_filters[4], kernel_size=filter_size[4], padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv5")
            self.conv_final = tf.keras.layers.Conv2D(1, kernel_size=filter_size[5], padding="same",
                                                     kernel_regularizer=kernel_regularizer, name="conv_final")
        else:
            self.conv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=filter_size, padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv1")
            self.conv2 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=filter_size, padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv2")
            self.conv3 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=filter_size, padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv3")
            self.conv4 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=filter_size, padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv4")
            self.conv5 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=filter_size, padding="same",
                                                kernel_regularizer=kernel_regularizer, name="conv5")
            self.conv_final = tf.keras.layers.Conv2D(1, kernel_size=filter_size, padding="same",
                                                     kernel_regularizer=kernel_regularizer, name="conv_final")

        if self.use_gradients:
            self.handcrafted = Handcrafted()

        if self.use_dropout:
            self.drop1 = tf.keras.layers.Dropout(0.5)
            self.drop2 = tf.keras.layers.Dropout(0.5)
            self.drop3 = tf.keras.layers.Dropout(0.5)

        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
        self.relu1 = tf.keras.layers.ReLU(name="relu1")

        self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
        self.relu2 = tf.keras.layers.ReLU(name="relu2")

        self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")
        self.relu3 = tf.keras.layers.ReLU(name="relu3")

        self.bn4 = tf.keras.layers.BatchNormalization(name="bn4")
        self.relu4 = tf.keras.layers.ReLU(name="relu4")

        self.bn5 = tf.keras.layers.BatchNormalization(name="bn5")
        self.relu5 = tf.keras.layers.ReLU(name="relu5")

        self.bn_final = tf.keras.layers.BatchNormalization(name="bn_final")

    def call(self, inputs, training=None):

        # ensure greyscale image
        if inputs.shape[-1] == 3:
            inputs = tf.image.rgb_to_grayscale(inputs)

        if self.use_gradients:
            x = self.handcrafted(inputs)
        else:
            x = inputs

        if self.concat_handcrafted and self.use_gradients:
            x = tf.concat([inputs, x], axis=3)

        # 5 learnable blocks
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        if self.use_dropout:
            x = self.drop1(x, training=training)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x, training=training)
        if self.use_dropout:
            x = self.drop2(x, training=training)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x, training=training)
        if self.use_dropout:
            x = self.drop3(x, training=training)
        x = self.relu5(x)

        x = self.bn_final(x, training=training)
        return self.conv_final(x)

class SuperPoint(tf.keras.Model):
    """
    Inspired by the SuperPoint Architecture:
    Encoder -> spatial downsampling / dimension reduction
    Decoder -> upsamling (response/score map)
    """
    def __init__(self, concat_handcrafted=False, use_gradients=False, relu_final=False, use_dropout=False, kernel_regularizer=None,
                 block1=(64, 3),
                 block2=(64, 3),
                 block3=(64, 3),
                 block4=(64, 3),
                 block5=(128, 3),
                 block6=(128, 3),
                 block7=(128, 3),
                 block8=(128, 3),
                 block9=(256, 3),
                 grid_size=8):
        super(SuperPoint, self).__init__()

        self.grid_size = grid_size
        self.use_dropout = use_dropout
        self.concat_handcrafted = concat_handcrafted
        self.use_gradients = use_gradients

        if self.use_gradients:
            self.handcrafted = Handcrafted()

        self.encoder = []
        self.encoder.append(VGG_Block(block1[0], block1[1], kernel_regularizer=kernel_regularizer))
        self.encoder.append(VGG_Block(block2[0], block2[1], kernel_regularizer=kernel_regularizer))
        self.encoder.append(tf.keras.layers.MaxPool2D((2, 2), 2, padding='same'))
        self.encoder.append(VGG_Block(block3[0], block3[1], kernel_regularizer=kernel_regularizer))
        self.encoder.append(VGG_Block(block4[0], block4[1], kernel_regularizer=kernel_regularizer))
        self.encoder.append(tf.keras.layers.MaxPool2D((2, 2), 2, padding='same'))
        if self.use_dropout:
            self.encoder.append(tf.keras.layers.Dropout(0.5))
        self.encoder.append(VGG_Block(block5[0], block5[1], kernel_regularizer=kernel_regularizer))
        self.encoder.append(VGG_Block(block6[0], block6[1], kernel_regularizer=kernel_regularizer))
        self.encoder.append(tf.keras.layers.MaxPool2D((2, 2), 2, padding='same'))
        if self.use_dropout:
            self.encoder.append(tf.keras.layers.Dropout(0.5))
        self.encoder.append(VGG_Block(block7[0], block7[1], kernel_regularizer=kernel_regularizer))
        self.encoder.append(VGG_Block(block8[0], block8[1], kernel_regularizer=kernel_regularizer))
        self.encoder.append(VGG_Block(block9[0], block9[1], kernel_regularizer=kernel_regularizer))
        if self.use_dropout:
            self.encoder.append(tf.keras.layers.Dropout(0.5))
        self.encoder.append(VGG_Block(pow(self.grid_size, 2), 1, use_relu=relu_final, kernel_regularizer=kernel_regularizer))

    def call(self, inputs, training=None):

        # ensure greyscale image
        if inputs.shape[-1] == 3:
            inputs = tf.image.rgb_to_grayscale(inputs)

        if self.use_gradients:
            x = self.handcrafted(inputs)
        else:
            x = inputs

        if self.concat_handcrafted and self.use_gradients:
            x = tf.concat([inputs, x], axis=3)

        for layer in self.encoder:
            x = layer(x, training=training)
        x = tf.nn.depth_to_space(x, self.grid_size)

        return x

class Unet(tf.keras.Model):
    """
    Encoder-Decoder pair commonly used in Image Segmentation. The architecture consists of two (symmetrical) branches:
    - Encoder: contracting path used for downsampling
    - Decoder: expanding path used for upsampling

    Arguments:
        use_skips: whether to concatenate outputs from the contracting path during upsampling.
    """
    def __init__(self, use_skips=True, concat_handcrafted=False, use_gradients=False, use_dropout=False,
                 kernel_regularizer=None, relu_final=False, bn_final=False, use_bias=True,
                 architecture=((32, 3), (64, 3), (128, 3), (256, 3), (512, 3))
                 ):
        super(Unet, self).__init__()

        assert len(architecture) <= 5, "'architecture' only supports max. 5 pooling steps, as image dimensions" \
                                       "must be divisible by 32 (2^5)."

        self.use_skips = use_skips
        self.concat_handcrafted = concat_handcrafted
        self.use_gradients = use_gradients

        if self.use_gradients:
            self.handcrafted = Handcrafted()

        # Encoder
        self.downstack = []
        for idx, block in enumerate(architecture):
            self.downstack.append(VGG_Block(block[0], block[1],
                                            #kernel_regularizer=kernel_regularizer,
                                            use_bias=use_bias))
            self.downstack.append(VGG_Block(block[0], block[1],
                                            #kernel_regularizer=kernel_regularizer,
                                            use_bias=use_bias))
            self.downstack.append(tf.keras.layers.MaxPool2D((2, 2), 2, padding='same', name='pool' + str(idx+1)))

        # Decoder
        self.upstack = []
        for idx, block in enumerate(reversed(architecture)):
            # only last 3 blocks should use dropout if use_dropout=True
            if idx < len(architecture)-3:
                self.upstack.append(Upsample_Block(block[0], block[1], use_dropout=False,
                                                   #kernel_regularizer=kernel_regularizer,
                                                   use_bias=use_bias))
            else:
                self.upstack.append(Upsample_Block(block[0], block[1], use_dropout=use_dropout,
                                                   kernel_regularizer=kernel_regularizer,
                                                   use_bias=use_bias))

        self.last = VGG_Block(1, 3, use_relu=relu_final, use_bn=bn_final, kernel_regularizer=kernel_regularizer,
                              use_bias=True)

    def call(self, inputs, training=None):

        # ensure greyscale image
        if inputs.shape[-1] == 3:
            inputs = tf.image.rgb_to_grayscale(inputs)

        if self.use_gradients:
            x = self.handcrafted(inputs)
        else:
            x = inputs

        if self.concat_handcrafted and self.use_gradients:
            x = tf.concat([inputs, x], axis=3)

        x, skips = self.encode(x, training=training)
        skips = reversed(skips)
        x = self.decode(x, skips, training=training)

        return x

    def encode(self, x, training=None):
        skips = []

        for layer in self.downstack:
            if 'pool' in layer.name:
                skips.append(x)
            x = layer(x, training=training)

        return x, skips

    def decode(self, x, skips, training=None):

        if self.use_skips:
            for up, skip in zip(self.upstack, skips):
                x = up(x, training=training)
                concat = tf.keras.layers.Concatenate()
                x = concat([x, skip])
        else:
            for up in self.upstack:
                x = up(x, training=training)

        x = self.last(x)
        return x

class UnetMod(tf.keras.Model):
    """
    Modified U-Net architecture consisting of an Encoder-Decoder pair. This model optionally uses a pretrained ImageNet
    model as the encoder to overcome overfitting issues by the approach of transfer learning.
    """
    def __init__(self, use_pretrained_encoder=False, concat_handcrafted=False, use_gradients=False, use_dropout=False,
                 kernel_regularizer=None, use_bias=True,
                 up_blocks=((512, 3), (256, 3), (128, 3), (64, 3))):
        super(UnetMod, self).__init__()

        assert len(up_blocks) == 4, f"This architecture needs exactly 4 upsample blocks. {len(up_blocks)} blocks provided."
        if use_pretrained_encoder and use_gradients:
            raise ValueError('Cannot use pretrained imagenet weights in combination with gradient features.')

        weights = 'imagenet' if use_pretrained_encoder else None
        if weights == 'imagenet':
            inp_shape = [CONFIG["H"], CONFIG["W"], 3]
        else:
            inp_shape = [CONFIG["H"], CONFIG["W"], 1]
        if use_gradients:
            inp_shape[2] = 13

        base_model = tf.keras.applications.MobileNetV2(input_shape=inp_shape, include_top=False, weights=weights)
        # preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        # model expects pixel values in [-1, 1]

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',
            'block_3_expand_relu',
            'block_6_expand_relu',
            'block_13_expand_relu',
            'block_16_project',
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        self.use_pretrained_encoder = use_pretrained_encoder
        self.concat_handcrafted = concat_handcrafted
        self.use_gradients = use_gradients

        if self.use_gradients:
            self.handcrafted = Handcrafted()

        # Create the feature extraction model
        self.down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
        self.down_stack.trainable = not use_pretrained_encoder

        # Decoder
        self.up_stack = []
        for idx, block in enumerate(up_blocks):
            # only last 3 blocks should use dropout if use_dropout=True
            if idx < len(up_blocks)-3:
                self.up_stack.append(Upsample_Block(block[0], block[1], use_dropout=False,
                                                    #kernel_regularizer=kernel_regularizer,
                                                    use_bias=use_bias))
            else:
                self.up_stack.append(Upsample_Block(block[0], block[1], use_dropout=use_dropout,
                                                    kernel_regularizer=kernel_regularizer,
                                                    use_bias=use_bias))

        self.last = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same',
                                                    kernel_regularizer=kernel_regularizer,
                                                    use_bias=True)

    def call(self, inputs, training=None):

        # ensure greyscale image
        if inputs.shape[-1] == 3 and not self.use_pretrained_encoder:
            inputs = tf.image.rgb_to_grayscale(inputs)

        if self.use_gradients:
            x = self.handcrafted(inputs)
        else:
            x = inputs

        if self.concat_handcrafted and self.use_gradients:
            x = tf.concat([inputs, x], axis=3)

        # Downsampling through the model
        skips = self.down_stack(x, training=training)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x, training=training)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        x = self.last(x)
        return x