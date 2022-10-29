###
# Heavy influence from
# https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb
###

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input
# Layer = Conv2D | MaxPooling2D | UpSampling2D | Concatenate | Input # 3.11 typing (no tensorflow yet)
from typing import Union
Layer = Union[Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input]


class UNet():
    """ A down-then-up-scaling network with skip connections, assumes 256x256 """
    @staticmethod
    def new() -> Model:
        inputs = Input((256, 256, 3))
        filter_nums = [16, 32, 64, 128, 256]

        skip1, down1 = UNet.down(inputs, filter_nums[0])  # \
        skip2, down2 = UNet.down(down1, filter_nums[1])  # \
        skip3, down3 = UNet.down(down2, filter_nums[2])  # \
        skip4, down4 = UNet.down(down3, filter_nums[3])  # \

        bn = UNet.bottleneck(down4, filter_nums[4])  # -

        up1 = UNet.up(bn, skip4, filter_nums[3])  # /
        up2 = UNet.up(up1, skip3, filter_nums[2])  # /
        up3 = UNet.up(up2, skip2, filter_nums[1])  # /
        up4 = UNet.up(up3, skip1, filter_nums[0])  # /

        outputs = Conv2D(3, (3, 3), padding="same",
                         strides=1, activation="relu")(up4)

        return Model(inputs, outputs)

    @staticmethod
    def down(inp: Layer, n_filters: int, kernel_size=(3, 3), padding="same", strides=1) -> tuple[Layer, Layer]:
        """ 
        Run the input through two convolutional layers, then return that 
        output as well as a 2x downscale. One output for skip, one not.
        """
        conv1 = Conv2D(n_filters, kernel_size,
                       padding=padding, strides=strides, activation="relu")(inp)
        conv2 = Conv2D(n_filters, kernel_size,
                       padding=padding, strides=strides, activation="relu")(conv1)
        shrink = MaxPooling2D((2, 2), (2, 2))(conv2)
        return conv2, shrink  # conv2 is "skipped" forward, shrink continues down through the U

    @staticmethod
    def up(inp: Layer, skip: Layer, n_filters: int, kernel_size=(3, 3), padding="same", strides=1) -> Layer:
        """
        Combine the input from the U with the skip layer, then run
        that through two convolutional layers and return the output.
        """
        expand = UpSampling2D((2, 2))(inp)
        combin = Concatenate()([expand, skip])
        conv1 = Conv2D(n_filters, kernel_size,
                       padding=padding, strides=strides, activation="relu")(combin)
        conv2 = Conv2D(n_filters, kernel_size,
                       padding=padding, strides=strides, activation="relu")(conv1)
        return conv2

    @staticmethod
    def bottleneck(inp: Layer, n_filters, kernel_size=(3, 3), padding="same", strides=1) -> Layer:
        """
        Just two convolutional layers, no down/up sampling
        """
        conv1 = Conv2D(n_filters, kernel_size,
                       padding=padding, strides=strides, activation="relu")(inp)
        conv2 = Conv2D(n_filters, kernel_size,
                       padding=padding, strides=strides, activation="relu")(conv1)
        return conv2
