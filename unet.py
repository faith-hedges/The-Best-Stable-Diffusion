###
# Heavy influence from
# https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb
###

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input
# Layer = Conv2D | MaxPooling2D | UpSampling2D | Concatenate | Input # 3.11 typing (no tensorflow yet)
from typing import Union, Tuple
Layer = Union[Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input]


class UNet():
    """ A down-then-up-scaling network with skip connections """
    @staticmethod
    def new(img_length = 64, n_downblocks = 4, downscale_factor=2) -> Model:
        # start with a simple input layer
        inputs = Input((img_length, img_length, 3))

        # filter_nums starts at img_length and gets divided by downscale_factor each time
        filter_nums = [img_length]
        for _ in range(1, n_downblocks+1):
            filter_nums.append(filter_nums[-1] // downscale_factor)

        # Form the down-blocks and skip connections
        skip_down_pairs = [UNet.down(inputs, filter_nums[0])]
        for i in range(1, n_downblocks):
            skip_down_pairs.append(UNet.down(skip_down_pairs[-1][1], filter_nums[i]))

        # Form the bottleneck
        bn = UNet.bottleneck(skip_down_pairs[-1][1], filter_nums[-1])

        # Form the up-blocks
        ups = [UNet.up(bn, skip_down_pairs[-1][0], filter_nums[-2])]
        for i in range(1, n_downblocks):
            ups.append(UNet.up(ups[-1], skip_down_pairs[-i-1][0], filter_nums[-i-2]))

        # Output layer
        outputs = Conv2D(3, (3, 3), padding="same",
                         strides=1, activation="relu")(ups[-1])

        assert len(filter_nums) - 1 == len(skip_down_pairs) == len(ups)
        return Model(inputs, outputs)

    @staticmethod
    def down(inp: Layer, n_filters: int, kernel_size=(3, 3), padding="same", strides=1) -> Tuple[Layer, Layer]:
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
