from numpy import ndarray
import matplotlib.pyplot as plt
from math import ceil


class ImgUtils:
    @staticmethod
    def int_to_float_img(img: ndarray) -> ndarray:
        """ 
            Take a ndarray of (0, 255) and make it (0, 1).
            JPGs get read in as (0, 255), (0, 1) is often better.
        """
        return img / 255

    @staticmethod
    def scale_img(img: ndarray) -> ndarray:
        """ 
            Take a ndarray of (0, 1) and make it (-1, 1).
            Noising is done with (-1, 1).
        """
        return 2 * img - 1

    @staticmethod
    def unscale_img(img: ndarray) -> ndarray:
        """ 
            Take a ndarray of (-1, 1) and make it (0, 1).
            Imgs must be (0, 1) to show.
        """
        return ((img + 1) / 2).clip(0, 1)

    @staticmethod
    def show_images(imgs: list[ndarray], cols: int = 3, size: int = 4,
                    title: str = None, subtitles: list[str] = None) -> None:
        """
            Use matplotlib to show a list of images.

            Populates columns, then goes onto the next line.
            Plot size, title, and subtitles can be specified.
        """
        image_count = len(imgs)
        rows = ceil(image_count/cols)
        plt.figure(figsize=(cols*size, rows*size))
        for i, img in enumerate(imgs):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            if subtitles is not None:
                plt.title(subtitles[i])
            plt.axis('off')
        if title is not None:
            plt.suptitle(title)
        plt.show()
