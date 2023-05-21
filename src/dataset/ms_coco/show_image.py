import os

from PIL import Image

import dataset.ms_coco.const as const


def show_image(image_id: str) -> None:
    """
    Show image.

    Parameters
    ----------
    image_id : str
        image id.
    """

    image_file_name = os.path.join(const.images_dir_name, "COCO_val2014_%s.jpg" % image_id)
    image = Image.open(image_file_name)
    image.show()
