import os

import common.config as config

imagenette_dir_name = os.path.abspath(os.path.dirname(__file__))
imagenette_storage_dir_name = os.path.join(imagenette_dir_name, "storage")
imagenette_csv_file_name = os.path.join(imagenette_storage_dir_name, "noisy_imagenette.csv")
imagenette_image_info_file_name = os.path.join(imagenette_storage_dir_name, "image_info.pkl")
if config.GPU:
    imagenette_image_caches_without_extension = os.path.join("/content", "imagenette_image_caches")
else:
    imagenette_image_caches_without_extension = os.path.join(imagenette_storage_dir_name, "image_caches")
imagenette_image_caches = imagenette_image_caches_without_extension + ".npz"
