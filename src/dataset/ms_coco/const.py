import os

import common.config as config

"""
Constant file and directory paths
"""

# Directory name
dataset_dir_name = os.path.abspath(os.path.dirname(__file__))
storage_dir_name = os.path.join(dataset_dir_name, "storage")

# MS-COCO
annotations_file_name = os.path.join(storage_dir_name, "annotations.json")
images_dir_name = os.path.join(storage_dir_name, "images")

# Preprocessed Data
used_image_ids_file_name = os.path.join(storage_dir_name, "used_image_ids.pkl")
caption_caches_file_name = os.path.join(storage_dir_name, "caption_caches.pkl")
compressed_image_caches_file_name_without_extension = os.path.join(storage_dir_name, "compressed_image_caches")
compressed_image_caches_file_name = compressed_image_caches_file_name_without_extension + ".npz"
if config.GPU:
    image_caches_file_name_without_extension = os.path.join("/content", "image_caches")
else:
    image_caches_file_name_without_extension = os.path.join(storage_dir_name, "image_caches")
image_caches_file_name = image_caches_file_name_without_extension + ".npz"
