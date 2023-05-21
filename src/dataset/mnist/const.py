import os

mnist_dir_name = os.path.dirname(__file__)
mnist_dataset_without_extension = os.path.join(mnist_dir_name, "dataset")
mnist_dataset_file_name = mnist_dataset_without_extension + ".npz"
