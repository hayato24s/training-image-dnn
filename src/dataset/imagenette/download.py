import os
import shutil
import urllib
from http.client import HTTPResponse

import dataset.imagenette.const as const


def chunk_read(response: HTTPResponse, chunk_size: int = 10**9) -> None:
    """
    Download data by chunk_size bytes.
    """

    while True:
        chunk = response.read(chunk_size)
        if chunk:
            yield chunk
        else:
            break


def download_imagenette() -> None:
    print("downloading imagenette data ...")

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

    os.makedirs(const.imagenette_storage_dir_name, exist_ok=True)

    tar_file = os.path.join(const.imagenette_dir_name, "_tmp.tgz")
    output_dir = const.imagenette_dir_name

    # download file
    response = urllib.request.urlopen(url)
    content_length = int(response.info()["Content-Length"])

    # save data in a tar file
    with open(tar_file, mode="wb") as f:
        # with open(tar_file, "wb") as f:
        downloaded_chunk = 0
        for chunk in chunk_read(response):
            f.write(chunk)
            downloaded_chunk += len(chunk)
            print("%d / %d" % (downloaded_chunk, content_length))

    # uppack the tar file
    shutil.unpack_archive(tar_file, output_dir)

    # remove the tar file
    os.remove(tar_file)

    # rename
    os.rename(os.path.join(const.imagenette_dir_name, "imagenette2-320"), const.imagenette_storage_dir_name)

    print("finished downloading imagenette data")
