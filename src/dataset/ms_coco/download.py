import os
import shutil
import urllib.request
from http.client import HTTPResponse

import dataset.ms_coco.const as const


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


def download_zip(download_url: str, output_dir: str) -> None:
    """
    Download zip file from download_url and save the zip file under output_dir.
    """

    zip_file = os.path.join(const.storage_dir_name, "_tmp.zip")

    # download file
    response = urllib.request.urlopen(download_url)
    content_length = int(response.info()["Content-Length"])

    # save data in a zip file
    with open(zip_file, "wb") as f:
        downloaded_chunk = 0
        for chunk in chunk_read(response):
            f.write(chunk)
            downloaded_chunk += len(chunk)
            print("%d / %d" % (downloaded_chunk, content_length))

    # uppack the zip file
    shutil.unpack_archive(zip_file, output_dir)

    # remove the zip file
    os.remove(zip_file)


def download_annotations() -> None:
    """
    Download '2014 Train/Val annotations [241MB]' from MS-COCO dataset (https://cocodataset.org/#download).
    We use only val annotations.
    """

    print("downloading annotations ...")

    download_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    output_dir = os.path.join(const.storage_dir_name, "_tmp")

    download_zip(download_url, output_dir)
    os.rename(os.path.join(output_dir, "annotations", "captions_val2014.json"), const.annotations_file_name)
    shutil.rmtree(output_dir)

    print("finished downloading annotations.")


def download_images() -> None:
    """
    Download '2014 Val images [41K/6GB]' from MS-COCO dataset (https://cocodataset.org/#download).
    We use only val images.
    """

    print("downloading images ...")

    download_url = "http://images.cocodataset.org/zips/val2014.zip"
    output_dir = os.path.join(const.storage_dir_name, "_tmp")

    download_zip(download_url, output_dir)
    os.rename(os.path.join(output_dir, "val2014"), const.images_dir_name)
    shutil.rmtree(output_dir)

    print("finished downloading images.")
