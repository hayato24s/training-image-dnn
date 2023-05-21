import pickle
from typing import Any, Dict, List

import dataset.imagenette.const as const

label_dict = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

label_to_id = {
    "tench": 0,
    "English springer": 1,
    "cassette player": 2,
    "chain saw": 3,
    "church": 4,
    "French horn": 5,
    "garbage truck": 6,
    "gas pump": 7,
    "golf ball": 8,
    "parachute": 9,
}

id_to_label = {
    0: "tench",
    1: "English springer",
    2: "cassette player",
    3: "chain saw",
    4: "church",
    5: "French horn",
    6: "garbage truck",
    7: "gas pump",
    8: "golf ball",
    9: "parachute",
}


def create_image_info(used_image_info: List[Dict[str, str]]) -> None:
    with open(const.imagenette_image_info_file_name, "wb") as f:
        pickle.dump(used_image_info, f)


def get_image_info(data_type="all") -> List[Any]:
    """
    Parameters
    ----------
    data_type : str, optional
        all, train or val, by default "all"

    Returns
    -------
    List[Dict[str, str]]
        { path: str, labe: str }

    Raises
    ------
    ValueError
        data_type must be all, train or val
    """

    if data_type != "all" and data_type != "train" and data_type != "val":
        raise ValueError("data_type must be all, train or val")

    with open(const.imagenette_image_info_file_name, "rb") as f:
        used_image_info = pickle.load(f)

        if data_type == "train":
            used_image_info = [image_info for image_info in used_image_info if not image_info["is_valid"]]
        elif data_type == "val":
            used_image_info = [image_info for image_info in used_image_info if image_info["is_valid"]]

    return used_image_info
