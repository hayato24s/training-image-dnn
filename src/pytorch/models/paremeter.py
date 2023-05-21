import os

import pytorch_lightning as pl

parameters_dir_name = os.path.join(os.path.dirname(__file__), "parameters")


def get_parameter_file_path(model: pl.LightningModule, lr: float) -> str:
    if not os.path.exists(parameters_dir_name):
        os.makedirs(parameters_dir_name, exist_ok=True)

    parameter_file_path = os.path.join(parameters_dir_name, "%s_lr=%f.pt" % (type(model).__name__, lr))

    return parameter_file_path
