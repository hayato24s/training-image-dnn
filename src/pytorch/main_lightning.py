import hydra
import pytorch_lightning as pl
import torch
from data_modules.mnist import MNISTDataModule
from models.lightning_model import LightningLeNet5
from models.paremeter import get_parameter_file_path
from omegaconf import DictConfig

RANDOM_SEED = 42
N_CLASSES = 10
BATCH_SIZE = 64
N_EPOCHS = 5


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    lr = cfg["lr"]

    torch.manual_seed(RANDOM_SEED)
    data_module = MNISTDataModule(BATCH_SIZE)
    model = LightningLeNet5(N_CLASSES, lr)

    trainer = pl.Trainer(max_epochs=N_EPOCHS)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    parameter_file_path = get_parameter_file_path(model, lr)
    torch.save(model, parameter_file_path)


if __name__ == "__main__":
    run()
