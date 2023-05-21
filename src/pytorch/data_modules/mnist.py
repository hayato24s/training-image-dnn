import pytorch_lightning as pl
from datasets.mnist import MNISTDataset
from torch.utils.data import DataLoader
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])

        self.mnist_train = MNISTDataset(is_train=True, transform=transform)
        self.mnist_test = MNISTDataset(is_train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(dataset=self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.mnist_test, batch_size=self.batch_size, shuffle=False)
