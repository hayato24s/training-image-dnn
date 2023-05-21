import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class LightningLeNet5(pl.LightningModule):
    """
    References
    ----------
    [1] https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
    [2] https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
    """

    def __init__(self, n_classes, lr: float):
        super(LightningLeNet5, self).__init__()

        self.lr = lr

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = x.float()

        # (batch size, 1, 28, 28) -> (batch size, 120, 1, 1)
        x = self.feature_extractor(x)

        # (batch size, 120, 1, 1) -> (batch size, 120)
        x = torch.flatten(x, 1)

        # (batch size, 120) -> (batch size, n_classes)
        logits = self.classifier(x)

        probs = F.softmax(logits, dim=1)

        return logits, probs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.forward(x)

        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.forward(x)

        loss = F.cross_entropy(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
