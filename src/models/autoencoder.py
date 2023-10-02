from typing import Any, Dict, Tuple

import torch
from components.decoder import Decoder
from components.pointnetpp import Net
from lightning import LightningModule
from pytorch3d.loss import chamfer_distance
from torchmetrics import MaxMetric, MeanMetric


class AutoEncoder(LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile
        self.decoder = Decoder(1024, 2048)

        self.criterion = chamfer_distance

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def model_step(self, batch, batch_idx):
        x, pos, y = batch
        x_hat = self.net(x)
        loss = chamfer_distance(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        x, pos, y = batch
        x_hat = self.net(x)
        loss = chamfer_distance(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization. Normally you'd need one. But
        in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
