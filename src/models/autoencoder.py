from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from pytorch3d.loss import chamfer_distance as _chamfer_distance


def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)
    y = y.transpose(1, 2)
    return _chamfer_distance(x, y)[0]


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
        self.save_hyperparameters(logger=False, ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile

        self.criterion = chamfer_distance

    def forward(self, x):
        x = self.encoder.encode(x)
        x = self.decoder(x)
        return x

    def model_step(self, batch, batch_idx):
        x = batch["pointcloud"]
        x_hat = self.forward(x)
        loss = self.criterion(x, x_hat)
        return x_hat, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.model_step(batch, batch_idx)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss = self.model_step(batch, batch_idx)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        _, loss = self.model_step(batch, batch_idx)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

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
