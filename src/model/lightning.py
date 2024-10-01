import logging
import torch
import lightning as L

from src.model.graph_executor import GraphNeuralExecutor

logger = logging.getLogger(__name__)


class VCEPLightning(L.LightningModule):
    def __init__(self, model, train):
        super().__init__()
        self.save_hyperparameters(
            {'model': model, 'train': train},
            logger=False
        )
        self.model_cfg = model
        self.train_cfg = train
        self.model = GraphNeuralExecutor(
            hidden_dim=model.hidden_dim,
            eps=model.eps
        )
        self.eps = model.eps 

    def training_step(self, batch, batch_idx):
        batch_size = batch.x.size(0)
        results = self.model.teacher_forcing(batch, self.current_epoch)
        keys = ["x_del_loss", "x_loss", "y_loss", "optimal_set_loss"]
        if self.eps:
            keys.append("eps_loss")
        current_losses = torch.stack([results[key] for key in keys])
        total_loss = torch.sum(current_losses)
        self.log("train_loss", total_loss, batch_size=batch_size)
        for key, value in results.items():
            self.log(f"train_{key}", value, batch_size=batch_size)
        return total_loss 

    def validation_step(self, batch, batch_idx):
        batch_size = batch.x.size(0)
        results = self.model.teacher_forcing(batch)
        keys = ["x_del_loss", "x_loss", "y_loss", "optimal_set_loss"]
        if self.eps:
            keys.append("eps_loss")
        current_losses = torch.stack([results[key] for key in keys])
        total_loss = torch.sum(current_losses) 
        self.log("val_loss", total_loss, batch_size=batch_size)
        for key, value in results.items():
            self.log(f"val_{key}", value, batch_size=batch_size)
        return total_loss 

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch.x.size(0)
        results = self.model.test(batch)
        for key, value in results.items():
            self.log(f"test_{key}", value, batch_size=batch_size)
        return results

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.train_cfg.optimizer.lr,
            weight_decay = self.train_cfg.optimizer.weight_decay,
            betas=(self.train_cfg.optimizer.beta1, self.train_cfg.optimizer.beta2)
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    factor = self.train_cfg.scheduler.factor,
                    patience = self.train_cfg.scheduler.patience
                ),
                "monitor": "val_loss",
            },
        }
