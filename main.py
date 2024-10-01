import logging
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor 

logger = logging.getLogger(__name__)


@hydra.main(version_base='1.3.2', config_path='conf', config_name='config')
def main(cfg: DictConfig) -> float:

    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Instantiation
    data = instantiate(cfg.data)
    model = instantiate(cfg.model)

    # Wandb logger
    if cfg.wandb_use:
        wandb_logger = instantiate(cfg.wandb) 
    else:
        wandb_logger = False 
    
    # Training 
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg.model_dir,
        filename=cfg.model_save_name + '-{epoch:02d}-{val_loss:.2f}',
        mode ='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if wandb_logger:
        callbacks = [checkpoint_callback, lr_monitor] 
    else:
        callbacks = [checkpoint_callback] 
    trainer = Trainer(**cfg.pl_trainer,
                    logger=wandb_logger,
                    devices="auto",
                    callbacks=callbacks) 
    if not cfg.inference_only:
        trainer.fit(model, data, ckpt_path=cfg.checkpoint)

    # Testing
    checkpoint_path = 'best' if cfg.checkpoint is None else cfg.checkpoint
    trainer.test(model, ckpt_path=checkpoint_path, dataloaders=data)

    return 0


if __name__ == '__main__':
    main()
