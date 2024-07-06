import torch
import torch.nn as nn
import torch
import pytorch_lightning as pl
from typing import Dict
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

from scripts.classifier import Classifier
from dataset.dataset import (
    create_data_loaders,
    perform_data_split,
    count_images_per_class,
)


class TrainConfig:
    def __init__(
        self, dataset_root: Path, max_epochs: int, batch_size: int, optimizer: Dict
    ):
        self.dataset_root = dataset_root
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer


def train(cfg: TrainConfig) -> None:
    wandb.finish()

    wandb.init(
        project="pytorch-plant-disease",
        config={
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.optimizer["learning_rate"],
            "epochs": cfg.max_epochs,
            "optimizer": cfg.optimizer["type"],
        },
    )

    logger = WandbLogger(name="Wandb", project="pytorch-plant-disease")

    checkpoint_callback = ModelCheckpoint(
        dirpath="scripts/checkpoints",
        filename="epoch={epoch}-step={global_step}",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback],
        # accelerator = "cuda",
        devices=1,
        strategy="auto",
        enable_model_summary=True,
        logger=logger,
    )

    model = Classifier(n_classes=8)
    data_path = Path("data/plant_disease")

    count_images_per_class(data_path)

    perform_data_split(data_path)

    # Define data directories
    train_data_dir = "train-test-splitted/train"
    test_data_dir = "train-test-splitted/test"
    val_data_dir = "train-test-splitted/val"

    # Create datasets and data loaders
    train_loader, validation_loader, test_loader, train_data, test_data = (
        create_data_loaders(train_data_dir, test_data_dir, val_data_dir)
    )

    trainer.fit(model, train_loader, validation_loader)
    # torch.save(
    #     model.state_dict(), "/scripts/Modified_CNN.pt"
    # )

    trainer.test(model, test_loader)

    wandb.finish()


if __name__ == "__main__":
    config_values = {
        "max_epochs": 20,
        "dataset_root": Path("/kaggle/input/plant-disease-dataset/plant_disease"),
        "batch_size": 32,
        "optimizer": {"type": "Adam", "learning_rate": 1e-3},
    }

    train_cfg = TrainConfig(**config_values)
    train(train_cfg)
