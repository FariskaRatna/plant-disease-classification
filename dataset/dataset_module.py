from typing import Any, Optional, Dict
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class PlantDiseaseDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            batch_size: int = 32,
            num_classes: int = 8,
            num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.transforms_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0, contrast=0.2, saturation=0, hue=0)
        ])

        self.transforms_test = transforms.Compose([
            transforms.resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.transforms_val = transforms.Compose([
            transforms.resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        @property
        def num_classes(self) -> int:
            return 8
        
        def prepare_data(self) -> None:
            