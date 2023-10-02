import glob
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from lightning import LightningDataModule
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


class OakInkPointDataset(Dataset):
    def __init__(self, root_dir: str = "/root/autodl-tmp/project/func-mani", num_points: int = 1024):
        super().__init__()

        grasp_config_dir = Path(root_dir) / "data/oakink_shadow_dataset_valid_new"

        statistics = []
        for filepath in glob.glob(str(grasp_config_dir) + "/*/*.json", recursive=True):
            with open(filepath) as f:
                data = json.load(f)
            statistics.append({"category": data["category"], "code": data["object_code"]})
        statistics = pd.DataFrame(statistics).drop_duplicates().reset_index(drop=True)
        statistics["category"] = statistics["category"].astype("category").cat.codes

        # collect filepaths
        filepaths = []
        categories = []
        codes = []
        mesh_dir = Path(root_dir) / "assets" / "oakink"
        for _, item in statistics.iterrows():
            filepath = mesh_dir / item["code"] / "align" / "decomposed.obj"
            if not filepath.exists():
                continue
            filepaths.append(filepath)
            categories.append(item["category"])
            codes.append(item["code"])

        meshes = load_objs_as_meshes(filepaths, load_textures=False, device="cpu")
        self.pointclouds = sample_points_from_meshes(meshes, num_samples=num_points)
        self.pointclouds = torch.einsum("b n d -> b d n", self.pointclouds)
        self.categories = torch.tensor(categories, dtype=torch.long)
        self.codes = codes

        self.num_categories = int(torch.max(self.categories) + 1)
        self.num_points = num_points
        self.num_samples = self.categories.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> dict:
        return {
            "pointcloud": self.pointclouds[index],
            "category": self.categories[index],
        }


class OakInkDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        num_points: int = 1024,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_points = num_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        dataset = OakInkPointDataset(num_points=self.num_points)
        num_samples = len(dataset)
        num_train = int(num_samples * 0.8)
        num_val = int(num_samples * 0.1)
        num_test = num_samples - num_train - num_val
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [num_train, num_val, num_test])

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
