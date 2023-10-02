import glob
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from lightning import LightningDataModule
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import RotateAxisAngle
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


class OakInkPointDataset(Dataset):
    def __init__(self, num_points: int = 1024):
        super().__init__()

        statistics = []
        for filepath in glob.glob("data/oakink_shadow_dataset_valid_new/*/*.json", recursive=True):
            with open(filepath) as f:
                data = json.load(f)
            statistics.append({"category": data["category"], "code": data["object_code"]})
        statistics = pd.DataFrame(statistics).drop_duplicates().reset_index(drop=True)
        statistics["category"] = statistics["category"].astype("category").cat.codes

        filepaths = []
        categories = []
        codes = []
        for _, item in statistics.iterrows():
            filepath = Path("assets") / "oakink" / item["code"] / "align" / "decomposed.obj"
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
    def __init__(self, data_dir: str = "data/"):
        pass

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        return super().setup(stage)

    def train_dataloader(self) -> DataLoader[Any]:
        return super().train_dataloader()

    def val_dataloader(self) -> DataLoader[Any]:
        return super().val_dataloader()

    def test_dataloader(self) -> DataLoader[Any]:
        return super().test_dataloader()
