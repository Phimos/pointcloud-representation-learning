import glob
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from joblib import Parallel, delayed
from lightning import LightningDataModule
from pytorch3d.io.obj_io import (
    Device,
    Meshes,
    PathManager,
    TexturesAtlas,
    TexturesUV,
    join_meshes_as_batch,
    load_obj,
)
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def load_single_obj(
    index: int,
    filename: str,
    device: Optional[Device] = None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    path_manager: Optional[PathManager] = None,
):
    verts, faces, aux = load_obj(
        filename,
        load_textures=load_textures,
        create_texture_atlas=create_texture_atlas,
        texture_atlas_size=texture_atlas_size,
        texture_wrap=texture_wrap,
        path_manager=path_manager,
    )
    tex = None
    if create_texture_atlas:
        # TexturesAtlas type
        tex = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
    else:
        # TexturesUV type
        tex_maps = aux.texture_images
        if tex_maps is not None and len(tex_maps) > 0:
            verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
            faces_uvs = faces.textures_idx.to(device)  # (F, 3)
            image = list(tex_maps.values())[0].to(device)[None]
            tex = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)

    mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex)
    return (index, mesh)


def load_objs_as_meshes(
    files: list,
    device: Optional[Device] = None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    path_manager: Optional[PathManager] = None,
):
    """Load meshes from a list of .obj files using the load_obj function, and return them as a Meshes object. This only
    works for meshes which have a single texture image for the whole mesh. See the load_obj function for more details.
    material_colors and normals are not stored.

    Args:
        files: A list of file-like objects (with methods read, readline, tell,
            and seek), pathlib paths or strings containing file names.
        device: Desired device of returned Meshes. Default:
            uses the current device for the default tensor type.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas, texture_atlas_size, texture_wrap: as for load_obj.
        path_manager: optionally a PathManager object to interpret paths.

    Returns:
        New Meshes object.
    """
    mesh_list = Parallel(n_jobs=48, verbose=100)(
        delayed(load_single_obj)(
            index,
            filename,
            device=device,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
        )
        for index, filename in enumerate(files)
    )
    mesh_list = sorted(mesh_list, key=lambda x: x[0])
    mesh_list = [mesh for _, mesh in mesh_list]

    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes_as_batch(mesh_list)


class OakInkPointDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "/root/autodl-tmp/project/func-mani",
        grasping_dir: str = "data/oakink_shadow_dataset_valid_new",
        num_points: int = 1024,
    ):
        super().__init__()

        grasp_config_dir = Path(root_dir) / grasping_dir

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
        self.pointclouds *= 100.0  # unit: m -> cm
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


class OakInkGraspPointDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "/root/autodl-tmp/project/func-mani",
        grasping_dir: str = "data/oakink_shadow_func_grasp_meshes",
        num_points: int = 1024,
    ):
        super().__init__()

        grasp_mesh_dir = Path(root_dir) / grasping_dir
        filepaths = list(grasp_mesh_dir.glob("*.pt"))

        pointclouds = []
        names = []
        for filepath in tqdm(filepaths):
            pointcloud = torch.load(filepath).to(torch.float32)
            if pointcloud.ndim != 2 or pointcloud.shape[0] != 2048 or pointcloud.shape[1] != 3:
                print(filepath, pointcloud.shape)
                continue
            pointclouds.append(pointcloud)
            names.append(filepath.stem)
            # pointclouds.append(torch.load(filepath))
        pointclouds = torch.stack(pointclouds, dim=0)
        pointclouds = torch.einsum("b n d -> b d n", pointclouds)
        pointclouds *= 100.0

        self.pointclouds = pointclouds
        self.num_samples = self.pointclouds.shape[0]
        self.codes = names

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> dict:
        return {"pointcloud": self.pointclouds[index]}


class OakInkDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str = "/root/autodl-tmp/project/func-mani",
        grasping_dir: str = "data/oakink_shadow_dataset_valid_new",
        grasp: bool = False,
        num_points: int = 1024,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.grasping_dir = grasping_dir
        self.grasp = grasp
        self.num_points = num_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            if self.grasp:
                dataset = OakInkGraspPointDataset(
                    root_dir=self.root_dir, grasping_dir=self.grasping_dir, num_points=self.num_points
                )
            else:
                dataset = OakInkPointDataset(
                    root_dir=self.root_dir, grasping_dir=self.grasping_dir, num_points=self.num_points
                )
            num_samples = len(dataset)
            num_train = int(num_samples * 0.8)
            num_val = int(num_samples * 0.1)
            num_test = num_samples - num_train - num_val
            self.dataset = dataset
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, [num_train, num_val, num_test], generator=torch.Generator().manual_seed(42)
            )

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
