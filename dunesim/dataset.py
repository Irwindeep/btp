import os
import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Tuple, Any


class DunesimDataset(Dataset):
    def __init__(self, root: str, memory: int, future: int) -> None:
        super().__init__()

        self.root = root

        self.mem = memory
        self.future = future

        self.data_dirs = [os.path.join(self.root, dir) for dir in os.listdir(self.root)]

        self.samples = []
        for data_dir in self.data_dirs:
            bedrock_dir = os.path.join(data_dir, "bedrock")
            num_frames = len(os.listdir(bedrock_dir))
            for idx in range(self.mem, num_frames - self.future):
                self.samples.append((data_dir, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        data_dir, idx = self.samples[idx]

        bedrock_dir = os.path.join(data_dir, "bedrock")
        sediments_dir = os.path.join(data_dir, "sediments")

        vegetation_path = os.path.join(data_dir, "vegetation.npz")
        resistance_path = os.path.join(data_dir, "bedrock_hardness.npz")
        wind_field_path = os.path.join(data_dir, "wind_field.npz")

        ip_indices = range(idx - self.mem, idx)
        op_indices = range(idx, idx + self.future)

        ip_bedrocks = [os.path.join(bedrock_dir, f"{i:04d}.npz") for i in ip_indices]
        op_bedrocks = [os.path.join(bedrock_dir, f"{i:04d}.npz") for i in op_indices]

        ip_sediments = [os.path.join(sediments_dir, f"{i:04d}.npz") for i in ip_indices]
        op_sediments = [os.path.join(sediments_dir, f"{i:04d}.npz") for i in op_indices]

        input_bedrocks = np.stack([np.load(path)["arr_0"] for path in ip_bedrocks])
        input_sediments = np.stack([np.load(path)["arr_0"] for path in ip_sediments])
        input_array = np.stack([input_bedrocks, input_sediments], axis=1)

        output_bedrocks = np.stack([np.load(path)["arr_0"] for path in op_bedrocks])
        output_sediments = np.stack([np.load(path)["arr_0"] for path in op_sediments])
        output_array = np.stack([output_bedrocks, output_sediments], axis=1)

        input_tensor = torch.tensor(input_array)
        output_tensor = torch.tensor(output_array)

        auxiliary_array = np.stack(
            [
                np.load(vegetation_path)["arr_0"],
                np.load(resistance_path)["arr_0"],
            ],
        )
        auxiliary_array = np.concatenate(
            [
                auxiliary_array,
                np.load(wind_field_path)["arr_0"].transpose(2, 0, 1),
            ]
        )
        auxiliary_tensor = torch.tensor(auxiliary_array)

        return input_tensor, output_tensor, auxiliary_tensor
