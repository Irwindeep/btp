import os
import numpy as np

from wind import generate_random_wind_field
from dune import DuneSediment
from tqdm.auto import tqdm

np.random.seed(12)

DATA_DIR = "dunesim_dataset"
os.makedirs(DATA_DIR, exist_ok=True)

num_samples = 1000
nx, ny = 128, 128  # low resolution data for experiments

for i in tqdm(range(num_samples), desc="Preparing Dataset"):
    r_min = np.random.uniform(low=0.5, high=5)
    r_max = r_min + np.random.uniform(low=0.0, high=3.0)

    mean_wind_speed = np.random.uniform(low=2.0, high=10.0)
    theta = np.random.uniform(low=0.0, high=2 * np.pi)

    mean_wind_x = mean_wind_speed * np.cos(theta)
    mean_wind_y = mean_wind_speed * np.sin(theta)

    r = np.random.uniform(0, 1)
    num_steps = 5 * int(30 + 50 * (r**4))

    wind_field = generate_random_wind_field(
        N=nx,
        timesteps=1,
        mean_wind=(mean_wind_x, mean_wind_y),
        random_seed=i,
    )

    vegetation = np.random.choice([0, 1], p=[0.7, 0.3])
    abration = np.random.choice([0, 1], p=[0.7, 0.3])

    dune = DuneSediment(
        nx,
        ny,
        r_min=r_min,
        r_max=r_max,
        wind=(mean_wind_x, mean_wind_y),
        vegetation_on=bool(vegetation),
        abrasion_on=bool(abration),
    )

    dune.wind_x = wind_field[0, :, :, 0]
    dune.wind_y = wind_field[0, :, :, 1]

    os.makedirs(os.path.join(DATA_DIR, f"{i:04d}"), exist_ok=True)

    bedrock_path = os.path.join(DATA_DIR, f"{i:04d}", "bedrock")
    sediment_path = os.path.join(DATA_DIR, f"{i:04d}", "sediments")

    wind_field_path = os.path.join(DATA_DIR, f"{i:04d}", "wind_field.npz")
    vegetation_path = os.path.join(DATA_DIR, f"{i:04d}", "vegetation.npz")
    hardness_path = os.path.join(DATA_DIR, f"{i:04d}", "bedrock_hardness.npz")

    os.makedirs(bedrock_path, exist_ok=True)
    os.makedirs(sediment_path, exist_ok=True)

    np.savez_compressed(vegetation_path, dune.vegetation)
    np.savez_compressed(hardness_path, dune.bedrock_hardness)
    np.savez_compressed(
        wind_field_path,
        np.stack([dune.wind_x, dune.wind_y], axis=-1),
    )

    np.savez_compressed(os.path.join(bedrock_path, f"{0:04d}.npz"), dune.bedrock)
    np.savez_compressed(os.path.join(sediment_path, f"{0:04d}.npz"), dune.sediments)

    for step in range(1, num_steps + 1):
        dune.step()

        np.savez_compressed(os.path.join(bedrock_path, f"{step:04d}.npz"), dune.bedrock)
        np.savez_compressed(
            os.path.join(sediment_path, f"{step:04d}.npz"), dune.sediments
        )


print(f"Dataset prepared successfully at - `{DATA_DIR}`")
