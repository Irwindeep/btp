import os
import pandas as pd
import numpy as np

from dune import DuneSediment
from tqdm.auto import tqdm

np.random.seed(12)

DATA_DIR = "dunesim_dataset"
os.makedirs(DATA_DIR, exist_ok=True)

num_samples = 2000
nx, ny = 128, 128  # low resolution data for experiments

df = pd.DataFrame(
    {
        col: []
        for col in ["r_min", "r_max", "wind_x", "wind_y", "vegetation", "abrasion"]
    }
)

for i in tqdm(range(num_samples), desc="Preparing Dataset"):
    r_min = np.random.uniform(low=0.5, high=5)
    r_max = r_min + np.random.uniform(low=0.0, high=3.0)

    wind_speed = np.random.uniform(low=2.0, high=10.0)
    theta = np.random.uniform(low=0.0, high=2 * np.pi)

    wind_x = wind_speed * np.cos(theta)
    wind_y = wind_speed * np.sin(theta)

    vegetation = np.random.choice([0, 1], p=[0.7, 0.3])
    abration = np.random.choice([0, 1], p=[0.7, 0.3])

    r = np.random.uniform(0, 1)
    num_steps = int(100 + 100 * (r**4))

    dune = DuneSediment(
        nx,
        ny,
        r_min=r_min,
        r_max=r_max,
        wind=(wind_x, wind_y),
        vegetation_on=bool(vegetation),
        abrasion_on=bool(abration),
    )

    os.makedirs(os.path.join(DATA_DIR, f"{i:04d}"), exist_ok=True)

    bedrock_path = os.path.join(DATA_DIR, f"{i:04d}", "bedrock")
    sediment_path = os.path.join(DATA_DIR, f"{i:04d}", "sediments")
    vegetation_path = os.path.join(DATA_DIR, f"{i:04d}", "vegetation")

    os.makedirs(bedrock_path, exist_ok=True)
    os.makedirs(sediment_path, exist_ok=True)
    os.makedirs(vegetation_path, exist_ok=True)

    np.save(os.path.join(bedrock_path, f"{0:04d}.npy"), dune.bedrock)
    np.save(os.path.join(sediment_path, f"{0:04d}.npy"), dune.sediments)
    np.save(os.path.join(vegetation_path, f"{0:04d}.npy"), dune.vegetation)

    for step in range(1, num_steps + 1):
        dune.step()
        np.save(os.path.join(bedrock_path, f"{step:04d}.npy"), dune.bedrock)
        np.save(os.path.join(sediment_path, f"{step:04d}.npy"), dune.sediments)
        np.save(os.path.join(vegetation_path, f"{step:04d}.npy"), dune.vegetation)

    df.loc[len(df)] = [r_min, r_max, wind_x, wind_y, vegetation, abration]

df.to_csv("data_config.csv", index=False)
