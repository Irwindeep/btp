import os
import subprocess
import pandas as pd
import numpy as np

np.random.seed(12)

repo_url = "https://github.com/aparis69/Desertscapes-Simulation"
clone_dir = "desertsim"

if not (os.path.exists(clone_dir) and len(os.listdir(clone_dir)) > 0):
    subprocess.run(["git", "clone", "-q", repo_url, clone_dir], check=True)

    print(f"Repository cloned to {clone_dir}")

if os.path.exists("data_config.csv"):
    print("Loaded Config from `data_config.csv`")
    df = pd.read_csv("data_config.csv")

else:
    data_params = [
        "rMin",
        "rMax",
        "windX",
        "windY",
        "vegetation",
        "abrasion",
        "preSimSteps",
    ]
    num_data_points = 1000 * int(input("# Data Samples (in thousands): "))

    df = pd.DataFrame({param: [] for param in data_params})

    num_cate = num_data_points // 5

    print(f"Generating Config for {num_cate:,} Transverse Dunes ...")
    for _ in range(num_cate):
        rMin = np.exp(np.random.uniform(low=np.log(2.0), high=np.log(6.0)))
        rMax = rMin + np.random.uniform(low=1.0, high=3.0)

        theta = np.random.uniform(low=0.0, high=2 * np.pi)
        r = np.random.uniform(low=3.0, high=6.0)
        windX = r * np.cos(theta)
        windY = r * np.sin(theta)

        preSimSteps = np.exp(np.random.uniform(low=np.log(50), high=np.log(300)))

        df.loc[len(df)] = [rMin, rMax, windX, windY, 0, 0, preSimSteps]

    print(f"Generating Config for {num_cate:,} Barchan Dunes ...")
    for _ in range(num_cate):
        rMin = np.exp(np.random.uniform(low=np.log(0.2), high=np.log(1.0)))
        rMax = rMin + np.random.uniform(low=1.0, high=2.0)

        theta = np.random.uniform(low=0.0, high=2 * np.pi)
        r = np.random.uniform(low=3.0, high=6.0)
        windX = r * np.cos(theta)
        windY = r * np.sin(theta)

        preSimSteps = np.exp(np.random.uniform(low=np.log(50), high=np.log(300)))

        df.loc[len(df)] = [rMin, rMax, windX, windY, 0, 0, preSimSteps]

    print(f"Generating Config for {num_cate:,} Nabkha Dunes ...")
    for _ in range(num_cate):
        rMin = np.exp(np.random.uniform(low=np.log(2.0), high=np.log(6.0)))
        rMax = rMin + np.random.uniform(low=1.0, high=3.0)

        theta = np.random.uniform(low=0.0, high=2 * np.pi)
        r = np.random.uniform(low=3.0, high=6.0)
        windX = r * np.cos(theta)
        windY = r * np.sin(theta)

        preSimSteps = np.exp(np.random.uniform(low=np.log(50), high=np.log(300)))

        df.loc[len(df)] = [rMin, rMax, windX, windY, 1, 0, preSimSteps]

    print(f"Generating Config for {num_cate:,} Yardangs Dunes ...")
    for _ in range(num_cate):
        rMin = np.exp(np.random.uniform(low=np.log(0.2), high=np.log(1.0)))
        rMax = rMin + np.exp(
            np.random.uniform(low=np.log(0.0 + 1e-8), high=np.log(0.5))
        )

        theta = np.random.uniform(low=0.0, high=2 * np.pi)
        r = np.random.uniform(low=3.0, high=6.0)
        windX = r * np.cos(theta)
        windY = r * np.sin(theta)

        preSimSteps = np.exp(np.random.uniform(low=np.log(100), high=np.log(400)))

        df.loc[len(df)] = [rMin, rMax, windX, windY, 0, 1, preSimSteps]

    print(f"Generating Config for {num_cate:,} Random Dunes ...")
    for _ in range(num_cate):
        rMin = np.exp(np.random.uniform(low=np.log(0.5), high=np.log(6.0)))
        rMax = rMin + np.random.uniform(low=0.0, high=3.0)

        theta = np.random.uniform(low=0.0, high=2 * np.pi)
        r = np.random.uniform(low=3.0, high=6.0)
        windX = r * np.cos(theta)
        windY = r * np.sin(theta)

        preSimSteps = 0
        df.loc[len(df)] = [
            rMin,
            rMax,
            windX,
            windY,
            np.random.choice(2),
            np.random.choice(2),
            preSimSteps,
        ]

    df.to_csv("data_config.csv", index=False)

cmd = [
    "g++",
    "-o",
    "dunesim",
    "dunesim.cpp",
    "desertsim/Code/Source/desert-flow.cpp",
    "desertsim/Code/Source/desert-simulation.cpp",
    "desertsim/Code/Source/desert.cpp",
    "-I",
    "desertsim/Code/Include",
]


try:
    subprocess.run(cmd, check=True)
    print("Compilation successful!")
except subprocess.CalledProcessError as e:
    print("Compilation failed with error code:", e.returncode)


subprocess.run(["./dunesim"], check=True)
