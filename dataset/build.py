import os
import subprocess
import sys
import shutil
import platform
import glob

GIT_REPO = "https://github.com/Irwindeep/Desertscapes-Simulation.git"
CLONE_DIR = "build"


def run(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(result.returncode)


def find_shared_library(search_dir):
    ext = ".dll" if platform.system() == "Windows" else ".so"
    files = glob.glob(os.path.join(search_dir, f"**/*{ext}"), recursive=True)

    if len(files) == 0:
        print(f"No {ext} file found!")
        sys.exit(1)
    elif len(files) > 1:
        print(f"Multiple {ext} files found! Expected only one.")
        sys.exit(1)
    return files[0]


if __name__ == "__main__":
    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR)
    run(["git", "clone", GIT_REPO, CLONE_DIR])

    cwd = CLONE_DIR
    run(["make"], cwd=cwd)

    so_file = find_shared_library(cwd)
    print(f"Using shared library at {so_file}")

    dest_file = os.path.join(".", os.path.basename(so_file))
    shutil.copy2(so_file, dest_file)
    print(f"Copied {so_file} to {dest_file}")

    print("Cleaning up...")
    shutil.rmtree(CLONE_DIR)
    print("Done!")
