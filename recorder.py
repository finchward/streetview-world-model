from config import Config
from simulator import Simulator
import asyncio
import argparse
import torch
import h5py

async def record_hdf5(initial_pages, steps, save_path):
    sim = Simulator(initial_pages)
    await sim.setup()

    # probe one step to know shapes
    imgs = await sim.get_images()   # [pages, 3, H, W]
    moves = await sim.move()        # [pages, 6]

    num_pages, c, h, w = imgs.shape
    _, move_dim = moves.shape

    f = h5py.File(save_path, "w")

    f.create_dataset(
        "images",
        shape=(0, num_pages, c, h, w),
        maxshape=(None, num_pages, c, h, w),
        dtype="float32",
        chunks=True,
    )
    f.create_dataset(
        "moves",
        shape=(0, num_pages, move_dim),
        maxshape=(None, num_pages, move_dim),
        dtype="float32",
        chunks=True,
    )

    # append first step
    f["images"].resize(1, axis=0)
    f["moves"].resize(1, axis=0)
    f["images"][0] = imgs.numpy()
    f["moves"][0] = moves.numpy()

    for step in range(1, steps):
        imgs = await sim.get_images()
        moves = await sim.move()

        f["images"].resize(step+1, axis=0)
        f["moves"].resize(step+1, axis=0)

        f["images"][step] = imgs.numpy()
        f["moves"][step] = moves.numpy()

    f.close()


def replay_hdf5(load_path, step_idx):
    with h5py.File(load_path, "r") as f:
        imgs = torch.tensor(f["images"][step_idx])
        moves = torch.tensor(f["moves"][step_idx])
    return imgs, moves



if __name__ == '__main__':
    asyncio.run(record_hdf5(Config.initial_pages, 100_000_000, "dataset.h5"))