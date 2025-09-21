import torch
import h5py
import asyncio
from simulator import Simulator   # your live version

class ReplaySimulator:
    def __init__(self, path):
        self.f = h5py.File(path, "r")
        self.num_steps = self.f["images"].shape[0]
        self.step = 0

    async def setup(self):
        pass  # no-op, just for API compatibility

    async def get_images(self):
        if self.step >= self.num_steps:
            raise IndexError("Replay exhausted")
        imgs = torch.tensor(self.f["images"][self.step])
        return imgs

    async def move(self):
        if self.step >= self.num_steps:
            raise IndexError("Replay exhausted")
        moves = torch.tensor(self.f["moves"][self.step])
        self.step += 1
        return moves

    def close(self):
        self.f.close()
