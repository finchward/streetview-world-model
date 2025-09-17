import os
from config import Config
from model import WorldModel
from train import Trainer
from remote_simulator import RemoteSimulator
from simulator import Simulator
import asyncio
import argparse
import torch.nn as nn
import torch.distributed as dist
import torch.nn.parallel
import torch.multiprocessing as mp
import signal
import sys

def main_ddp(rank, world_size, sim_url=None, sim_pass=None):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    def signal_handler(sig, frame):
        print(f"\nRank {rank} caught Ctrl+C. Destroying process group...")
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(0)

    model = WorldModel()
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # The asyncio part needs to be run within the DDP process
    asyncio.run(async_main(model, sim_url, sim_pass))

async def async_main(model, sim_url=None, sim_pass=None):
    # This is the original content of the async main function
    # The model is now passed in as an argument
    # simulator = RemoteSimulator(base_url=sim_url, password=sim_pass)
    simulator = Simulator(Config.initial_pages)
    await simulator.setup()
    trainer = Trainer(model, simulator)
    if Config.load_model:
        trainer.load_checkpoint()
    await trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training with remote simulator")
    parser.add_argument("--sim_url", type=str, default="https://1758a8c5b1ae.ngrok-free.app", help="Simulator URL")
    parser.add_argument("--sim_pass", type=str, default="6Lw9;Q8BUSAnOCDQKBQAOZ/qsBR.", help="Simulator password")
    args = parser.parse_args()

    if Config.is_multi_gpu:
        print("Using DistributedDataParallel (DDP) for multi-GPU training.")
        world_size = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(main_ddp,
                 args=(world_size, args.sim_url, args.sim_pass),
                 nprocs=world_size,
                 join=True)
    else:
        print("Using single GPU or CPU.")
        model = WorldModel()
        asyncio.run(async_main(model, sim_url=args.sim_url, sim_pass=args.sim_pass))