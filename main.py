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

async def main(sim_url=None, sim_pass=None):
    if Config.is_multi_gpu:
        dist.init_process_group(backend="nccl") # Initializes the distributed process group
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank) # Sets the device for the current process

    model = WorldModel()
    
    if Config.is_multi_gpu:
        model = model.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    #simulator = RemoteSimulator(base_url=sim_url, password=sim_pass)
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

    asyncio.run(main(sim_url=args.sim_url, sim_pass=args.sim_pass))
