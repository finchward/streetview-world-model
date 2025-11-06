from config import Config
from transformer import WorldModel
from train import Trainer
from replaysim import ReplaySimulator
from simulator import Simulator
import asyncio
import argparse
import torch.nn as nn

async def main(sim_url=None, sim_pass=None):
    model = WorldModel()

    #simulator = RemoteSimulator(base_url=sim_url, password=sim_pass)
    train_simulator = ReplaySimulator(dataset_dir="dataset/webdataset_sharded", num_parallel_sequences=Config.batch_size)
    val_simulator = ReplaySimulator(dataset_dir="dataset/webdataset_sharded_val", num_parallel_sequences=1)
    
    await train_simulator.setup()
    await val_simulator.setup()
    trainer = Trainer(model, train_simulator, val_simulator)
    if Config.load_model:
        trainer.load_checkpoint()
    await trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training with remote simulator")
    parser.add_argument("--sim_url", type=str, default="https://1758a8c5b1ae.ngrok-free.app", help="Simulator URL")
    parser.add_argument("--sim_pass", type=str, default="6Lw9;Q8BUSAnOCDQKBQAOZ/qsBR.", help="Simulator password")
    args = parser.parse_args()

    asyncio.run(main(sim_url=args.sim_url, sim_pass=args.sim_pass))