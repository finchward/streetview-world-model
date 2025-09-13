from config import Config
from model import WorldModel
from train import Trainer
from simulator import Simulator
import asyncio

async def main():
    try:
        model = WorldModel()
        simulator = Simulator()
        await simulator.setup()
        trainer = Trainer(model, simulator)
        if Config.load_model:
            trainer.load_checkpoint()
        await trainer.train()
    finally:
        await simulator.close()

if __name__ == '__main__':
    asyncio.run(main())