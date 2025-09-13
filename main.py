from config import Config
from model import WorldModel
from train import Trainer
from simulator import Simulator

def main():
    model = WorldModel()
    simulator = Simulator()
    simulator.setup()
    trainer = Trainer(model, simulator)
    if Config.load_model:
        trainer.load_checkpoint()
    trainer.train()

if __name__ == '__main__':
    main()