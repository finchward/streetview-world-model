from config import Config
from data import get_data_loaders
from model import get_vae, get_decoder
from train import Trainer

def main():
    train_loader, val_loader = get_data_loaders(Config)
    model = get_vae(Config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    trainer = Trainer(model, train_loader, val_loader, Config)
    if Config.use_checkpoint:
        trainer.load_checkpoint(Config)
    trainer.train()

if __name__ == '__main__':
    main()