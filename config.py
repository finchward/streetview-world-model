from dataclasses import dataclass

@dataclass
class Config:
    batch_size:int = 96
    data_loader_workers:int = 3

    #Training
    learning_rate:float = 1e-4
    weight_decay:float = 1e-4
    validation_dataset_size = 192
    num_epochs = 3000

    kl_warmup_steps = 10_000
    kl_coefficient = 0.03

    #Model specifications
    image_resolution:int = 128 #Power of two
    initial_hidden_dim:int = 64
    latent_dim:int = 32

    model_name:str = 'res128_latent256_klwarmup_2'

    validate_every_x = 50

    use_checkpoint = False
    loaded_model_name = 'res128_latent256'
    loaded_checkpoint = 'epoch_430.pth'
    save_every_x_epoch = 10

    graph_update_every = 5
    graph_grouped_loss_size = 1
    graph_recent_losses_shown = 2000

    img_dir = 'images'