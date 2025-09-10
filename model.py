import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    def __init__(self, initial_hidden_dim=32,  image_dim=256, latent_dim=512):
        super().__init__()
        num_layers = int(math.log2(image_dim)) - 2
        layers = [
            nn.Conv2d(in_channels=3, out_channels=initial_hidden_dim, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(initial_hidden_dim),
            nn.LeakyReLU(inplace=True)
            ]

        for layer_num in range(num_layers - 1):
            layer_input_dim = initial_hidden_dim * (2 ** layer_num)
            layers.extend([
                nn.Conv2d(in_channels=layer_input_dim, out_channels=layer_input_dim*2,  padding=1, kernel_size=3, stride=2),
                nn.BatchNorm2d(layer_input_dim*2),
                nn.LeakyReLU(inplace=True)
               ])


        self.encode = nn.Sequential(*layers)
        final_size = initial_hidden_dim * 2 ** (num_layers-1) * 4 * 4
        self.mu_head = nn.Linear(final_size, latent_dim)
        self.logvar_head = nn.Linear(final_size, latent_dim)
    
    def forward(self, x):
        hidden_state = self.encode(x) # [batch_size, hidden_dim, 4, 4]
        hidden_state = hidden_state.view(hidden_state.size(0), -1)
        mu = self.mu_head(hidden_state)
        logvar = self.logvar_head(hidden_state)
        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self, initial_hidden_dim=32, image_dim=256, latent_dim=512):
        super().__init__()
        num_layers = int(math.log2(image_dim)) - 2
        encoder_dim = initial_hidden_dim * 2 ** (num_layers-1) * 4 * 4
        split_dim = encoder_dim // 16
        self.project = nn.Linear(latent_dim, encoder_dim)
        layers = []
        for layer_num in range(num_layers-1):
            layer_input_dim = split_dim // (2**layer_num)
            layer_output_dim = layer_input_dim // 2
            layers.extend([
                nn.ConvTranspose2d(in_channels=layer_input_dim, out_channels=layer_output_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(layer_output_dim),
                nn.LeakyReLU(inplace=True),
            ])
        decoder_dim = split_dim // (2**(num_layers-1))
        layers.extend([
            nn.ConvTranspose2d(in_channels=decoder_dim, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        ])
        self.decode = nn.Sequential(*layers)


    def forward(self, x):
        projected = self.project(x) # [batch_num, hidden_dim*4*4]
        projected = projected.view(projected.size(0), -1, 4, 4)
        prediction = self.decode(projected)
        return prediction



class Vae(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config.initial_hidden_dim, config.image_resolution, config.latent_dim)
        self.decoder = Decoder(config.initial_hidden_dim, config.image_resolution, config.latent_dim)
    
    def forward(self, in_img):
        mu, logvar = self.encoder(in_img)
        epsilon = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * epsilon
        out_img = self.decoder(z)
        return out_img, mu, logvar

def get_vae(config):
    return Vae(config)

def get_decoder(config):
    return Decoder(config.initial_hidden_dim, config.image_resolution, config.latent_dim)
