import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.condition = nn.Linear(Config.movement_embedding_dim + Config.latent_dimension + Config.time_embedding_dim, out_channels*2)
        #self.memory_conditioning = nn.Linear(Config.movement_embedding_dim + Config.latent_dimension, out_channels)
        #self.time_conditioning = nn.Linear(Config.time_embedding_dim, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x, conditioning):
        fx = self.relu(self.bn1(self.conv1(x))) # test no batchnorm
        fx = self.bn2(self.conv2(fx))

        gamma_beta = self.condition(conditioning)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma_scaled = gamma.unsqueeze(-1).unsqueeze(-1)
        beta_scaled = beta.unsqueeze(-1).unsqueeze(-1)
        #Test doing a layer/groupnorm here before we modulate to be more like adaGN.
        fx = fx * gamma_scaled + beta_scaled

        hx = self.relu(fx + self.shortcut(x))
        return hx

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.condition = nn.Linear(Config.movement_embedding_dim + Config.latent_dimension + Config.time_embedding_dim, out_channels*2)
        #self.memory_conditioning = nn.Linear(Config.movement_embedding_dim + Config.latent_dimension, out_channels)
        #self.time_conditioning = nn.Linear(Config.time_embedding_dim, out_channels)
    
    def forward(self, x, conditioning):
        fx = self.relu(self.bn1(self.conv1(x))) # test no batchnorm
        fx = self.bn2(self.conv2(fx))

        gamma_beta = self.condition(conditioning)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma_scaled = gamma.unsqueeze(-1).unsqueeze(-1)
        beta_scaled = beta.unsqueeze(-1).unsqueeze(-1)
        #Test doing a layer/groupnorm here before we modulate to be more like adaGN.
        fx = fx * gamma_scaled + beta_scaled

        hx = self.relu(fx)
        return hx

class Dynamics(nn.Module):
    def __init__(self, in_channels=3):
        #Give the dynamics the current timestep too - integer, not the float 0-1.
        super().__init__()
        features = Config.features
        layers = []
        for feature in features:
            layers.append(ResBlock(in_channels, feature))
            layers.append(nn.Conv2d(feature, feature, kernel_size=3, stride=2, padding=1))
            in_channels = feature
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down = nn.Sequential(*layers)

        self.project_img = nn.Linear(features[-1], Config.latent_dimension)
        self.predict = nn.Linear(Config.latent_dimension*2, Config.latent_dimension)

    def forward(self, img, prev_state):
        latent_img = self.down(img).squeeze(-1).squeeze(-1)
        latent_img = self.project_img(latent_img)
        state = torch.cat((latent_img, prev_state), dim=1)
        new_state = self.predict(state)
        return new_state


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        features = Config.features
        self.downs = nn.ModuleList()
        self.strides = nn.ModuleList()
        for feature in features:
            self.downs.append(ResBlock(in_channels, feature))
            self.strides.append(nn.Conv2d(feature, feature, kernel_size=3, stride=2, padding=1))
            in_channels = feature
        
        self.bottleneck = ResBlock(features[-1], features[-1]*2)
        
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        rev_features = features[::-1]
        for feature in rev_features:
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.up_convs.append(ConvBlock(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.embed_movement = nn.Linear(6, Config.movement_embedding_dim)

        
    def forward(self, x, t, m, l_emb):
        skip_connections = []

        #Embedding time step
        t = t.unsqueeze(-1) #[num_dim, 1]
        half_time_dim = Config.time_embedding_dim // 2
        freq_exponents = torch.arange(half_time_dim, dtype=torch.float32, device=t.device)
        freq_exponents = -math.log(10000) * freq_exponents / half_time_dim
        freqs = torch.exp(freq_exponents) 
        freqs = freqs.unsqueeze(0)
        angles = t * freqs  
        t_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        m_emb = self.embed_movement(m)
        conditioning_emb = torch.cat([m_emb, l_emb, t_emb], dim=1)

        for down, stride in zip(self.downs, self.strides):
            x = down(x, conditioning_emb)
            skip_connections.append(x)
            x = stride(x)

            #Maybe do conditioning here
        
        x = self.bottleneck(x) # add transformer layers here.
        
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.ups)):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            x = torch.cat([skip_connection, x], dim=1)
            x = self.up_convs[idx](x, conditioning_emb)
        
        return self.final_conv(x) #delta predictions in each channel.

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = UNet()
        self.dynamics = Dynamics()
    
    def predict_delta(self, x, t, m, l_emb):
        return self.backbone(x, t, m, l_emb)

    def predict_dynamics(self, img, prev_state):
        return self.dynamics(img, prev_state)


def get_model():
    return WorldModel()