import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_conditioned=True, squeeze_factor=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=Config.group_size,  num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=Config.group_size,  num_channels=out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.is_conditioned = is_conditioned
        if is_conditioned:
            self.condition = nn.Linear(Config.movement_embedding_dim + Config.latent_dimension + Config.time_embedding_dim, out_channels*2)
        #self.memory_conditioning = nn.Linear(Config.movement_embedding_dim + Config.latent_dimension, out_channels)
        #self.time_conditioning = nn.Linear(Config.time_embedding_dim, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.is_squeezing = squeeze_factor != None
        if squeeze_factor:
            self.squeeze = nn.AdaptiveAvgPool2d((1,1))
            self.excite = nn.Sequential(
                nn.Linear(out_channels, out_channels//squeeze_factor),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels//squeeze_factor, out_channels),
                nn.Sigmoid()
            )

    
    def forward(self, x, conditioning=None):
        fx = self.relu(self.bn1(self.conv1(x))) # test no batchnorm
        fx = self.bn2(self.conv2(fx))

        if self.is_conditioned:
            gamma_beta = self.condition(conditioning)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            gamma_scaled = gamma.unsqueeze(-1).unsqueeze(-1)
            beta_scaled = beta.unsqueeze(-1).unsqueeze(-1)
            #Test doing a layer/groupnorm here before we modulate to be more like adaGN.
            fx = fx * gamma_scaled + beta_scaled

        hx = self.relu(fx + self.shortcut(x))
        if self.is_squeezing:
            hx = hx * self.excite(self.squeeze(hx).squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
        return hx

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, squeeze_factor=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=Config.group_size,  num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=Config.group_size,  num_channels=out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.condition = nn.Linear(Config.movement_embedding_dim + Config.latent_dimension + Config.time_embedding_dim, out_channels*2)

        self.is_squeezing = squeeze_factor != None
        if squeeze_factor:
            self.squeeze = nn.AdaptiveAvgPool2d((1,1))
            self.excite = nn.Sequential(
                nn.Linear(out_channels, out_channels//squeeze_factor),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels//squeeze_factor, out_channels),
                nn.Sigmoid()
            )

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

        if self.is_squeezing:
            hx = hx * self.excite(self.squeeze(hx).squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
        return hx

class AttentionGate(nn.Module):
    def __init__(self, decoder_channels, skip_channels, hidden_channels):
        super().__init__()
        self.q = nn.Conv2d(decoder_channels, hidden_channels, kernel_size=1)
        self.k = nn.Conv2d(skip_channels, hidden_channels, kernel_size=1)
        self.attn = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, encoder_features, decoder_features):
        q = self.q(decoder_features)
        k = self.k(encoder_features)
        a = self.attn(q + k)
        return a * encoder_features
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, batch_first=True, num_heads=num_heads)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, sequence):
        attn_output, _ = self.attn(sequence, sequence, sequence) 
        out = self.norm(sequence + attn_output)
        return out


class Dynamics(nn.Module):
    def __init__(self, in_channels=3):
        #Give the dynamics the current timestep too - integer, not the float 0-1.
        super().__init__()
        features = Config.features
        layers = []
        for feature in features:
            layers.append(ResBlock(in_channels, feature, is_conditioned=False))
            layers.append(nn.Conv2d(feature, feature, kernel_size=3, stride=2, padding=1))
            in_channels = feature
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down = nn.Sequential(*layers)

        self.project_img = nn.Linear(features[-1], Config.latent_dimension)

        self.predict =  nn.Sequential(
            nn.Linear(Config.latent_dimension*2, Config.latent_dimension*2),
            nn.ReLU(),
            nn.Linear(Config.latent_dimension*2, Config.latent_dimension)
        )

    def forward(self, img, prev_state):
        latent_img = self.down(img).squeeze(-1).squeeze(-1)
        latent_img = self.project_img(latent_img)
        new_state = prev_state + self.predict(torch.cat((prev_state, latent_img), dim=-1)) 
        new_state = F.normalize(new_state, dim=1, p=2)
        return new_state

def get_2d_sin_cos_positional_encoding(H, W, C, device):
    assert C % 4 == 0, "C must be divisible by 4"

    c_half = C // 2
    c_quarter = C // 4

    # Y-axis
    y_pos = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)  # [H, 1]
    div_term = torch.exp(torch.arange(0, c_half, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / c_half))
    pe_y = torch.zeros(H, c_half, device=device)  # [H, C/2]
    pe_y[:, 0::2] = torch.sin(y_pos * div_term)
    pe_y[:, 1::2] = torch.cos(y_pos * div_term)
    pe_y = pe_y.unsqueeze(1).repeat(1, W, 1)  # [H, W, C/2]

    # X-axis
    x_pos = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1)  # [W, 1]
    pe_x = torch.zeros(W, c_half, device=device)
    pe_x[:, 0::2] = torch.sin(x_pos * div_term)
    pe_x[:, 1::2] = torch.cos(x_pos * div_term)
    pe_x = pe_x.unsqueeze(0).repeat(H, 1, 1)  # [H, W, C/2]

    pe = torch.cat([pe_y, pe_x], dim=-1)  # [H, W, C]
    pe = pe.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    return pe

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        features = Config.features
        self.downs = nn.ModuleList()
        self.strides = nn.ModuleList()
        for feature in features:
            self.downs.append(ResBlock(in_channels, feature, squeeze_factor=Config.squeeze_factor))
            self.strides.append(nn.Conv2d(feature, feature, kernel_size=3, stride=2, padding=1))
            in_channels = feature
        
        self.bottleneck = ResBlock(features[-1], features[-1]*2)
        self.self_attn = SelfAttention(features[-1]*2, Config.bottleneck_heads)
        self.self_attn2 = SelfAttention(features[-1]*2, Config.bottleneck_heads)
        
        self.ups = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        rev_features = features[::-1]
        for feature in rev_features:
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.up_attns.append(AttentionGate(feature, feature, feature//2))
            self.up_convs.append(ConvBlock(feature*2, feature, squeeze_factor=Config.squeeze_factor))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.embed_movement = nn.Sequential(
            nn.Linear(6, Config.movement_embedding_dim),
            nn.ReLU(),
            nn.Linear(Config.movement_embedding_dim, Config.movement_embedding_dim)
        )

        
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
        
        x = self.bottleneck(x, conditioning_emb)
        pos_enc = get_2d_sin_cos_positional_encoding(x.shape[2], x.shape[3], x.shape[1], x.device)
        x = x + pos_enc
        dim = x.shape
        x_flattened = x.view(dim[0], dim[1], -1)
        x_flattened = x_flattened.transpose(1, 2) #[b, s, c]
        x_flattened = self.self_attn(x_flattened)
        x_flattened = self.self_attn2(x_flattened)
        x_flattened = x_flattened.transpose(1, 2)
        x = x_flattened.view(dim[0], dim[1], dim[2], dim[3])        

        skip_connections = skip_connections[::-1]
        for idx in range(len(self.ups)):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            skip_attended = self.up_attns[idx](skip_connection, x)
            
            x = torch.cat([skip_attended, x], dim=1)
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