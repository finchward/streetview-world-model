import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config
from diffusers import AutoencoderKL

class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_scene = nn.Sequential(
            nn.Linear(4 * 32 * 32, 1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024)
        )
        self.embed_movement = nn.Linear(6, 64)

        self.predict =  nn.Sequential(
            nn.Linear(Config.hidden_size + 1024 + 64, Config.hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(Config.hidden_size, Config.hidden_size),
            nn.LayerNorm(Config.hidden_size)
        )
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self, scene, movement, prev_state):
        #take in [b, 4, 32, 32], [b, 6], [b, 1024]
        scene = scene.reshape(scene.size(0), -1)
        scene_emb = self.embed_scene(scene)
        movement_emb = self.embed_movement(movement)

        new_state = prev_state + self.predict(torch.cat((prev_state, scene_emb, movement_emb), dim=-1)) 
        return new_state


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        modified_linear = nn.Linear(Config.hidden_size, 6 * Config.hidden_size)
        with torch.no_grad():
            modified_linear.weight[:Config.hidden_size*2].zero_()
            modified_linear.bias[:Config.hidden_size*2].zero_()

        self.condition = nn.Sequential(
            nn.SiLU(),
            modified_linear
        )

        self.attention = nn.MultiheadAttention(Config.hidden_size, Config.heads, dropout=Config.attn_dropout)
        self.feedforward = nn.Sequential(
             nn.Linear(Config.hidden_size, Config.hidden_size * Config.mlp_hidden_size_ratio),
             nn.GELU(),
             nn.Dropout(Config.ffd_dropout),
             nn.Linear(Config.hidden_size * Config.mlp_hidden_size_ratio, Config.hidden_size),
        )

    def forward(self, x, conditioning): #[b, s, 1024], [b, 1, 1024]
        modulation = self.condition(conditioning)
        a1, a2, b1, y1, b2, y2 = torch.split(modulation, Config.hidden_size, dim=-1)    

        x_res = F.layer_norm(
            x,
            Config.hidden_size,
            weight=None,
            bias=None
        )
        x_res = x_res * y1 + b1
        x_res = x_res.transpose(0, 1)
        x_res, _ = self.attention(x_res, x_res, x_res)
        x_res = x_res.transpose(0, 1)
        x = x + x_res * a1

        x_res = F.layer_norm(
            x,
            Config.hidden_size,
            weight=None,
            bias=None
        )
        x_res = x_res * y2 + b2
        x_res = self.feedforward(x_res)
        x = x + x_res * a2

        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patchify = nn.Linear(4 * Config.patch_size * Config.patch_size, Config.hidden_size)

        self.embed_timestep = nn.Sequential(
            nn.Linear(256, Config.hidden_size),
            nn.SiLU(),
            nn.Linear(Config.hidden_size, Config.hidden_size),
            nn.SiLU()
        )
        self.embed_step_size = nn.Embedding(num_embeddings=9, embedding_dim=Config.hidden_size)

        self.transformer_layers = nn.ModuleList([TransformerBlock() for _ in range(Config.layers)])

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(Config.hidden_size, 2 * Config.hidden_size)
        )
        self.depatchify = nn.Linear(Config.hidden_size, 4 * Config.patch_size * Config.patch_size)


    def forward(self, x, timestep, memory, step_size):
        #conditioning consists of: timestep[b], memory[b, 1024]

        #[b, 4, 32, 32]
        base_b, base_c, base_h, base_w = x.shape
        x = x.reshape(base_b, 
                      base_c,
                      base_h // Config.patch_size, 
                      Config.patch_size, 
                      base_w // Config.patch_size, 
                      Config.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)
        b, h_p, w_p, ph, pw, c = x.shape  # b=batch, h_p=w_p=patches, ph=pw=patch height/width, c=channels
        x = x.reshape(b, h_p * w_p, ph * pw * c)

        pos_y, pos_x = torch.meshgrid(
            torch.arange(h_p, device=x.device),
            torch.arange(w_p, device=x.device),
            indexing='ij'
        )
        pos_x = pos_x.reshape(-1)  # [64]
        pos_y = pos_y.reshape(-1)  # [64]

        def fourier_encode(pos, dim=256):
            i = torch.arange(dim // 2, device=pos.device)
            denom = 1/(10_000**(2*i/dim))
            enc = torch.cat([torch.sin(pos.unsqueeze(-1) / denom),
                     torch.cos(pos.unsqueeze(-1) / denom)], dim=-1) #[64, 256]

            return enc  # [64, 256]

        enc_x = fourier_encode(pos_x / (w_p-1), Config.hidden_size//2)
        enc_y = fourier_encode(pos_y / (h_p -1), Config.hidden_size//2)
        pos_emb = torch.cat([enc_x, enc_y], dim=-1)  # [64, 256]
        x = self.patchify(x) #[b, 64, 1024]
        x = x + pos_emb.unsqueeze(0)  

        timestep_emb = fourier_encode(timestep * 1000, dim=256)
        timestep_emb = self.embed_timestep(timestep_emb)

        step_sizes_vocab = torch.tensor([0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1], device=step_size.device)
        eq_matches = torch.isclose(step_size.unsqueeze(-1), step_sizes_vocab.unsqueeze(0), atol=1e-8)
        indices = eq_matches.float().argmax(dim=1)
        step_size_emb = self.embed_step_size(indices)

        conditioning = timestep_emb + memory + step_size_emb
        conditioning = conditioning.unsqueeze(1)

        for layer in self.transformer_layers:
            x = layer(x, conditioning)

        modulation = self.adaLN(conditioning)
        b, y = torch.split(modulation, Config.hidden_size, dim=-1)    

        x = F.layer_norm(
            x,
            Config.hidden_size,
            weight=None,
            bias=None
        )
        x = x * y + b
        x = self.depatchify(x)    

        x = x.reshape(b, h_p, w_p, ph, pw, c)
        x = x.permute(0, 5, 1, 3, 2, 4) 
        x = x.reshape(b, c, base_h, base_w)  

        return x
    

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Transformer()
        self.dynamics = Dynamics()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        encoder = vae.encoder
        decoder = vae.decoder
        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            self.encoder = encoder.eval()
            self.decoder = decoder.eval()
    
    def predict_delta(self, x, timestep, memory, step_size):
        return self.backbone(x, timestep, memory, step_size)
    
    def encode_image(self, x):
        return self.encoder(x)
    
    def decode_image(self, x):
        return self.decoder(x)

    def predict_dynamics(self, scene, movement, prev_state):
        return self.dynamics(scene, movement, prev_state)

def get_model():
    return WorldModel()