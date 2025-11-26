import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import traceback


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
            (Config.hidden_size,),
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
            (Config.hidden_size,),
            weight=None,
            bias=None
        )
        x_res = x_res * y2 + b2
        x_res = self.feedforward(x_res)
        x = x + x_res * a2

        return x

def fourier_encode(pos, dim=256):
    i = torch.arange(dim // 2, device=pos.device)
    denom = torch.pow(1e4, -2 * i / dim)
    enc = torch.cat([torch.sin(pos.unsqueeze(-1) * denom),
                     torch.cos(pos.unsqueeze(-1) * denom)], dim=-1)
    return enc

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patchify = nn.Linear(4 * Config.patch_size * Config.patch_size, Config.hidden_size)

        self.embed_timestep = nn.Sequential(
            nn.Linear(256, Config.hidden_size),
            nn.SiLU(),
            nn.Linear(Config.hidden_size, Config.hidden_size),
        )
        self.embed_step_size = nn.Embedding(num_embeddings=9, embedding_dim=Config.hidden_size)

        self.transformer_layers = nn.ModuleList([TransformerBlock() for _ in range(Config.layers)])

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(Config.hidden_size, 2 * Config.hidden_size)
        )
        self.depatchify = nn.Linear(Config.hidden_size, 4 * Config.patch_size * Config.patch_size)

        self.classifier_token = nn.Embedding(num_embeddings=1, embedding_dim=Config.hidden_size)
        self.rnn_layers = nn.ModuleList([TransformerBlock() for _ in range(Config.memory_layers)])
        self.embed_movement = nn.Sequential(
            nn.Linear(6, Config.hidden_size),
            nn.SiLU(),
            nn.Linear(Config.hidden_size, Config.hidden_size),
        )
        self.norm_latent = nn.LayerNorm(Config.hidden_size)
                                            

    def img_to_tokens(self, x):
        base_b, base_c, base_h, base_w = x.shape
        x = x.reshape(base_b, 
                        base_c,
                        base_h // Config.patch_size, 
                        Config.patch_size, 
                        base_w // Config.patch_size, 
                        Config.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)
        b, h_p, w_p, ph, pw, c = x.shape
        x = x.reshape(b, h_p * w_p, ph * pw * c)

        pos_y, pos_x = torch.meshgrid(
            torch.arange(h_p, device=x.device),
            torch.arange(w_p, device=x.device),
            indexing='ij'
        )
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        enc_x = fourier_encode(pos_x / (w_p-1), Config.hidden_size//2)
        enc_y = fourier_encode(pos_y / (h_p -1), Config.hidden_size//2)
        pos_emb = torch.cat([enc_x, enc_y], dim=-1)  
        x = self.patchify(x)
        x = x + pos_emb.unsqueeze(0)  
        return x

    def forward(self, x, timestep, memory, step_size):
        try:
            base_b, base_c, base_h, base_w = x.shape
            x = self.img_to_tokens(x)
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
            beta, gamma = torch.split(modulation, Config.hidden_size, dim=-1)    

            x = F.layer_norm(
                x,
                (Config.hidden_size,),
                weight=None,
                bias=None
            )
            x = x * gamma + beta
            x = self.depatchify(x)    

            x = x.reshape(base_b, base_h // Config.patch_size, base_w // Config.patch_size, Config.patch_size, Config.patch_size, base_c)
            x = x.permute(0, 5, 1, 3, 2, 4) 
            x = x.reshape(base_b, base_c, base_h, base_w)  

            return x
        except Exception as e:
            print("Error:", e)
            traceback.print_exc()

    def dynamics(self, scene, movement, prev_state):
        scene_tokens = self.img_to_tokens(scene)
        conditioning = self.embed_movement(movement).unsqueeze(1)
        tokens = torch.cat([prev_state.unsqueeze(1), scene_tokens], dim=1)
        for layer in self.rnn_layers:
            tokens = layer(tokens, conditioning)
        new_state = tokens[:, 0, :]
        new_state = new_state + prev_state
        new_state = self.norm_latent(new_state)
        return new_state
    

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Transformer()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse") 
        for param in vae.parameters():
            param.requires_grad = False
        self.vae = vae.eval()
        self.image_processor = VaeImageProcessor(do_normalize=True)
    
    def predict_delta(self, x, timestep, memory, step_size):
        return self.backbone(x, timestep, memory, step_size)
    
    def encode_image(self, x):
        x = self.image_processor.preprocess(x, height=Config.model_resolution[0], width=Config.model_resolution[1])
        latent = self.vae.encode(x).latent_dist.mode()
        latent = latent * 0.18215
        return latent
    
    def decode_image(self, x):
        x = x / 0.18215
        x = self.vae.decode(x).sample
        x = self.image_processor.postprocess(x, output_type="pt", do_denormalize=[True])
        return x

    def predict_dynamics(self, scene, movement, prev_state):
        return self.backbone.dynamics(scene, movement, prev_state)

def get_model():
    return WorldModel()