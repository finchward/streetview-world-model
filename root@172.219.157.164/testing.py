from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

# Inspect modules
print(vae.encoder)  # should show encoder layers
print(vae.decoder)  # should show decoder layers

# Test with dummy input
import torch

dummy_img = torch.randn(1, 3, 256, 256)  # typical image size
latent = vae.encode(dummy_img).latent_dist.sample()  # output of encoder
print("Encoder output shape:", latent.shape)

recon = vae.decode(latent).sample  # decoder output
print("Decoder output shape:", recon.shape)
