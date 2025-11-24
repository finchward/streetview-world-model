from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

# Test with dummy input
import torch

dummy_img = torch.randn(1, 3, 384, 512)  # typical image size
latent = vae.encode(dummy_img).latent_dist.sample()  # output of encoder
print("Encoder output shape:", latent.shape)

recon = vae.decode(latent).sample  # decoder output
print("Decoder output shape:", recon.shape)
