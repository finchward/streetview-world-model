import torch
import matplotlib
# Set a non-interactive backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
from pathlib import Path
from config import Config
# Distributed imports
import torch.distributed as dist

fig, axes, ims = None, None, None


@torch.no_grad()
def sample_next_img(model, device, sample_name, prev_img, movement, latent, next_img=None):
    model.eval()

    if Config.from_noise:
        starting_img = torch.randn_like(prev_img)
    else:
        starting_img = prev_img.clone()

    # Determine if we are the main process
    is_main_process = (not Config.is_multi_gpu) or (dist.get_rank() == 0)
    if not is_main_process:
        return

    out_img = None

    # Sampling loop
    for time_step in range(Config.inference_samples):
        time_tensor = torch.tensor([time_step/Config.inference_samples]).to(device)
        dx = Config.inference_step_size / Config.inference_samples
        
        if Config.is_multi_gpu:
            delta = model.module.predict_delta(starting_img, time_tensor, movement, latent)
        else:
            delta = model.predict_delta(starting_img, time_tensor, movement, latent)
        
        starting_img += delta * dx
        starting_img = torch.clamp(starting_img, 0, 1)

    # Convert the final predicted image to numpy
    out_img = starting_img.squeeze(0).detach().cpu().numpy()
    out_img = np.transpose(out_img, (1, 2, 0))

    model_dir = Path.cwd() / Config.img_dir / Config.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the predicted image
    save_path_pred = model_dir / f"sample_{sample_name}.png"
    plt.imsave(save_path_pred, np.clip(out_img, 0, 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    base = np.transpose(prev_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
    target = np.transpose(next_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0))

    for ax, img, title in zip(axes, [base, out_img, target], ['Base', 'Prediction', 'Target']):
        ax.imshow(np.clip(img, 0, 1))
        ax.axis("off")
        ax.set_title(title)

    save_path_fig = model_dir / f"sample_{sample_name}_all.png"
    plt.tight_layout()
    plt.savefig(save_path_fig)
    plt.close(fig)  # Close figure to free memory