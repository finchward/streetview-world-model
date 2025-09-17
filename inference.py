import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
from pathlib import Path
from config import Config
# Distributed imports
import torch.distributed as dist

fig, axes, ims = None, None, None
if Config.is_colab:
    from IPython.display import display, clear_output
    display_handle = None
    from google.colab import drive
    drive.mount('/content/drive')

@torch.no_grad()
def sample_next_img(model, device, sample_name, prev_img, movement, latent, next_img=None):
    model.eval()
    base_img = prev_img.clone()
    
    # Check if we are the main process (rank 0)
    is_main_process = (not Config.is_multi_gpu) or (dist.get_rank() == 0)

    if is_main_process:
        global fig, axes, ims
        if Config.is_colab:
            global display_handle

        if fig is None:
            if not Config.is_colab:
                plt.ion()
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            ims = []
            for i, title in enumerate(['Base', 'Prediction', 'Target']):
                ims.append(axes[i].imshow(np.zeros((Config.model_resolution[0], Config.model_resolution[1], 3), dtype=np.float32)))
                axes[i].axis("off")
                axes[i].set_title(title)
            if not Config.is_colab:
                plt.tight_layout()
                plt.show()
            
            base = np.transpose(base_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
            target = np.transpose(next_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
            ims[0].set_data(np.clip(base, 0, 1))
            ims[2].set_data(np.clip(target, 0, 1))
    
    out_img = None
    pbar = tqdm.tqdm(range(Config.inference_samples), desc=f'Sampling')
    for time_step in pbar:
        time_step = torch.tensor([time_step]).to(device)
        dx = Config.inference_step_size / Config.inference_samples
        if Config.is_multi_gpu:
            delta = model.module.predict_delta(prev_img, time_step, movement, latent)
        else:
            delta = model.predict_delta(prev_img, time_step, movement, latent)
        prev_img += delta * dx
        prev_img = torch.clamp(prev_img, 0, 1)
        
        # Only plot and display from the main process
        if is_main_process and (not Config.is_colab or (time_step % Config.inference_display_update_freq == 0 or time_step == Config.inference_samples - 1)):
            out_img = prev_img.squeeze(0).detach().cpu().numpy()
            out_img = np.transpose(out_img, (1, 2, 0))
            ims[1].set_data(out_img)

            if Config.is_colab:
                clear_output(wait=True)
                display(fig)
            else:
                fig.canvas.draw()
                fig.canvas.flush_events()
    
    # Only save images from the main process
    if is_main_process:
            if Config.is_colab:
                current_dir = Path.cwd()
            else:   
                current_dir = Path(Config.drive_dir)
        model_dir = current_dir / Path(Config.img_dir) / Config.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        save_path = model_dir / f"sample_{sample_name}.png"
        plt.imsave(save_path, np.clip(out_img, 0, 1))

        target = np.transpose(next_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
        save_path = model_dir / f"sample_{sample_name}_target.png"
        plt.imsave(save_path, np.clip(target, 0, 1))

        base = np.transpose(base_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
        save_path = model_dir / f"sample_{sample_name}_base.png"
        plt.imsave(save_path, np.clip(base, 0, 1))